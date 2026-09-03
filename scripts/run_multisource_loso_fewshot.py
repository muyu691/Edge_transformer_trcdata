#!/usr/bin/env python
"""Multi-source leave-one-network-out training and few-shot adaptation.

The transfer protocol is label-free at normalization time: each domain scaler
is fitted from old flows and known edge attributes only. Existing processed
PyG tensors are first inverted with their legacy scaler, then re-normalized
with the input-only scaler used by this experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import os.path as osp
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import graphgps  # noqa: E402,F401
from constraint_violation.metrics import ConstraintViolationAccumulator  # noqa: E402
from graphgps.loss.flow_conservation_loss import compute_pinn_loss  # noqa: E402
from graphgps.utils import match_edge_indices  # noqa: E402
from evaluate_zero_shot import MetricBundle, sanitize_for_json  # noqa: E402


NETWORKS = {
    "siouxfalls": "SiouxFalls",
    "ema": "EMA",
    "anaheim": "Anaheim",
}


@dataclass(frozen=True)
class DomainNormalizer:
    network: str
    flow_mean: float
    flow_std: float
    flow_count: int
    attr_mean: tuple[float, ...]
    attr_std: tuple[float, ...]
    attr_count: int
    legacy_flow_mean: float
    legacy_flow_std: float
    legacy_attr_mean: tuple[float, ...]
    legacy_attr_std: tuple[float, ...]
    fitted_from: str = "train.flow_old + train.edge_attr_old/new; no flow_new labels"


class StreamingMoments:
    """Numerically stable population moments for a fixed feature dimension."""

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.count = 0
        self.sum = np.zeros(self.width, dtype=np.float64)
        self.sum_sq = np.zeros(self.width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if array.size == 0:
            return
        self.count += int(array.shape[0])
        self.sum += array.sum(axis=0)
        self.sum_sq += np.square(array).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.count <= 0:
            raise ValueError("Cannot fit moments from an empty input.")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        return mean, scale, self.count


def canonical_network(value: str) -> str:
    key = value.strip().lower()
    if key not in NETWORKS:
        raise ValueError(f"Unknown network '{value}'. Expected one of {tuple(NETWORKS.values())}.")
    return NETWORKS[key]


def network_key(value: str) -> str:
    name = canonical_network(value)
    return name.lower()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_split(dataset_dir: Path, split: str) -> list:
    path = dataset_dir / f"{split}_dataset.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    data = torch_load(path)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of PyG Data objects in {path}, got {type(data).__name__}.")
    logging.info("Loaded %s split from %s: %d graphs", split, path, len(data))
    return data


def load_legacy_scalers(dataset_dir: Path) -> tuple[object, object]:
    attr_path = dataset_dir / "scalers" / "attr_scaler.pkl"
    flow_path = dataset_dir / "scalers" / "flow_scaler.pkl"
    if not attr_path.exists() or not flow_path.exists():
        raise FileNotFoundError(f"Missing legacy scalers below {dataset_dir / 'scalers'}")
    return load_pickle(attr_path), load_pickle(flow_path)


def _scaler_vector(scaler, field: str) -> np.ndarray:
    value = np.asarray(getattr(scaler, field), dtype=np.float64).reshape(-1)
    if value.size == 0:
        raise ValueError(f"Scaler field {field} is empty.")
    return value


def fit_input_only_normalizer(
    network: str,
    train_data: list,
    legacy_attr_scaler,
    legacy_flow_scaler,
) -> DomainNormalizer:
    """Fit using target/source inputs only; y is deliberately never read."""
    legacy_flow_mean = float(_scaler_vector(legacy_flow_scaler, "mean_")[0])
    legacy_flow_std = float(max(_scaler_vector(legacy_flow_scaler, "scale_")[0], 1e-12))
    legacy_attr_mean = _scaler_vector(legacy_attr_scaler, "mean_")
    legacy_attr_std = np.maximum(_scaler_vector(legacy_attr_scaler, "scale_"), 1e-12)

    flow_moments = StreamingMoments(1)
    attr_moments = StreamingMoments(int(legacy_attr_mean.size))
    for graph in train_data:
        raw_old_flow = graph.flow_old.detach().cpu().numpy() * legacy_flow_std + legacy_flow_mean
        flow_moments.update(raw_old_flow)

        for field in ("edge_attr_old", "edge_attr_new"):
            normalized_attr = getattr(graph, field).detach().cpu().numpy()
            raw_attr = normalized_attr * legacy_attr_std + legacy_attr_mean
            attr_moments.update(raw_attr)

    flow_mean, flow_std, flow_count = flow_moments.finish()
    attr_mean, attr_std, attr_count = attr_moments.finish()
    normalizer = DomainNormalizer(
        network=canonical_network(network),
        flow_mean=float(flow_mean[0]),
        flow_std=float(flow_std[0]),
        flow_count=int(flow_count),
        attr_mean=tuple(float(value) for value in attr_mean),
        attr_std=tuple(float(value) for value in attr_std),
        attr_count=int(attr_count),
        legacy_flow_mean=legacy_flow_mean,
        legacy_flow_std=legacy_flow_std,
        legacy_attr_mean=tuple(float(value) for value in legacy_attr_mean),
        legacy_attr_std=tuple(float(value) for value in legacy_attr_std),
    )
    logging.info(
        "%s input-only scaler: flow mean=%.6f std=%.6f count=%d; attr mean=%s std=%s",
        normalizer.network,
        normalizer.flow_mean,
        normalizer.flow_std,
        normalizer.flow_count,
        normalizer.attr_mean,
        normalizer.attr_std,
    )
    return normalizer


def renormalize_graph_in_place(graph, normalizer: DomainNormalizer) -> None:
    """Exact legacy->real->input-only conversion for inputs and evaluation y."""
    legacy_flow_mean = graph.flow_old.new_tensor(normalizer.legacy_flow_mean)
    legacy_flow_std = graph.flow_old.new_tensor(normalizer.legacy_flow_std)
    flow_mean = graph.flow_old.new_tensor(normalizer.flow_mean)
    flow_std = graph.flow_old.new_tensor(normalizer.flow_std)

    raw_old = graph.flow_old * legacy_flow_std + legacy_flow_mean
    raw_target = graph.y * legacy_flow_std + legacy_flow_mean
    graph.flow_old = ((raw_old - flow_mean) / flow_std).to(torch.float32)
    graph.y = ((raw_target - flow_mean) / flow_std).to(torch.float32)

    for field in ("edge_attr_old", "edge_attr_new"):
        value = getattr(graph, field)
        legacy_mean = value.new_tensor(normalizer.legacy_attr_mean)
        legacy_std = value.new_tensor(normalizer.legacy_attr_std)
        new_mean = value.new_tensor(normalizer.attr_mean)
        new_std = value.new_tensor(normalizer.attr_std)
        raw_attr = value * legacy_std + legacy_mean
        setattr(graph, field, ((raw_attr - new_mean) / new_std).to(torch.float32))


def renormalize_split(data: list, normalizer: DomainNormalizer) -> None:
    for graph in data:
        renormalize_graph_in_place(graph, normalizer)


def dataset_dir_for(data_root: Path, network: str) -> Path:
    return data_root / f"{network_key(network)}_pyg_baseline_perturb"


def configure_graphgym(args: argparse.Namespace, initial_normalizer: DomainNormalizer) -> None:
    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=args.cfg, opts=[]))
    cfg.seed = int(args.seed)
    cfg.accelerator = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    cfg.wandb.use = False
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.current_epoch = 0
    cfg.dataset.network_name = initial_normalizer.network
    cfg.dataset.flow_mean = float(initial_normalizer.flow_mean)
    cfg.dataset.flow_std = float(initial_normalizer.flow_std)
    cfg.share.dim_in = 1
    cfg.share.dim_out = 1
    cfg.share.num_splits = 3
    torch.set_num_threads(int(args.num_threads))


def inner_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def activate_domain(
    model: torch.nn.Module,
    normalizer: DomainNormalizer,
    network: str,
) -> None:
    cfg.dataset.network_name = canonical_network(network)
    cfg.dataset.flow_mean = float(normalizer.flow_mean)
    cfg.dataset.flow_std = float(normalizer.flow_std)
    target = inner_model(model)
    if not hasattr(target, "flow_mean") or not hasattr(target, "flow_std"):
        raise AttributeError("Topology model does not expose flow_mean/flow_std buffers.")
    with torch.no_grad():
        target.flow_mean.fill_(float(normalizer.flow_mean))
        target.flow_std.fill_(float(normalizer.flow_std))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_new_edge_mask(batch) -> torch.Tensor:
    match_idx = match_edge_indices(
        edge_index_old=batch.edge_index_old,
        edge_index_new=batch.edge_index_new,
        total_nodes=batch.num_nodes,
    )
    return match_idx < 0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    normalizer: DomainNormalizer,
    network: str,
    device: torch.device,
) -> dict:
    activate_domain(model, normalizer, network)
    model.eval()
    metrics = MetricBundle()
    constraints = ConstraintViolationAccumulator()
    total_graphs = 0
    total_forward = 0.0

    for batch in loader:
        batch.split = "test"
        batch = batch.to(device)
        total_graphs += int(batch.num_graphs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred, true = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_forward += time.perf_counter() - start

        pred_norm = pred.detach().cpu().float().view(-1)
        true_norm = true.detach().cpu().float().view(-1)
        pred_real = pred_norm * normalizer.flow_std + normalizer.flow_mean
        true_real = true_norm * normalizer.flow_std + normalizer.flow_mean
        if hasattr(batch, "new_edge_mask"):
            new_mask = batch.new_edge_mask.detach().cpu().bool().view(-1)
        else:
            new_mask = compute_new_edge_mask(batch).detach().cpu().bool().view(-1)
        metrics.update(pred_real, true_real, new_mask)
        constraints.update(
            pred_real=pred_real,
            edge_index_new=batch.edge_index_new.detach().cpu(),
            net_demand=batch.net_demand.detach().cpu(),
            node_batch=batch.batch.detach().cpu(),
            ptr=batch.ptr.detach().cpu(),
        )

    if total_graphs == 0:
        raise RuntimeError(f"No graphs evaluated for {network}.")
    real_metrics = metrics.as_dict()
    result = {
        "metrics_real": real_metrics,
        "constraint_violation": constraints.as_dict(),
        "timing": {
            "num_graphs": int(total_graphs),
            "forward_seconds_total": float(total_forward),
            "forward_milliseconds_per_graph": float(total_forward * 1000.0 / total_graphs),
        },
    }
    logging.info(
        "%s evaluation: WMAPE=%.6f new-edge WMAPE=%.6f R2=%.6f RelCon=%.6f",
        network,
        real_metrics["all_edges"]["wmape"],
        real_metrics["new_edges"]["wmape"],
        real_metrics["all_edges"]["r2"],
        result["constraint_violation"]["relcon"],
    )
    return result


def make_loader(
    data: list,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        data,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(pin_memory),
        generator=generator,
    )


def train_one_batch(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    normalizer: DomainNormalizer,
    network: str,
    device: torch.device,
    grad_clip: float,
) -> float:
    activate_domain(model, normalizer, network)
    model.train()
    batch.split = "train"
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    pred, _ = model(batch)
    loss, _ = compute_pinn_loss(pred, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss while training on {network}: {loss.item()}")
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def make_source_scheduler(optimizer, epochs: int, warmup_epochs: int):
    stable_end = max(int(epochs * 0.75), warmup_epochs)

    def factor(epoch: int) -> float:
        current = epoch + 1
        if warmup_epochs > 0 and current <= warmup_epochs:
            return max(current / warmup_epochs, 1e-3)
        if current <= stable_end:
            return 1.0
        progress = (current - stable_end) / max(epochs - stable_end, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))), 1e-3)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def macro_source_wmape(results: dict[str, dict]) -> float:
    return float(np.mean([item["metrics_real"]["all_edges"]["wmape"] for item in results.values()]))


def train_multisource_loso(
    model: torch.nn.Module,
    source_train: dict[str, list],
    source_val: dict[str, list],
    normalizers: dict[str, DomainNormalizer],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    last_checkpoint_path: Path,
) -> tuple[dict, list[dict]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.source_lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = make_source_scheduler(optimizer, args.source_epochs, args.warmup_epochs)
    history = []
    best_wmape = float("inf")
    best_state = None
    best_epoch = -1
    sources = list(source_train)
    start_epoch = 0

    if last_checkpoint_path.exists() and not args.overwrite:
        try:
            last_checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            last_checkpoint = torch.load(last_checkpoint_path, map_location=device)
        if last_checkpoint.get("protocol_signature") != args.protocol_signature:
            raise RuntimeError(
                f"Existing source-training state has another protocol signature: {last_checkpoint_path}"
            )
        model.load_state_dict(last_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(last_checkpoint["optimizer_state"])
        scheduler.load_state_dict(last_checkpoint["scheduler_state"])
        start_epoch = int(last_checkpoint["next_epoch"])
        history = list(last_checkpoint.get("history", []))
        best_wmape = float(last_checkpoint.get("best_wmape", float("inf")))
        best_epoch = int(last_checkpoint.get("best_epoch", -1))
        best_state = last_checkpoint.get("best_state")
        logging.info(
            "Resuming source training at epoch %d/%d from %s",
            start_epoch + 1,
            args.source_epochs,
            last_checkpoint_path,
        )

    for epoch in range(start_epoch, args.source_epochs):
        cfg.train.current_epoch = int(epoch)
        epoch_losses = {network: [] for network in sources}
        loaders = {
            network: make_loader(
                source_train[network],
                args.batch_size,
                shuffle=True,
                seed=args.seed + epoch * 101 + idx,
                pin_memory=device.type == "cuda",
            )
            for idx, network in enumerate(sources)
        }
        steps = max(len(loader) for loader in loaders.values())
        iterators = {network: iter(loader) for network, loader in loaders.items()}

        # One update per source per round prevents large-graph domains from
        # dominating merely because they contain more edges.
        for _ in range(steps):
            for network in sources:
                try:
                    batch = next(iterators[network])
                except StopIteration:
                    iterators[network] = iter(loaders[network])
                    batch = next(iterators[network])
                value = train_one_batch(
                    model,
                    batch,
                    optimizer,
                    normalizers[network],
                    network,
                    device,
                    args.grad_clip,
                )
                epoch_losses[network].append(value)
        scheduler.step()

        should_eval = epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.source_epochs
        record = {
            "epoch": int(epoch + 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": {
                network: float(np.mean(values)) for network, values in epoch_losses.items()
            },
        }
        if should_eval:
            val_results = {}
            for network in sources:
                val_loader = make_loader(
                    source_val[network],
                    args.eval_batch_size,
                    shuffle=False,
                    seed=args.seed,
                    pin_memory=device.type == "cuda",
                )
                val_results[network] = evaluate(
                    model,
                    val_loader,
                    normalizers[network],
                    network,
                    device,
                )
            score = macro_source_wmape(val_results)
            record["source_val"] = val_results
            record["macro_source_val_wmape"] = score
            logging.info(
                "Epoch %d/%d source loss=%s macro val WMAPE=%.6f",
                epoch + 1,
                args.source_epochs,
                record["train_loss"],
                score,
            )
            if score < best_wmape:
                best_wmape = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                payload = {
                    "model_state": best_state,
                    "protocol_signature": str(args.protocol_signature),
                    "best_epoch": int(best_epoch),
                    "macro_source_val_wmape": float(best_wmape),
                    "sources": sources,
                    "target": canonical_network(args.target),
                    "normalization": "per-domain input-only",
                    "normalizers": {key: asdict(value) for key, value in normalizers.items()},
                }
                torch.save(payload, checkpoint_path)
                logging.info("Saved new best LOSO checkpoint: %s", checkpoint_path)
        history.append(record)
        torch.save(
            {
                "protocol_signature": str(args.protocol_signature),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "next_epoch": int(epoch + 1),
                "best_wmape": float(best_wmape),
                "best_epoch": int(best_epoch),
                "best_state": best_state,
                "history": history,
            },
            last_checkpoint_path,
        )

    if best_state is None:
        raise RuntimeError("LOSO training completed without a validation checkpoint.")
    model.load_state_dict(best_state, strict=True)
    return {
        "best_epoch": int(best_epoch),
        "macro_source_val_wmape": float(best_wmape),
        "checkpoint": str(checkpoint_path),
        "resumed_from_epoch": int(start_epoch),
    }, history


def load_loso_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch_load(checkpoint_path)
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"No model_state found in {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    return checkpoint


def fine_tune_target(
    model: torch.nn.Module,
    target_subset: list,
    normalizer: DomainNormalizer,
    target: str,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.adapt_lr),
        weight_decay=float(args.weight_decay),
    )
    losses = []
    cfg.train.current_epoch = int(getattr(cfg.optim, "max_epoch", args.source_epochs))
    for epoch in range(args.adapt_epochs):
        loader = make_loader(
            target_subset,
            args.batch_size,
            shuffle=True,
            seed=args.seed + seed_offset * 1009 + epoch,
            pin_memory=device.type == "cuda",
        )
        epoch_loss = []
        for batch in loader:
            epoch_loss.append(
                train_one_batch(
                    model,
                    batch,
                    optimizer,
                    normalizer,
                    target,
                    device,
                    args.grad_clip,
                )
            )
        losses.append(float(np.mean(epoch_loss)))
    return {
        "epochs": int(args.adapt_epochs),
        "optimizer_steps": int(args.adapt_epochs * math.ceil(len(target_subset) / args.batch_size)),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "learning_rate": float(args.adapt_lr),
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def protocol_signature(args: argparse.Namespace, sources: list[str]) -> str:
    cfg_path = Path(args.cfg).expanduser().resolve()
    cfg_digest = hashlib.sha256(cfg_path.read_bytes()).hexdigest() if cfg_path.exists() else "missing"
    payload = {
        "target": canonical_network(args.target),
        "sources": sources,
        "k_values": args.k_values,
        "source_epochs": args.source_epochs,
        "adapt_epochs": args.adapt_epochs,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "source_lr": args.source_lr,
        "adapt_lr": args.adapt_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "cfg_sha256": cfg_digest,
        "normalization": "input-only-per-domain-v1",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "multisource_loso_fewshot.log", mode="a", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="configs/GatedGCN/network-pairs-topology.yaml")
    parser.add_argument("--data-root", default="create_sioux_data/processed_data")
    parser.add_argument("--target", required=True, choices=tuple(NETWORKS.values()))
    parser.add_argument("--output-root", default="results/multisource_loso_fewshot")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[0, 50, 100, 250, 500, 1000, 2000, 4000],
    )
    parser.add_argument("--source-epochs", type=int, default=200)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.k_values = sorted(set(int(value) for value in args.k_values))
    if not args.k_values or args.k_values[0] != 0 or any(value < 0 for value in args.k_values):
        parser.error("--k-values must contain 0 and only non-negative integers.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device=cuda requested but CUDA is unavailable.")
    return args


def main() -> None:
    args = parse_args()
    target = canonical_network(args.target)
    sources = [network for network in NETWORKS.values() if network != target]
    output_dir = Path(args.output_root).expanduser().resolve() / f"target_{target.lower()}"
    setup_logging(output_dir)
    signature = protocol_signature(args, sources)
    args.protocol_signature = signature
    final_path = output_dir / "summary.json"
    if final_path.exists() and not args.overwrite:
        existing = load_json(final_path)
        if existing.get("protocol_signature") == signature:
            logging.info("Completed matching result already exists; skipping: %s", final_path)
            return
        raise RuntimeError(
            f"Existing summary has another protocol signature: {final_path}. "
            "Use --overwrite only if replacement is intentional."
        )

    set_seed(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    cfg_path = Path(args.cfg).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    # All train inputs are loaded because each domain needs an independently
    # fitted input-only normalizer. Target labels are not touched by fitting.
    train_data: dict[str, list] = {}
    normalizers: dict[str, DomainNormalizer] = {}
    for network in NETWORKS.values():
        dataset_dir = dataset_dir_for(data_root, network)
        train_data[network] = load_split(dataset_dir, "train")
        legacy_attr, legacy_flow = load_legacy_scalers(dataset_dir)
        normalizers[network] = fit_input_only_normalizer(
            network,
            train_data[network],
            legacy_attr,
            legacy_flow,
        )
        renormalize_split(train_data[network], normalizers[network])

    if max(args.k_values) > len(train_data[target]):
        raise ValueError(
            f"Requested k={max(args.k_values)}, but {target} has only "
            f"{len(train_data[target])} training graphs. Use the Vera 10k dataset "
            "with a 6000/2000/2000 split."
        )

    source_val = {}
    for network in sources:
        dataset_dir = dataset_dir_for(data_root, network)
        source_val[network] = load_split(dataset_dir, "val")
        renormalize_split(source_val[network], normalizers[network])
    target_test = load_split(dataset_dir_for(data_root, target), "test")
    renormalize_split(target_test, normalizers[target])

    save_json(
        output_dir / "input_only_normalizers.json",
        {
            "protocol": "per-domain input-only",
            "target_train_inputs_used_for_zero-label_calibration": True,
            "target_flow_new_used_for_scaler_fit": False,
            "normalizers": {key: asdict(value) for key, value in normalizers.items()},
        },
    )

    configure_graphgym(args, normalizers[sources[0]])
    device = torch.device("cuda" if cfg.accelerator == "cuda" else "cpu")
    if args.device == "cuda":
        device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = create_model().to(device)
    cfg.params = params_count(model)
    logging.info("Target=%s sources=%s device=%s parameters=%d", target, sources, device, cfg.params)

    loso_checkpoint = output_dir / "loso_best.ckpt"
    loso_last_checkpoint = output_dir / "loso_last.ckpt"
    loso_complete_path = output_dir / "source_training_complete.json"
    history_path = output_dir / "source_training_history.json"
    if loso_checkpoint.exists() and loso_complete_path.exists() and not args.overwrite:
        completion = load_json(loso_complete_path)
        if completion.get("protocol_signature") != signature:
            raise RuntimeError(
                f"Existing source-training completion marker has another protocol: {loso_complete_path}"
            )
        checkpoint_meta = load_loso_checkpoint(model, loso_checkpoint, device)
        if checkpoint_meta.get("protocol_signature") != signature:
            raise RuntimeError(
                f"Existing LOSO checkpoint has another protocol signature: {loso_checkpoint}. "
                "Use --overwrite only if replacement is intentional."
            )
        loso_training = {
            "best_epoch": int(checkpoint_meta.get("best_epoch", -1)),
            "macro_source_val_wmape": float(checkpoint_meta.get("macro_source_val_wmape", float("nan"))),
            "checkpoint": str(loso_checkpoint),
            "reused_existing_checkpoint": True,
        }
        logging.info("Reusing existing LOSO checkpoint: %s", loso_checkpoint)
    else:
        loso_training, history = train_multisource_loso(
            model,
            {network: train_data[network] for network in sources},
            source_val,
            normalizers,
            args,
            device,
            loso_checkpoint,
            loso_last_checkpoint,
        )
        save_json(history_path, history)
        loso_training["reused_existing_checkpoint"] = False
        save_json(
            loso_complete_path,
            {
                "protocol_signature": signature,
                "source_epochs_completed": int(args.source_epochs),
                "best_epoch": int(loso_training["best_epoch"]),
                "macro_source_val_wmape": float(loso_training["macro_source_val_wmape"]),
            },
        )
        load_loso_checkpoint(model, loso_checkpoint, device)

    generator = np.random.default_rng(args.seed)
    nested_order = generator.permutation(len(train_data[target])).astype(np.int64)
    np.savez_compressed(output_dir / "fewshot_nested_indices.npz", order=nested_order)
    test_loader = make_loader(
        target_test,
        args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    k_results = {}
    for k in args.k_values:
        result_path = output_dir / f"fewshot_k{k:04d}.json"
        if result_path.exists() and not args.overwrite:
            result = load_json(result_path)
            if result.get("protocol_signature") == signature:
                k_results[str(k)] = result
                logging.info("Reusing completed k=%d result", k)
                continue
            raise RuntimeError(f"Protocol mismatch in existing result: {result_path}")

        load_loso_checkpoint(model, loso_checkpoint, device)
        adaptation = None
        if k > 0:
            subset_indices = nested_order[:k]
            subset = [train_data[target][int(index)] for index in subset_indices]
            adaptation = fine_tune_target(
                model,
                subset,
                normalizers[target],
                target,
                args,
                device,
                seed_offset=k,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target": target,
                    "k": int(k),
                    "source_checkpoint": str(loso_checkpoint),
                    "normalizer": asdict(normalizers[target]),
                },
                output_dir / f"fewshot_k{k:04d}.ckpt",
            )

        evaluation = evaluate(model, test_loader, normalizers[target], target, device)
        result = {
            "protocol_signature": signature,
            "target": target,
            "sources": sources,
            "k": int(k),
            "zero_label_target_input_normalization": True,
            "adaptation": adaptation,
            "test": evaluation,
        }
        save_json(result_path, result)
        k_results[str(k)] = result

    final = {
        "task": "multi-source-loso-few-shot",
        "protocol_signature": signature,
        "target": target,
        "sources": sources,
        "k_values": args.k_values,
        "seed": int(args.seed),
        "repeat_count": 1,
        "normalization": {
            "policy": "per-domain input-only",
            "flow_fit_inputs": "train.flow_old only",
            "attribute_fit_inputs": "train.edge_attr_old + train.edge_attr_new",
            "target_labels_used_for_normalization": False,
            "target_unlabeled_train_inputs_used": True,
        },
        "source_training": loso_training,
        "fewshot_subset_policy": "single deterministic nested subset; every k starts from the same LOSO checkpoint",
        "num_parameters": int(cfg.params),
        "config": str(cfg_path),
        "data_root": str(data_root),
        "k_results": k_results,
    }
    save_json(final_path, final)
    logging.info("Completed target %s. Summary: %s", target, final_path)


if __name__ == "__main__":
    main()
