#!/usr/bin/env python
"""Evaluate zero-shot transfer of a trained topology model on another network."""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import sys
import time
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from yacs.config import CfgNode

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import graphgps  # noqa: E402,F401
from constraint_violation.metrics import ConstraintViolationAccumulator  # noqa: E402
from graphgps.loss.flow_conservation_loss import compute_pinn_loss  # noqa: E402
from graphgps.metric_wrapper import get_flow_metric_tensors, wmape  # noqa: E402
from graphgps.utils import match_edge_indices  # noqa: E402


SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}


class RegressionAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_true = 0.0
        self.sum_true_sq = 0.0
        self.sum_abs_true = 0.0

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        pred = pred.view(-1).double()
        true = true.view(-1).double()
        if pred.numel() == 0:
            return
        error = pred - true
        self.count += int(pred.numel())
        self.sum_abs_error += float(error.abs().sum().item())
        self.sum_squared_error += float(error.pow(2).sum().item())
        self.sum_true += float(true.sum().item())
        self.sum_true_sq += float(true.pow(2).sum().item())
        self.sum_abs_true += float(true.abs().sum().item())

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "wmape": 0.0}
        denom = self.sum_true_sq - (self.sum_true**2) / self.count
        return {
            "count": int(self.count),
            "mae": float(self.sum_abs_error / self.count),
            "rmse": float(np.sqrt(self.sum_squared_error / self.count)),
            "r2": float(0.0 if denom <= 1e-12 else 1.0 - self.sum_squared_error / denom),
            "wmape": float(
                0.0 if self.sum_abs_true <= 1e-12 else self.sum_abs_error / self.sum_abs_true
            ),
        }


class MetricBundle:
    def __init__(self) -> None:
        self.all_edges = RegressionAccumulator()
        self.old_edges = RegressionAccumulator()
        self.new_edges = RegressionAccumulator()

    def update(self, pred: torch.Tensor, true: torch.Tensor, new_edge_mask: torch.Tensor) -> None:
        pred = pred.view(-1)
        true = true.view(-1)
        mask_new = new_edge_mask.view(-1).bool()
        self.all_edges.update(pred, true)
        self.old_edges.update(pred[~mask_new], true[~mask_new])
        self.new_edges.update(pred[mask_new], true[mask_new])

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            "all_edges": self.all_edges.as_dict(),
            "old_edges": self.old_edges.as_dict(),
            "new_edges": self.new_edges.as_dict(),
        }


class SplitMetrics:
    def __init__(self) -> None:
        self.normalized = MetricBundle()
        self.real = MetricBundle()

    def as_dict(self) -> dict[str, dict[str, dict[str, float | int]]]:
        return {"normalized": self.normalized.as_dict(), "real": self.real.as_dict()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation: load a source-network checkpoint and evaluate it on a target network."
    )
    parser.add_argument("--cfg", required=True, help="GraphGPS config used by the source checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to source-network checkpoint.")
    parser.add_argument("--source-network", required=True, help="Network used to train the checkpoint.")
    parser.add_argument("--target-network", required=True, help="Network used only for zero-shot evaluation.")
    parser.add_argument(
        "--target-dir",
        default="",
        help="Processed target dataset directory. If set, overrides dataset.dir and dataset.processed_root.",
    )
    parser.add_argument(
        "--target-processed-root",
        default="",
        help="Root containing processed network directories. Used with --target-network.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=tuple(SPLIT_TO_INDEX),
        help="Dataset splits to evaluate.",
    )
    parser.add_argument("--out-dir", default="results/zero_shot_generalization")
    parser.add_argument("--batch-size", type=int, default=0, help="Override cfg.train.batch_size when > 0.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device. Use auto on Vera unless a specific device is required.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Smoke-test limiter. 0 means evaluate the full selected split.",
    )
    parser.add_argument(
        "--scaler-policy",
        default="target",
        choices=("target", "checkpoint"),
        help=(
            "target resets model flow_mean/flow_std buffers to the target dataset scaler after loading "
            "the checkpoint; checkpoint keeps the source-network scaler saved in the checkpoint."
        ),
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys. Intended only for debugging incompatible checkpoints.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional GraphGPS KEY VALUE overrides.")
    return parser.parse_args()


def setup_logging(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    log_path = osp.join(out_dir, "zero_shot_eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
    )
    return log_path


def limited_loader(loader: Iterable, max_batches: int) -> Iterable:
    if max_batches <= 0:
        yield from loader
        return
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        yield batch


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    return value


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, indent=2, ensure_ascii=False)


def cfg_to_json_dict(node):
    if isinstance(node, CfgNode):
        return {key: cfg_to_json_dict(value) for key, value in dict(node).items()}
    if isinstance(node, (list, tuple)):
        return [cfg_to_json_dict(value) for value in node]
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    return str(node)


def load_json_if_exists(path: str) -> dict:
    if not osp.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_graphgym(args: argparse.Namespace) -> None:
    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=args.cfg, opts=args.opts))

    cfg.seed = int(args.seed)
    cfg.accelerator = args.device
    cfg.wandb.use = False
    cfg.dataset.network_name = args.target_network
    cfg.train.current_epoch = int(getattr(cfg.optim, "max_epoch", 0))
    cfg.train.mode = "inference-only"

    if args.batch_size > 0:
        cfg.train.batch_size = int(args.batch_size)
    if args.target_dir:
        cfg.dataset.dir = args.target_dir
        cfg.dataset.processed_root = args.target_dir
    elif args.target_processed_root:
        cfg.dataset.processed_root = args.target_processed_root
        cfg.dataset.dir = args.target_processed_root

    auto_select_device()
    torch.set_num_threads(int(getattr(cfg, "num_threads", 1)))
    seed_everything(cfg.seed)


def extract_state_dict(checkpoint_path: str) -> tuple[dict, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = {}
    if isinstance(checkpoint, dict):
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state", "model_state_dict", "state_dict", "optimizer_state", "scheduler_state"}
        }
        for key in ("model_state", "model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state, metadata
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint, metadata
    raise ValueError(f"Could not find a model state dict in checkpoint: {checkpoint_path}")


def adapt_state_dict_keys(state_dict: dict, model: torch.nn.Module) -> dict:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    if state_keys & model_keys:
        return state_dict

    if not any(key.startswith("model.") for key in state_keys) and any(
        key.startswith("model.") for key in model_keys
    ):
        prefixed = {f"model.{key}": value for key, value in state_dict.items()}
        if set(prefixed.keys()) & model_keys:
            return prefixed

    if any(key.startswith("module.") for key in state_keys):
        stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
        if set(stripped.keys()) & model_keys:
            return stripped

    return state_dict


def load_model_checkpoint(model: torch.nn.Module, args: argparse.Namespace) -> dict:
    state_dict, checkpoint_metadata = extract_state_dict(args.checkpoint)
    state_dict = adapt_state_dict_keys(state_dict, model)
    incompatible = model.load_state_dict(state_dict, strict=not args.non_strict)
    if args.non_strict:
        logging.warning("Loaded checkpoint non-strictly: %s", incompatible)
    return checkpoint_metadata


def reset_flow_scaler_to_target(model: torch.nn.Module) -> dict[str, float] | None:
    target = model.model if hasattr(model, "model") else model
    if not hasattr(target, "flow_mean") or not hasattr(target, "flow_std"):
        return None
    with torch.no_grad():
        target.flow_mean.fill_(float(cfg.dataset.flow_mean))
        target.flow_std.fill_(float(cfg.dataset.flow_std))
    return {"flow_mean": float(cfg.dataset.flow_mean), "flow_std": float(cfg.dataset.flow_std)}


def compute_new_edge_mask(batch) -> torch.Tensor:
    match_idx = match_edge_indices(
        edge_index_old=batch.edge_index_old,
        edge_index_new=batch.edge_index_new,
        total_nodes=batch.num_nodes,
    )
    return match_idx < 0


def collect_batch_loss_stats(batch) -> dict[str, float]:
    stats = {}
    for key in (
        "loss_old",
        "loss_new",
        "lambda_new_current",
        "loss_sup",
        "loss_data",
        "loss_con",
        "lambda_con",
        "rho_terminal_abs_mean",
        "rho_terminal_abs_max",
    ):
        if not hasattr(batch, key):
            continue
        value = getattr(batch, key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        stats[key] = float(value)
    return stats


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_split(loader: Iterable, model: torch.nn.Module, split: str, max_batches: int) -> dict:
    model.eval()
    device = torch.device(cfg.accelerator)
    metrics = SplitMetrics()
    constraint_metrics = ConstraintViolationAccumulator()

    per_graph_wmapes = []
    total_forward_seconds = 0.0
    total_graphs = 0
    loss_total = 0.0
    loss_old = 0.0
    loss_new = 0.0
    loss_sup = 0.0
    loss_con = 0.0
    lambda_new_current = 0.0
    num_batches = 0

    for batch in limited_loader(loader, max_batches):
        batch.split = split
        batch.to(device)
        total_graphs += int(batch.num_graphs)

        maybe_cuda_synchronize(device)
        start = time.perf_counter()
        pred, true = model(batch)
        maybe_cuda_synchronize(device)
        total_forward_seconds += time.perf_counter() - start

        loss, pred_score = compute_pinn_loss(pred, batch)
        batch_loss_stats = collect_batch_loss_stats(batch)
        loss_total += float(loss.detach().cpu().item())
        loss_old += float(batch_loss_stats.get("loss_old", 0.0))
        loss_new += float(batch_loss_stats.get("loss_new", 0.0))
        loss_sup += float(batch_loss_stats.get("loss_sup", loss.detach().cpu().item()))
        loss_con += float(batch_loss_stats.get("loss_con", 0.0))
        lambda_new_current += float(batch_loss_stats.get("lambda_new_current", 0.0))
        num_batches += 1

        pred_cpu = pred_score.detach().cpu().float()
        true_cpu = true.detach().cpu().float()
        pred_real, true_real, _ = get_flow_metric_tensors(pred_cpu, true_cpu)

        if hasattr(batch, "new_edge_mask"):
            is_new_edge = batch.new_edge_mask.bool().detach().cpu()
        else:
            is_new_edge = compute_new_edge_mask(batch).detach().cpu()

        metrics.normalized.update(pred_cpu, true_cpu, is_new_edge)
        metrics.real.update(pred_real, true_real, is_new_edge)
        constraint_metrics.update(
            pred_real=pred_real.view(-1),
            edge_index_new=batch.edge_index_new.detach().cpu(),
            net_demand=batch.net_demand.detach().cpu(),
            node_batch=batch.batch.detach().cpu(),
            ptr=batch.ptr.detach().cpu(),
        )

        edge_batch = batch.batch[batch.edge_index_new[0]].detach().cpu()
        for graph_idx in range(batch.num_graphs):
            mask_graph = edge_batch == graph_idx
            pred_graph = pred_real[mask_graph]
            true_graph = true_real[mask_graph]
            if true_graph.numel() > 0:
                per_graph_wmapes.append(float(wmape(pred_graph, true_graph).item()))

    if num_batches == 0:
        raise RuntimeError(f"No batches were evaluated for split '{split}'.")

    w_tensor = None
    q_values = None
    if per_graph_wmapes:
        w_tensor = torch.tensor(per_graph_wmapes, dtype=torch.float64)
        q_values = torch.quantile(
            w_tensor,
            torch.tensor([0.25, 0.50, 0.75, 0.95], dtype=torch.float64),
        )

    denom = max(num_batches, 1)
    total_time_ms = total_forward_seconds * 1000.0
    result = {
        "loss": {
            "loss_total": loss_total / denom,
            "loss_old": loss_old / denom,
            "loss_new": loss_new / denom,
            "lambda_new_current": lambda_new_current / denom,
            "loss_sup": loss_sup / denom,
            "loss_con": loss_con / denom,
        },
        "metrics": metrics.as_dict(),
        "constraint_violation": constraint_metrics.as_dict(),
        "wmape_mean_real": float(w_tensor.mean()) if w_tensor is not None else None,
        "wmape_percentiles_real": {
            "p25": float(q_values[0]) if q_values is not None else None,
            "p50": float(q_values[1]) if q_values is not None else None,
            "p75": float(q_values[2]) if q_values is not None else None,
            "p95": float(q_values[3]) if q_values is not None else None,
        },
        "timing": {
            "forward_seconds_total": float(total_forward_seconds),
            "forward_milliseconds_per_graph": float(total_time_ms / max(total_graphs, 1)),
            "num_graphs": int(total_graphs),
            "num_batches": int(num_batches),
        },
    }
    log_split_summary(split, result)
    return result


def log_split_summary(split: str, result: dict) -> None:
    real = result["metrics"]["real"]
    normalized = result["metrics"]["normalized"]
    con = result["constraint_violation"]
    logging.info(
        "[%s] normalized WMAPE %.6f | real WMAPE %.6f | new-edge real WMAPE %.6f | "
        "old-edge real WMAPE %.6f | RelCon %.6f | graphs %s | ms/graph %.4f",
        split,
        normalized["all_edges"]["wmape"],
        real["all_edges"]["wmape"],
        real["new_edges"]["wmape"],
        real["old_edges"]["wmape"],
        con["relcon"],
        result["timing"]["num_graphs"],
        result["timing"]["forward_milliseconds_per_graph"],
    )


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.out_dir)
    configure_graphgym(args)

    if args.source_network.strip().lower() == args.target_network.strip().lower():
        logging.warning("Source and target network names are identical; this is not a zero-shot transfer.")

    logging.info("Loading target dataset '%s' from cfg.dataset.dir=%s", args.target_network, cfg.dataset.dir)
    loaders = create_loader()
    target_dataset_meta = load_json_if_exists(osp.join(cfg.dataset.dir, "dataset_meta.json"))

    logging.info("Creating model on %s", cfg.accelerator)
    model = create_model()
    checkpoint_metadata = load_model_checkpoint(model, args)
    scaler_after_load = None
    if args.scaler_policy == "target":
        scaler_after_load = reset_flow_scaler_to_target(model)
        logging.info("Reset model flow scaler buffers to target dataset scaler: %s", scaler_after_load)
    else:
        logging.info("Keeping checkpoint flow scaler buffers.")

    cfg.params = params_count(model)
    logging.info("Loaded checkpoint: %s", args.checkpoint)
    logging.info("Model parameters: %s", cfg.params)

    split_results = {}
    for split in args.splits:
        split_results[split] = evaluate_split(
            loaders[SPLIT_TO_INDEX[split]],
            model,
            split=split,
            max_batches=args.max_batches,
        )

    summary = {
        "task": "zero-shot-generalization",
        "source_network": args.source_network,
        "target_network": args.target_network,
        "checkpoint": osp.abspath(args.checkpoint),
        "config": osp.abspath(args.cfg),
        "device": cfg.accelerator,
        "seed": int(cfg.seed),
        "is_smoke_test": bool(args.max_batches > 0),
        "max_batches": int(args.max_batches),
        "scaler_policy": args.scaler_policy,
        "model_flow_scaler_after_load": scaler_after_load,
        "checkpoint_metadata": checkpoint_metadata,
        "target_dataset_dir": cfg.dataset.dir,
        "target_dataset_metadata": target_dataset_meta,
        "num_parameters": int(cfg.params),
        "splits": split_results,
        "graphgps_config": cfg_to_json_dict(cfg),
    }

    summary_path = osp.join(
        args.out_dir,
        f"zero_shot_{args.source_network.lower()}_to_{args.target_network.lower()}.json",
    )
    save_json(summary_path, summary)
    logging.info("Saved zero-shot summary to %s", summary_path)
    logging.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
