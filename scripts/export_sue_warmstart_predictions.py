#!/usr/bin/env python
"""Export Edge Transformer test predictions for the SUE warm-start benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.loader import DataLoader

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import graphgps  # noqa: E402,F401


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> tuple[Path, dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_path}")
    summary = load_json(summary_path)
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
    else:
        best_epoch = int(summary["best_epoch"])
        checkpoint_path = run_dir / "ckpt" / f"{best_epoch}.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return checkpoint_path, summary


def apply_saved_topology_config(training_summary: dict) -> None:
    saved = training_summary.get("config", {}).get("topology_gnn", {})
    if not saved:
        raise ValueError("Training summary has no config.topology_gnn; architecture cannot be reproduced.")
    for key, value in saved.items():
        if not hasattr(cfg.topology_gnn, key):
            raise KeyError(f"Current code has no topology_gnn.{key} required by the checkpoint.")
        setattr(cfg.topology_gnn, key, value)


def configure(args: argparse.Namespace, training_summary: dict) -> torch.device:
    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=args.cfg, opts=[]))
    apply_saved_topology_config(training_summary)
    cfg.seed = int(args.seed)
    cfg.wandb.use = False
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.current_epoch = int(getattr(cfg.optim, "max_epoch", 0))
    cfg.train.mode = "inference-only"
    cfg.dataset.network_name = args.network
    cfg.dataset.dir = str(args.dataset_dir)
    cfg.dataset.processed_root = str(args.dataset_dir)
    cfg.share.dim_in = 1
    cfg.share.dim_out = 1
    cfg.share.num_splits = 3
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        cfg.accelerator = "cuda"
    elif args.device == "cpu":
        cfg.accelerator = "cpu"
    else:
        cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(args.num_threads))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    return torch.device(cfg.accelerator)


def state_dict_from_checkpoint(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    for key in ("model_state", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("No model state dictionary found in checkpoint.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="configs/GatedGCN/network-pairs-topology.yaml")
    parser.add_argument("--network", required=True, choices=("SiouxFalls", "EMA", "Anaheim"))
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-graphs", type=int, default=0, help="0 exports the complete test split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path, training_summary = resolve_checkpoint(args.run_dir, args.checkpoint)
    trained_network = str(training_summary.get("network_name", ""))
    if trained_network.lower() != args.network.lower():
        raise ValueError(
            f"Checkpoint was trained on {trained_network!r}, requested network is {args.network!r}."
        )
    device = configure(args, training_summary)

    test_path = args.dataset_dir / "test_dataset.pt"
    split_path = args.dataset_dir / "split_indices.npz"
    if not test_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing test dataset or split indices under {args.dataset_dir}")
    test_dataset = torch_load(test_path)
    test_indices = np.load(split_path)["test_idx"].astype(np.int64)
    if len(test_dataset) != len(test_indices):
        raise RuntimeError(
            f"Test dataset length {len(test_dataset)} != test_idx length {len(test_indices)}."
        )
    if args.max_graphs > 0:
        test_dataset = test_dataset[: args.max_graphs]
        test_indices = test_indices[: args.max_graphs]

    model = create_model().to(device)
    checkpoint = torch_load(checkpoint_path)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    inner = model.model if hasattr(model, "model") else model
    flow_mean = float(inner.flow_mean.detach().cpu().item())
    flow_std = float(inner.flow_std.detach().cpu().item())
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    predicted_parts = []
    true_parts = []
    edge_counts = []
    forward_seconds = 0.0
    for batch in loader:
        batch.split = "test"
        batch = batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred_scaled, true_scaled = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - start

        pred_real = (pred_scaled * flow_std + flow_mean).view(-1).detach().cpu().numpy()
        true_real = (true_scaled * flow_std + flow_mean).view(-1).detach().cpu().numpy()
        edge_batch = batch.batch[batch.edge_index_new[0]].detach().cpu()
        counts = torch.bincount(edge_batch, minlength=int(batch.num_graphs)).numpy().astype(np.int64)
        cursor = 0
        for count in counts:
            next_cursor = cursor + int(count)
            predicted_parts.append(pred_real[cursor:next_cursor].astype(np.float32, copy=True))
            true_parts.append(true_real[cursor:next_cursor].astype(np.float32, copy=True))
            edge_counts.append(int(count))
            cursor = next_cursor
        if cursor != pred_real.size:
            raise RuntimeError("Failed to segment batched edge predictions by graph.")

    offsets = np.zeros(len(edge_counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(edge_counts, dtype=np.int64))
    predicted = np.concatenate(predicted_parts) if predicted_parts else np.zeros(0, dtype=np.float32)
    true = np.concatenate(true_parts) if true_parts else np.zeros(0, dtype=np.float32)
    if not np.all(np.isfinite(predicted)):
        raise FloatingPointError("Model predictions contain NaN or Inf.")
    abs_error = np.abs(predicted.astype(np.float64) - true.astype(np.float64))
    wmape = float(abs_error.sum() / max(np.abs(true.astype(np.float64)).sum(), 1e-12))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "test_predictions.npz"
    temporary = prediction_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pair_indices=test_indices,
            offsets=offsets,
            edge_counts=np.asarray(edge_counts, dtype=np.int64),
            predicted_flows=predicted,
            true_flows=true,
        )
    os.replace(temporary, prediction_path)

    metadata = {
        "task": "sue-warmstart-prediction-export",
        "network": args.network,
        "dataset_dir": str(args.dataset_dir),
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "num_graphs": int(len(test_indices)),
        "num_edges": int(predicted.size),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "negative_prediction_count": int(np.count_nonzero(predicted < 0.0)),
        "negative_prediction_fraction": float(np.mean(predicted < 0.0)) if predicted.size else 0.0,
        "prediction_wmape_against_dataset_labels": wmape,
        "forward_seconds_total": float(forward_seconds),
        "forward_milliseconds_per_graph": float(1000.0 * forward_seconds / max(len(test_indices), 1)),
        "prediction_file": str(prediction_path),
    }
    metadata_path = args.output_dir / "prediction_metadata.json"
    save_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
