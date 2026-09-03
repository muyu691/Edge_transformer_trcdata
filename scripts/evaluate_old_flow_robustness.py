#!/usr/bin/env python
"""Evaluate a trained Edge Transformer under corrupted old-flow observations."""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
from types import SimpleNamespace
from typing import Iterable

import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

from evaluate_zero_shot import (
    cfg_to_json_dict,
    evaluate_split,
    load_json_if_exists,
    load_model_checkpoint,
    reset_flow_scaler_to_target,
    save_json,
)


class OldFlowPerturber:
    """Apply deterministic per-graph corruption and rebuild the demand proxy."""

    def __init__(self, mode: str, level: float, seed: int) -> None:
        self.mode = mode
        self.level = float(level)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

        self.total_edges = 0
        self.changed_edges = 0
        self.clipped_edges = 0
        self.original_abs_sum = 0.0
        self.absolute_change_sum = 0.0

    def _missing(self, flow_real: torch.Tensor, edge_graph: torch.Tensor, num_graphs: int) -> None:
        for graph_idx in range(num_graphs):
            rows = torch.nonzero(edge_graph == graph_idx, as_tuple=False).view(-1)
            if rows.numel() == 0:
                continue
            permutation = torch.randperm(rows.numel(), generator=self.generator)
            num_missing = int(round(rows.numel() * self.level))
            if num_missing <= 0:
                continue
            selected = rows[permutation[:num_missing]]
            flow_real[selected] = 0.0
            self.changed_edges += int(selected.numel())

    def _gaussian_noise(self, flow_real: torch.Tensor) -> None:
        noise = torch.randn(
            flow_real.shape,
            generator=self.generator,
            dtype=flow_real.dtype,
            device=flow_real.device,
        )
        flow_real.mul_(1.0 + self.level * noise)
        negative_mask = flow_real < 0.0
        self.clipped_edges += int(negative_mask.sum().item())
        flow_real.clamp_(min=0.0)
        self.changed_edges += int(flow_real.numel())

    def __call__(self, batch):
        if batch.flow_old.device.type != "cpu":
            raise RuntimeError("Old-flow perturbation must run before the batch is moved to the GPU.")

        flow_mean = float(cfg.dataset.flow_mean)
        flow_std = max(float(cfg.dataset.flow_std), 1e-6)
        original_real = batch.flow_old.detach().float() * flow_std + flow_mean
        perturbed_real = original_real.clone()

        edge_graph = batch.batch[batch.edge_index_old[0]].cpu()
        if self.mode == "missing":
            self._missing(perturbed_real, edge_graph, int(batch.num_graphs))
        elif self.mode == "gaussian_noise":
            self._gaussian_noise(perturbed_real)
        else:
            raise ValueError(f"Unsupported perturbation mode: {self.mode}")

        self.total_edges += int(original_real.numel())
        self.original_abs_sum += float(original_real.abs().sum().item())
        self.absolute_change_sum += float((perturbed_real - original_real).abs().sum().item())

        batch.flow_old = ((perturbed_real - flow_mean) / flow_std).to(batch.flow_old.dtype)

        # net_demand is derived from old-flow divergence in the original dataset.
        # Recompute it from the corrupted observations to avoid leaking clean flow.
        flow_flat = perturbed_real.view(-1).to(batch.net_demand.dtype)
        src = batch.edge_index_old[0]
        dst = batch.edge_index_old[1]
        net_demand = flow_flat.new_zeros(int(batch.num_nodes))
        net_demand.index_add_(0, dst, flow_flat)
        net_demand.index_add_(0, src, -flow_flat)
        batch.net_demand = net_demand
        return batch

    def summary(self) -> dict[str, float | int | str]:
        changed_fraction = self.changed_edges / max(self.total_edges, 1)
        relative_l1_change = self.absolute_change_sum / max(self.original_abs_sum, 1e-12)
        return {
            "mode": self.mode,
            "requested_level": self.level,
            "total_old_edges": int(self.total_edges),
            "changed_old_edges": int(self.changed_edges),
            "changed_edge_fraction": float(changed_fraction),
            "relative_l1_flow_change": float(relative_l1_change),
            "negative_values_clipped": int(self.clipped_edges),
            "missing_value_imputation": "zero" if self.mode == "missing" else "not_applicable",
            "net_demand_policy": "recomputed_from_corrupted_old_flow",
        }


class PerturbedLoader:
    def __init__(self, loader: Iterable, perturber: OldFlowPerturber) -> None:
        self.loader = loader
        self.perturber = perturber

    def __iter__(self):
        for batch in self.loader:
            yield self.perturber(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate old-flow missingness or Gaussian-noise robustness."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=("missing", "gaussian_noise"), required=True)
    parser.add_argument(
        "--level",
        type=float,
        required=True,
        help="Fraction in [0, 1]. Missing ratio or Gaussian multiplicative-noise sigma.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0.0 <= args.level <= 1.0:
        parser.error("--level must be in [0, 1].")
    return args


def setup_logging(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    log_path = osp.join(out_dir, "old_flow_robustness.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
    )
    return log_path


def configure_graphgym(args: argparse.Namespace) -> None:
    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=args.cfg, opts=args.opts))
    cfg.seed = int(args.seed)
    cfg.accelerator = args.device
    cfg.wandb.use = False
    cfg.dataset.network_name = args.network
    cfg.dataset.dir = args.dataset_dir
    cfg.dataset.processed_root = args.dataset_dir
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.current_epoch = int(getattr(cfg.optim, "max_epoch", 0))
    cfg.train.mode = "inference-only"
    auto_select_device()
    torch.set_num_threads(int(getattr(cfg, "num_threads", 1)))
    seed_everything(cfg.seed)


def result_filename(network: str, mode: str, level: float) -> str:
    level_percent = int(round(level * 100.0))
    return f"old_flow_robustness_{network.lower()}_{mode}_{level_percent:02d}pct.json"


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.out_dir)
    configure_graphgym(args)

    logging.info("Loading %s dataset from %s", args.network, args.dataset_dir)
    loaders = create_loader()
    dataset_metadata = load_json_if_exists(osp.join(cfg.dataset.dir, "dataset_meta.json"))

    model = create_model()
    checkpoint_metadata = load_model_checkpoint(model, args)
    scaler_after_load = reset_flow_scaler_to_target(model)
    cfg.params = params_count(model)
    logging.info("Loaded checkpoint %s", args.checkpoint)
    logging.info("Target flow scaler restored after checkpoint load: %s", scaler_after_load)

    perturber = OldFlowPerturber(args.mode, args.level, args.seed)
    test_result = evaluate_split(
        PerturbedLoader(loaders[2], perturber),
        model,
        split="test",
        max_batches=args.max_batches,
    )

    payload = {
        "task": "old-flow-robustness",
        "network": args.network,
        "checkpoint": osp.abspath(args.checkpoint),
        "config": osp.abspath(args.cfg),
        "dataset_dir": cfg.dataset.dir,
        "dataset_metadata": dataset_metadata,
        "device": cfg.accelerator,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "is_smoke_test": bool(args.max_batches > 0),
        "max_batches": int(args.max_batches),
        "perturbation": perturber.summary(),
        "test": test_result,
        "model_flow_scaler_after_load": scaler_after_load,
        "checkpoint_metadata": checkpoint_metadata,
        "num_parameters": int(cfg.params),
        "graphgps_config": cfg_to_json_dict(cfg),
    }
    output_path = osp.join(args.out_dir, result_filename(args.network, args.mode, args.level))
    save_json(output_path, payload)
    logging.info("Saved robustness summary to %s", output_path)
    logging.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
