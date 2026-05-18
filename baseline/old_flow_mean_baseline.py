from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baseline.common import denormalize_flow, match_edge_indices, maybe_cuda_synchronize, set_seed
from baseline.data import build_loader, load_dataset_bundle
from baseline.metrics import SplitMetrics
from constraint_violation.metrics import ConstraintViolationAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the old-flow/mean-flow baseline. Retained links copy old flow; "
            "new links receive the mean old flow over retained links in the current batch."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Explicit dataset directory containing *.pt splits and scalers.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset alias to resolve under processed_root. Supported: ema, siouxfalls.",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="create_sioux_data/processed_data",
        help="Root directory that stores processed PyG datasets.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for metrics.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def predict_old_flow_mean(batch, total_nodes: int) -> torch.Tensor:
    """Copy old flow for retained links; fill new links with retained old-flow mean."""
    match_idx = match_edge_indices(
        edge_index_old=batch.edge_index_old,
        edge_index_new=batch.edge_index_new,
        total_nodes=total_nodes,
    )
    pred = torch.empty(
        (batch.edge_index_new.size(1), 1),
        dtype=batch.flow_old.dtype,
        device=batch.flow_old.device,
    )
    retained_mask = match_idx >= 0
    if retained_mask.any():
        retained_old_flow = batch.flow_old[match_idx[retained_mask]]
        mean_old_flow = retained_old_flow.mean()
        pred[retained_mask] = retained_old_flow
        pred[~retained_mask] = mean_old_flow
    else:
        pred.fill_(0.0)
    return pred


@torch.no_grad()
def evaluate_split(
    loader,
    device: torch.device,
    flow_mean: float,
    flow_std: float,
    measure_inference_time: bool,
) -> dict:
    split_metrics = SplitMetrics()
    constraint_metrics = ConstraintViolationAccumulator()
    total_forward_seconds = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        maybe_cuda_synchronize(device)
        start = time.perf_counter()
        pred = predict_old_flow_mean(batch=batch, total_nodes=int(batch.num_nodes))
        maybe_cuda_synchronize(device)
        if measure_inference_time:
            total_forward_seconds += time.perf_counter() - start

        pred_cpu = pred.detach().cpu()
        true_cpu = batch.y.detach().cpu()
        mask_cpu = batch.new_edge_mask.detach().cpu().bool()
        pred_real = denormalize_flow(pred_cpu, flow_mean=flow_mean, flow_std=flow_std)
        true_real = denormalize_flow(true_cpu, flow_mean=flow_mean, flow_std=flow_std)

        split_metrics.normalized.update(pred_cpu, true_cpu, mask_cpu)
        split_metrics.real.update(pred_real, true_real, mask_cpu)
        constraint_metrics.update(
            pred_real=pred_real.view(-1),
            edge_index_new=batch.edge_index_new.detach().cpu(),
            net_demand=batch.net_demand.detach().cpu(),
            node_batch=batch.batch.detach().cpu(),
            ptr=batch.ptr.detach().cpu(),
        )
        total_graphs += int(getattr(batch, "num_graphs", 1))

    forward_ms_per_graph = 0.0 if total_graphs == 0 else total_forward_seconds * 1000.0 / total_graphs
    return {
        "metrics": split_metrics.as_dict(),
        "timing": {
            "forward_seconds_total": total_forward_seconds if measure_inference_time else 0.0,
            "forward_milliseconds_per_graph": forward_ms_per_graph if measure_inference_time else 0.0,
            "num_graphs": total_graphs,
        },
        "constraint_violation": constraint_metrics.as_dict(),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = build_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_bundle(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        processed_root=args.processed_root,
    )
    loaders = {
        "train": build_loader(dataset.train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "val": build_loader(dataset.val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": build_loader(dataset.test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }

    split_results = {
        split: evaluate_split(
            loader=loader,
            device=device,
            flow_mean=dataset.flow_mean,
            flow_std=dataset.flow_std,
            measure_inference_time=(split == "test"),
        )
        for split, loader in loaders.items()
    }

    summary = {
        "model": "old_flow_mean_baseline",
        "description": (
            "For each edge in the reconfigured graph, copy the pre-reconfiguration "
            "flow if the directed edge exists in the old graph; otherwise predict "
            "the mean pre-reconfiguration flow over retained edges in the current batch."
        ),
        "dataset_dir": str(dataset.dataset_dir),
        "device_used": str(device),
        "seed": int(args.seed),
        "test_time": split_results["test"]["timing"],
        "test_metrics": split_results["test"]["metrics"],
        "constraint_violation": split_results["test"]["constraint_violation"],
        "split_results": split_results,
        "dataset_metadata": dataset.metadata,
        "config": {
            "args": vars(args),
            "flow_scaler": {"mean": dataset.flow_mean, "std": dataset.flow_std},
            "new_edge_fill_policy": "batch_retained_old_flow_mean",
        },
    }

    save_json(output_dir / "summary.json", summary)
    print(
        "[OldFlowMeanBaseline] "
        f"NormRMSE={summary['test_metrics']['normalized']['all_edges']['rmse']:.6f} "
        f"RealRMSE={summary['test_metrics']['real']['all_edges']['rmse']:.6f} "
        f"RealWMAPE={summary['test_metrics']['real']['all_edges']['wmape']:.6f} "
        f"OldWMAPE={summary['test_metrics']['real']['old_edges']['wmape']:.6f} "
        f"NewWMAPE={summary['test_metrics']['real']['new_edges']['wmape']:.6f} "
        f"RelCon={summary['constraint_violation']['relcon']:.6f} "
        f"TimeMsPerGraph={summary['test_time']['forward_milliseconds_per_graph']:.6f}"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
