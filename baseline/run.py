from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baseline.common import canonical_model_name, denormalize_flow, maybe_cuda_synchronize, set_seed
from baseline.data import build_loader, load_dataset_bundle
from baseline.losses import LossConfig, compute_weighted_supervised_loss
from baseline.metrics import SplitMetrics
from baseline.mlp_baseline import MLPBaseline
from baseline.model_common import ModelConfig, OURS_HIDDEN_DIM
from baseline.node_centric_gnn import NodeCentricGNN
from baseline.single_topology_gatedgcn import SingleTopologyGatedGCN
from constraint_violation.metrics import ConstraintViolationAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone baseline models on the network-pairs PyG dataset."
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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Baseline architecture to train: mlp_baseline, single_topology_gatedgcn, or NodeCentricGNN.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=OURS_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true", help="Enable residual GatedGCN connections.")
    parser.add_argument("--num_layers_old", type=int, default=3)
    parser.add_argument("--num_layers_new", type=int, default=3)
    parser.add_argument("--mlp_depth", type=int, default=3)
    parser.add_argument("--loss_name", type=str, default="l1", choices=["l1", "smoothl1", "mse"])
    parser.add_argument("--lambda_old", type=float, default=1.0)
    parser.add_argument("--lambda_new_start", type=float, default=1.0)
    parser.add_argument("--lambda_new_final", type=float, default=1.0)
    parser.add_argument("--lambda_new_warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--lambda_new_schedule",
        type=str,
        default="linear",
        choices=["linear", "constant", "none"],
    )
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _move_batch_to_device(batch, device: torch.device):
    return batch.to(device)


def build_model(model_name: str, config: ModelConfig) -> torch.nn.Module:
    canonical = canonical_model_name(model_name)
    if canonical == "mlp_baseline":
        return MLPBaseline(config)
    if canonical == "single_topology_gatedgcn":
        return SingleTopologyGatedGCN(config)
    if canonical == "node_centric_gnn":
        return NodeCentricGNN(config)
    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: LossConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_old": 0.0,
        "loss_new": 0.0,
        "num_batches": 0,
    }

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss, stats = compute_weighted_supervised_loss(
            pred=pred,
            true=batch.y,
            new_edge_mask=getattr(batch, "new_edge_mask", None),
            config=loss_config,
            epoch=epoch,
        )
        loss.backward()
        optimizer.step()

        running["loss_total"] += stats["loss_total"]
        running["loss_old"] += stats["loss_old"]
        running["loss_new"] += stats["loss_new"]
        running["num_batches"] += 1

    denom = max(running["num_batches"], 1)
    return {
        "loss_total": running["loss_total"] / denom,
        "loss_old": running["loss_old"] / denom,
        "loss_new": running["loss_new"] / denom,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    loss_config: LossConfig,
    epoch: int,
    flow_mean: float,
    flow_std: float,
    measure_inference_time: bool = False,
) -> dict:
    model.eval()
    split_metrics = SplitMetrics()
    loss_total = 0.0
    loss_old = 0.0
    loss_new = 0.0
    num_batches = 0
    total_graphs = 0
    total_forward_seconds = 0.0
    constraint_metrics = ConstraintViolationAccumulator()

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        maybe_cuda_synchronize(device)
        start = time.perf_counter()
        pred = model(batch)
        maybe_cuda_synchronize(device)
        if measure_inference_time:
            total_forward_seconds += time.perf_counter() - start

        _, loss_stats = compute_weighted_supervised_loss(
            pred=pred,
            true=batch.y,
            new_edge_mask=getattr(batch, "new_edge_mask", None),
            config=loss_config,
            epoch=epoch,
        )

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

        loss_total += loss_stats["loss_total"]
        loss_old += loss_stats["loss_old"]
        loss_new += loss_stats["loss_new"]
        num_batches += 1
        total_graphs += int(getattr(batch, "num_graphs", 1))

    denom = max(num_batches, 1)
    forward_ms_per_graph = 0.0 if total_graphs == 0 else (total_forward_seconds * 1000.0 / total_graphs)
    return {
        "loss": {
            "loss_total": loss_total / denom,
            "loss_old": loss_old / denom,
            "loss_new": loss_new / denom,
        },
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
    train_loader = build_loader(dataset.train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_loader(dataset.val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_loader(dataset.test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual=bool(args.residual),
        num_layers_old=args.num_layers_old,
        num_layers_new=args.num_layers_new,
        mlp_depth=args.mlp_depth,
    )
    loss_config = LossConfig(
        loss_name=args.loss_name,
        lambda_old=args.lambda_old,
        lambda_new_start=args.lambda_new_start,
        lambda_new_final=args.lambda_new_final,
        lambda_new_warmup_epochs=args.lambda_new_warmup_epochs,
        lambda_new_schedule=args.lambda_new_schedule,
    )

    model = build_model(args.model, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_epoch = 0
    best_val_rmse = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"
    epoch_seconds: list[float] = []

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_config=loss_config,
            epoch=epoch,
        )
        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_config=loss_config,
            epoch=epoch,
            flow_mean=dataset.flow_mean,
            flow_std=dataset.flow_std,
            measure_inference_time=False,
        )
        test_epoch_stats = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            loss_config=loss_config,
            epoch=epoch,
            flow_mean=dataset.flow_mean,
            flow_std=dataset.flow_std,
            measure_inference_time=False,
        )
        epoch_duration = time.perf_counter() - epoch_start
        epoch_seconds.append(epoch_duration)

        val_rmse = val_stats["metrics"]["normalized"]["all_edges"]["rmse"]
        history.append(
            {
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
                "test": test_epoch_stats,
                "val_loss": val_stats["loss"],
                "val_metrics": val_stats["metrics"],
                "test_loss": test_epoch_stats["loss"],
                "test_metrics": test_epoch_stats["metrics"],
                "epoch_seconds": epoch_duration,
            }
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "model_config": vars(model_config),
                    "loss_config": vars(loss_config),
                    "best_val_rmse": best_val_rmse,
                },
                best_ckpt_path,
            )

        print(
            f"> Epoch {epoch}: train_loss={train_stats['loss_total']:.4f} "
            f"val_loss={val_stats['loss']['loss_total']:.4f} "
            f"test_loss={test_epoch_stats['loss']['loss_total']:.4f} "
            f"val_rmse_norm={val_rmse:.4f} "
            f"best_epoch={best_epoch} "
            f"best_val_rmse_norm={best_val_rmse:.4f}"
        )

    total_train_seconds = time.perf_counter() - train_start
    checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_stats = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        loss_config=loss_config,
        epoch=best_epoch,
        flow_mean=dataset.flow_mean,
        flow_std=dataset.flow_std,
        measure_inference_time=True,
    )

    summary = {
        "model": canonical_model_name(args.model),
        "dataset_dir": str(dataset.dataset_dir),
        "device_used": str(device),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "selection_metric": "validation.normalized.all_edges.rmse",
        "training_time": {
            "total_seconds": total_train_seconds,
            "average_epoch_seconds": 0.0 if not epoch_seconds else sum(epoch_seconds) / len(epoch_seconds),
            "num_epochs": int(args.epochs),
        },
        "test_time": test_stats["timing"],
        "validation_best_rmse_normalized_all_edges": float(best_val_rmse),
        "test_metrics": test_stats["metrics"],
        "test_loss": test_stats["loss"],
        "constraint_violation": test_stats["constraint_violation"],
        "dataset_metadata": dataset.metadata,
        "config": {
            "args": vars(args),
            "model_config": vars(model_config),
            "loss_config": vars(loss_config),
            "flow_scaler": {"mean": dataset.flow_mean, "std": dataset.flow_std},
        },
    }

    save_json(output_dir / "history.json", {"epochs": history})
    save_json(output_dir / "summary.json", summary)
    print(
        "[ConstraintViolation] "
        f"NormRMSE={test_stats['metrics']['normalized']['all_edges']['rmse']:.6f} "
        f"RealWMAPE={test_stats['metrics']['real']['all_edges']['wmape']:.6f} "
        f"ConMAE={test_stats['constraint_violation']['con_mae']:.6f} "
        f"ConRMSE={test_stats['constraint_violation']['con_rmse']:.6f} "
        f"ConMax={test_stats['constraint_violation']['con_max']:.6f} "
        f"RelCon={test_stats['constraint_violation']['relcon']:.6f} "
        f"RelConP95={test_stats['constraint_violation']['relcon_p95']:.6f}"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
