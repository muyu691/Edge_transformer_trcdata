"""Plot one qualitative flow redistribution example from a trained ST-PINN GatedGCN checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import graphgps  # noqa: F401
from constraint_violation.metrics import compute_constraint_violation_stats
from graphgps.metric_wrapper import wmape
from graphgps.network.topology_model import NetworkPairsTopologyModel
from graphgps.utils import match_edge_indices


DATASETS = {
    "siouxfalls": {
        "network_name": "SiouxFalls",
        "dataset_dir": "siouxfalls_pyg_newpolicy_lhs",
        "run_dir": "network-pairs-topology-stpinn_gatedgcn_siouxfalls_10000",
    },
    "ema": {
        "network_name": "EMA",
        "dataset_dir": "ema_pyg_newpolicy_lhs",
        "run_dir": "network-pairs-topology-stpinn_gatedgcn_ema_10000",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a qualitative flow redistribution figure."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="siouxfalls")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--sample-index",
        default="auto",
        help="Zero-based sample index, or 'auto' to select a median-error test graph.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=120,
        help="Number of graphs scanned when --sample-index auto is used.",
    )
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path.")
    parser.add_argument("--dataset-dir", default="", help="Optional processed dataset path.")
    parser.add_argument(
        "--config",
        default="configs/GatedGCN/network-pairs-topology.yaml",
        help="GraphGym config path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output prefix without extension.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_dir(root: Path, args: argparse.Namespace) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir).resolve()
    return root / "create_sioux_data" / "processed_data" / DATASETS[args.dataset]["dataset_dir"]


def load_split(dataset_dir: Path, split: str) -> list:
    path = dataset_dir / f"{split}_dataset.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def configure_graphgym(root: Path, dataset_dir: Path, args: argparse.Namespace):
    dataset_info = DATASETS[args.dataset]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    opts = [
        "dataset.network_name",
        dataset_info["network_name"],
        "dataset.dir",
        str(dataset_dir),
        "dataset.processed_root",
        str(dataset_dir.parent),
        "accelerator",
        args.device,
        "topology_gnn.hidden_dim",
        "128",
        "topology_gnn.local_backbone",
        "gatedgcn",
        "topology_gnn.attention_every_k_steps",
        "1",
        "topology_gnn.enable_global_attn",
        "True",
        "topology_gnn.ffn_type",
        "swiglu",
        "topology_gnn.ffn_mult",
        "8/3",
        "topology_gnn.norm_type",
        "rmsnorm",
        "topology_gnn.norm_position",
        "pre",
    ]
    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path), opts=opts))

    flow_scaler = load_pickle(dataset_dir / "scalers" / "flow_scaler.pkl")
    cfg.dataset.flow_mean = float(flow_scaler.mean_[0])
    cfg.dataset.flow_std = float(flow_scaler.scale_[0])
    return flow_scaler


def resolve_checkpoint(root: Path, args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).resolve()

    run_dir = (
        root
        / "results"
        / "ours"
        / DATASETS[args.dataset]["run_dir"]
        / "0"
    )
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    best_epoch = int(summary["best_epoch"])
    checkpoint = run_dir / "ckpt" / f"{best_epoch}.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> NetworkPairsTopologyModel:
    model = NetworkPairsTopologyModel(0, 0).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"]
    if all(key.startswith("model.") for key in state):
        state = {key[len("model.") :]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_one(model: NetworkPairsTopologyModel, data, device: torch.device):
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        pred_norm, true_norm = model(batch)
    mean = float(cfg.dataset.flow_mean)
    std = float(cfg.dataset.flow_std)
    pred_real = pred_norm.view(-1).detach().cpu() * std + mean
    true_real = true_norm.view(-1).detach().cpu() * std + mean
    return pred_real, true_real


def select_sample(model, data_list: list, args: argparse.Namespace, device: torch.device) -> int:
    if str(args.sample_index).lower() != "auto":
        idx = int(args.sample_index)
        if idx < 0 or idx >= len(data_list):
            raise IndexError(f"sample-index {idx} outside split size {len(data_list)}")
        return idx

    limit = min(max(args.search_limit, 1), len(data_list))
    values = []
    for idx in range(limit):
        pred_real, true_real = predict_one(model, data_list[idx], device)
        values.append((idx, float(wmape(pred_real, true_real).item())))
    median = float(np.median([value for _, value in values]))
    return min(values, key=lambda item: abs(item[1] - median))[0]


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def inverse_edge_attrs(data, dataset_dir: Path) -> np.ndarray:
    if hasattr(data, "edge_attr_new_real"):
        return tensor_to_numpy(data.edge_attr_new_real).astype(float)
    scaler = load_pickle(dataset_dir / "scalers" / "attr_scaler.pkl")
    return scaler.inverse_transform(tensor_to_numpy(data.edge_attr_new).astype(float))


def projected_old_flow(data) -> np.ndarray:
    mean = float(cfg.dataset.flow_mean)
    std = float(cfg.dataset.flow_std)
    old_flow_real = tensor_to_numpy(data.flow_old).reshape(-1) * std + mean
    match_idx = match_edge_indices(
        edge_index_old=data.edge_index_old,
        edge_index_new=data.edge_index_new,
        total_nodes=int(data.num_nodes),
    )
    match_idx_np = tensor_to_numpy(match_idx).astype(int)
    projected = np.zeros(len(match_idx_np), dtype=float)
    retained = match_idx_np >= 0
    projected[retained] = old_flow_real[match_idx_np[retained]]
    return projected


def edge_array(data, field: str) -> np.ndarray:
    return tensor_to_numpy(getattr(data, field)).T.astype(int)


def collapse_values(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[tuple[int, int], list[float]] = {}
    for edge, value in zip(edges, values):
        key = tuple(sorted((int(edge[0]), int(edge[1]))))
        buckets.setdefault(key, []).append(float(value))
    collapsed_edges = []
    collapsed_values = []
    for key in sorted(buckets):
        collapsed_edges.append(key)
        collapsed_values.append(float(np.mean(buckets[key])))
    return np.asarray(collapsed_edges, dtype=int), np.asarray(collapsed_values, dtype=float)


def removed_edges(old_edges: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    new_set = {tuple(edge) for edge in new_edges.tolist()}
    return np.asarray(
        [edge for edge in old_edges.tolist() if tuple(edge) not in new_set],
        dtype=int,
    )


def normalized_layout(num_nodes: int, old_edges: np.ndarray, new_edges: np.ndarray, seed: int):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(map(tuple, old_edges.tolist()))
    graph.add_edges_from(map(tuple, new_edges.tolist()))
    pos_raw = nx.spring_layout(graph, seed=seed, iterations=300)
    xy = np.array([pos_raw[i] for i in range(num_nodes)], dtype=float)
    mins = xy.min(axis=0)
    spans = np.maximum(xy.max(axis=0) - mins, 1e-12)
    xy = (xy - mins) / spans
    return {idx: tuple(xy[idx]) for idx in range(num_nodes)}


def edge_segments(edges: np.ndarray, pos: dict[int, tuple[float, float]]) -> list:
    return [[pos[int(u)], pos[int(v)]] for u, v in edges]


def edge_curve_radius(edge: np.ndarray | tuple[int, int], edge_set: set[tuple[int, int]]) -> float:
    u, v = int(edge[0]), int(edge[1])
    if (v, u) not in edge_set:
        return 0.0
    return 0.08 if u < v else -0.08


def draw_directed_edges(
    ax,
    edges: np.ndarray,
    values: np.ndarray,
    widths: np.ndarray,
    pos: dict[int, tuple[float, float]],
    cmap,
    norm,
    edge_set: set[tuple[int, int]],
    *,
    color: str | None = None,
    linestyle: str | tuple = "solid",
    alpha: float = 0.96,
    zorder: int = 2,
    mutation_scale: float = 8.0,
) -> None:
    for edge, value, width in zip(edges, values, widths):
        u, v = int(edge[0]), int(edge[1])
        edge_color = color if color is not None else cmap(norm(float(value)))
        arrow = FancyArrowPatch(
            pos[u],
            pos[v],
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=float(width),
            linestyle=linestyle,
            color=edge_color,
            alpha=alpha,
            zorder=zorder,
            shrinkA=5.0,
            shrinkB=5.0,
            connectionstyle=f"arc3,rad={edge_curve_radius((u, v), edge_set)}",
        )
        ax.add_patch(arrow)


def width_from_capacity(capacity: np.ndarray) -> np.ndarray:
    if capacity.size == 0:
        return capacity
    cmin = float(np.min(capacity))
    cmax = float(np.max(capacity))
    if cmax <= cmin:
        return np.full_like(capacity, 1.6, dtype=float)
    scaled = (capacity - cmin) / (cmax - cmin)
    return 0.8 + 2.2 * np.clip(scaled, 0.0, 1.0)


def draw_panel(
    ax,
    edges: np.ndarray,
    values: np.ndarray,
    widths: np.ndarray,
    pos: dict[int, tuple[float, float]],
    title: str,
    cmap: str,
    norm,
    removed: np.ndarray,
    edge_set: set[tuple[int, int]],
    removed_edge_set: set[tuple[int, int]],
):
    cmap_obj = plt.get_cmap(cmap)
    draw_directed_edges(
        ax=ax,
        edges=edges,
        values=values,
        widths=widths,
        pos=pos,
        cmap=cmap_obj,
        norm=norm,
        edge_set=edge_set,
        zorder=2,
    )

    if removed.size:
        draw_directed_edges(
            ax=ax,
            edges=removed,
            values=np.ones(len(removed), dtype=float),
            widths=np.full(len(removed), 2.0, dtype=float),
            pos=pos,
            cmap=cmap_obj,
            norm=norm,
            edge_set=removed_edge_set,
            color="#c93a2f",
            linestyle=(0, (4, 2)),
            alpha=0.95,
            zorder=3,
            mutation_scale=9.0,
        )

    xy = np.array([pos[node] for node in sorted(pos)], dtype=float)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=28,
        facecolor="white",
        edgecolor="#263238",
        linewidth=0.9,
        zorder=4,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal")
    ax.grid(True, color="#e7e7e7", linewidth=0.6)
    ax.set_xlabel("Normalized layout x")
    ax.set_ylabel("Normalized layout y")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)


def make_figure(
    data,
    pred_real: torch.Tensor,
    true_real: torch.Tensor,
    dataset_dir: Path,
    args: argparse.Namespace,
    sample_index: int,
    output_prefix: Path,
):
    old_edges = edge_array(data, "edge_index_old")
    new_edges = edge_array(data, "edge_index_new")
    attrs_real = inverse_edge_attrs(data, dataset_dir)
    capacities = attrs_real[:, 0].reshape(-1)

    old_projected = projected_old_flow(data)
    pred_np = tensor_to_numpy(pred_real).reshape(-1)
    true_np = tensor_to_numpy(true_real).reshape(-1)
    true_delta = np.abs(true_np - old_projected)
    pred_delta = np.abs(pred_np - old_projected)
    abs_error = np.abs(pred_np - true_np)

    display_edges = new_edges
    true_display = true_delta
    pred_display = pred_delta
    error_display = abs_error
    capacity_display = capacities
    removed_display = removed_edges(old_edges, new_edges)
    new_edge_set = set(map(tuple, new_edges.tolist()))
    old_edge_set = set(map(tuple, old_edges.tolist()))

    widths = width_from_capacity(capacity_display)
    pos = normalized_layout(int(data.num_nodes), old_edges, new_edges, args.seed)

    delta_limit = max(
        float(np.max(true_display)) if true_display.size else 1.0,
        float(np.max(pred_display)) if pred_display.size else 1.0,
        1.0,
    )
    error_limit = max(float(np.max(error_display)) if error_display.size else 1.0, 1.0)

    delta_norm = plt.Normalize(vmin=0.0, vmax=delta_limit)
    error_norm = plt.Normalize(vmin=0.0, vmax=error_limit)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
    c0 = draw_panel(
        axes[0],
        display_edges,
        true_display,
        widths,
        pos,
        "Observed flow redistribution",
        "viridis",
        delta_norm,
        removed_display,
        new_edge_set,
        old_edge_set,
    )
    c1 = draw_panel(
        axes[1],
        display_edges,
        pred_display,
        widths,
        pos,
        "Predicted flow redistribution",
        "viridis",
        delta_norm,
        removed_display,
        new_edge_set,
        old_edge_set,
    )
    c2 = draw_panel(
        axes[2],
        display_edges,
        error_display,
        widths,
        pos,
        "Absolute prediction error",
        "viridis",
        error_norm,
        removed_display,
        new_edge_set,
        old_edge_set,
    )

    fig.colorbar(c1, ax=axes[:2], shrink=0.86, label="Absolute flow change")
    fig.colorbar(c2, ax=axes[2], shrink=0.86, label="Absolute error")

    graph_wmape = float(wmape(pred_real, true_real).item())
    residual_stats = compute_constraint_violation_stats(
        pred_real=pred_real,
        edge_index_new=data.edge_index_new,
        net_demand=data.net_demand,
        node_batch=torch.zeros(int(data.num_nodes), dtype=torch.long),
        ptr=torch.tensor([0, int(data.num_nodes)], dtype=torch.long),
    )[0]

    fig.suptitle(
        f"{DATASETS[args.dataset]['network_name']} test graph {sample_index}: "
        f"WMAPE={100.0 * graph_wmape:.2f}%, RelCon={residual_stats['relcon']:.3f}",
        fontsize=11,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_prefix.with_suffix(".json")
    payload = {
        "dataset": args.dataset,
        "sample_index": int(sample_index),
        "wmape": graph_wmape,
        "relcon": float(residual_stats["relcon"]),
        "con_mae": float(residual_stats["con_mae"]),
        "con_rmse": float(residual_stats["con_rmse"]),
        "num_new_edges": int(tensor_to_numpy(data.new_edge_mask).sum()),
        "num_edges": int(data.edge_index_new.shape[1]),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {metrics_path}")


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_dir = resolve_dataset_dir(root, args)
    configure_graphgym(root, dataset_dir, args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = resolve_checkpoint(root, args)
    model = load_model(checkpoint, device)
    data_list = load_split(dataset_dir, args.split)
    sample_index = select_sample(model, data_list, args, device)
    data = data_list[sample_index]
    pred_real, true_real = predict_one(model, data, device)

    if args.output_prefix:
        output_prefix = Path(args.output_prefix)
    else:
        output_prefix = (
            root
            / "results"
            / "flow_redistribution"
            / f"{args.dataset}_{args.split}_graph{sample_index}_flow_redistribution"
        )
    if not output_prefix.is_absolute():
        output_prefix = root / output_prefix

    make_figure(
        data=data,
        pred_real=pred_real,
        true_real=true_real,
        dataset_dir=dataset_dir,
        args=args,
        sample_index=sample_index,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    main()
