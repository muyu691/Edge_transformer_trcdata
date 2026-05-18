"""Plot original and reconfigured traffic networks from a PyG sample."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


ATTR_COLUMNS = {
    "capacity": 0,
    "speed": 1,
    "length": 2,
}

DATASET_DIRS = {
    "siouxfalls": "siouxfalls_pyg_newpolicy_lhs",
    "ema": "ema_pyg_newpolicy_lhs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a thesis-style topology reconfiguration figure."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DIRS),
        default="siouxfalls",
        help="Dataset to visualize.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="Optional processed PyG dataset directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split used to select the graph pair.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based sample index within the selected split.",
    )
    parser.add_argument(
        "--attribute",
        choices=sorted(ATTR_COLUMNS),
        default="capacity",
        help="Edge attribute used for the color scale.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output image path. Defaults to results/network_visualizations/...",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used for the spring layout.",
    )
    parser.add_argument(
        "--label-nodes",
        action="store_true",
        help="Draw node ids next to the markers.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_dir(root: Path, dataset: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return root / "create_sioux_data" / "processed_data" / DATASET_DIRS[dataset]


def load_split(dataset_dir: Path, split: str) -> list:
    split_path = dataset_dir / f"{split}_dataset.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    try:
        return torch.load(split_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(split_path, map_location="cpu")


def load_attr_scaler(dataset_dir: Path):
    scaler_path = dataset_dir / "scalers" / "attr_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing attribute scaler: {scaler_path}")
    with scaler_path.open("rb") as handle:
        return pickle.load(handle)


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_real_attrs(data, field: str, scaler) -> np.ndarray:
    real_field = f"{field}_real"
    if hasattr(data, real_field):
        return tensor_to_numpy(getattr(data, real_field)).astype(float)

    attrs = tensor_to_numpy(getattr(data, field)).astype(float)
    return scaler.inverse_transform(attrs)


def edge_array(data, field: str) -> np.ndarray:
    return tensor_to_numpy(getattr(data, field)).T.astype(int)


def normalized_layout(num_nodes: int, old_edges: np.ndarray, new_edges: np.ndarray, seed: int):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(map(tuple, old_edges.tolist()))
    graph.add_edges_from(map(tuple, new_edges.tolist()))

    raw_pos = nx.spring_layout(graph, seed=seed, iterations=300)
    xy = np.array([raw_pos[i] for i in range(num_nodes)], dtype=float)
    mins = xy.min(axis=0)
    spans = np.maximum(xy.max(axis=0) - mins, 1e-12)
    xy = (xy - mins) / spans
    return {i: tuple(xy[i]) for i in range(num_nodes)}


def edge_segments(edges: np.ndarray, pos: dict[int, tuple[float, float]]) -> list:
    return [[pos[int(u)], pos[int(v)]] for u, v in edges]


def collapse_parallel_edges(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse opposite directions into one displayed road segment."""
    buckets: dict[tuple[int, int], list[float]] = {}
    for (u, v), value in zip(edges, values):
        key = tuple(sorted((int(u), int(v))))
        buckets.setdefault(key, []).append(float(value))

    collapsed_edges = []
    collapsed_values = []
    for key in sorted(buckets):
        collapsed_edges.append(key)
        collapsed_values.append(float(np.mean(buckets[key])))

    return np.asarray(collapsed_edges, dtype=int), np.asarray(collapsed_values, dtype=float)


def collapse_edge_mask(edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Collapse a directed edge mask into unique displayed road segments."""
    keys = {tuple(sorted((int(u), int(v)))) for (u, v), keep in zip(edges, mask) if bool(keep)}
    return np.asarray(sorted(keys), dtype=int)


def removed_display_segments(old_edges: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    """Return old road segments that are absent from the displayed new network.

    The figure draws a single line for opposite directions. Therefore a segment
    should be marked as removed only when neither direction remains in the
    reconfigured network.
    """
    new_segments = {tuple(sorted((int(u), int(v)))) for u, v in new_edges}
    keys = {
        tuple(sorted((int(u), int(v))))
        for u, v in old_edges
        if tuple(sorted((int(u), int(v)))) not in new_segments
    }
    return np.asarray(sorted(keys), dtype=int)


def scaled_widths(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.full_like(values, 1.4, dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    return 0.65 + 1.85 * np.clip(scaled, 0.0, 1.0)


def draw_edge_collection(
    ax,
    edges: np.ndarray,
    values: np.ndarray,
    pos: dict[int, tuple[float, float]],
    cmap,
    norm,
    linewidths: np.ndarray,
    linestyle: str = "solid",
    alpha: float = 0.95,
    zorder: int = 1,
):
    if len(edges) == 0:
        return None
    collection = LineCollection(
        edge_segments(edges, pos),
        cmap=cmap,
        norm=norm,
        linewidths=linewidths,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_array(values)
    ax.add_collection(collection)
    return collection


def edge_curve_radius(edge: np.ndarray | tuple[int, int], edge_set: set[tuple[int, int]]) -> float:
    """Curve opposite directed edges so both directions are visible."""
    u, v = int(edge[0]), int(edge[1])
    if (v, u) not in edge_set:
        return 0.0
    return 0.09 if u < v else -0.09


def draw_directed_edges(
    ax,
    edges: np.ndarray,
    values: np.ndarray,
    pos: dict[int, tuple[float, float]],
    cmap,
    norm,
    linewidths: np.ndarray,
    edge_set: set[tuple[int, int]],
    linestyle: str | tuple = "solid",
    alpha: float = 0.95,
    zorder: int = 1,
    color: str | None = None,
    mutation_scale: float = 8.0,
):
    for edge, value, linewidth in zip(edges, values, linewidths):
        u, v = int(edge[0]), int(edge[1])
        edge_color = color if color is not None else cmap(norm(float(value)))
        arrow = FancyArrowPatch(
            pos[u],
            pos[v],
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=float(linewidth),
            linestyle=linestyle,
            color=edge_color,
            alpha=alpha,
            zorder=zorder,
            shrinkA=4.5,
            shrinkB=4.5,
            connectionstyle=f"arc3,rad={edge_curve_radius((u, v), edge_set)}",
        )
        ax.add_patch(arrow)


def draw_fixed_edges(
    ax,
    edges: np.ndarray,
    pos: dict[int, tuple[float, float]],
    color: str,
    linewidth: float,
    linestyle: str,
    alpha: float,
    zorder: int,
):
    if len(edges) == 0:
        return
    collection = LineCollection(
        edge_segments(edges, pos),
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)


def setup_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Normalized layout x")
    ax.set_ylabel("Normalized layout y")
    ax.grid(True, color="#e8e8e8", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)


def draw_nodes(ax, pos, node_labels, label_nodes: bool) -> None:
    xy = np.array([pos[i] for i in range(len(pos))])
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=12,
        facecolor="white",
        edgecolor="#1f2933",
        linewidth=0.55,
        zorder=5,
    )
    if label_nodes:
        for idx, label in enumerate(node_labels):
            ax.text(
                xy[idx, 0] + 0.008,
                xy[idx, 1] + 0.008,
                str(label),
                fontsize=6,
                color="#263238",
                zorder=6,
            )


def add_panel_note(ax, lines: list[str]) -> None:
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#c8c8c8",
            "alpha": 0.88,
        },
        zorder=8,
    )


def default_output_path(root: Path, dataset: str, split: str, sample_index: int, attribute: str) -> Path:
    return (
        root
        / "results"
        / "network_visualizations"
        / f"{dataset}_{split}_sample{sample_index}_{attribute}_reconfiguration.png"
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_dir = resolve_dataset_dir(root, args.dataset, args.dataset_dir)
    scaler = load_attr_scaler(dataset_dir)
    split_data = load_split(dataset_dir, args.split)

    if args.sample_index < 0 or args.sample_index >= len(split_data):
        raise IndexError(
            f"sample-index={args.sample_index} is outside split size {len(split_data)}"
        )

    data = split_data[args.sample_index]
    old_edges = edge_array(data, "edge_index_old")
    new_edges = edge_array(data, "edge_index_new")
    old_attrs = get_real_attrs(data, "edge_attr_old", scaler)
    new_attrs = get_real_attrs(data, "edge_attr_new", scaler)

    attr_idx = ATTR_COLUMNS[args.attribute]
    old_values = old_attrs[:, attr_idx]
    new_values = new_attrs[:, attr_idx]

    old_edge_set = set(map(tuple, old_edges.tolist()))
    new_edge_set = set(map(tuple, new_edges.tolist()))
    old_removed = np.array([tuple(edge) not in new_edge_set for edge in old_edges], dtype=bool)

    if hasattr(data, "new_edge_mask"):
        new_added = tensor_to_numpy(data.new_edge_mask).astype(bool)
    else:
        new_added = np.array([tuple(edge) not in old_edge_set for edge in new_edges], dtype=bool)

    num_nodes = int(getattr(data, "num_nodes", len(tensor_to_numpy(data.x))))
    node_ids = (
        tensor_to_numpy(data.node_ids).astype(int).tolist()
        if hasattr(data, "node_ids")
        else list(range(1, num_nodes + 1))
    )

    pos = normalized_layout(num_nodes, old_edges, new_edges, args.seed)
    old_edges_plot, old_values_plot = old_edges, old_values
    new_edges_plot, new_values_plot = new_edges, new_values
    removed_edges_plot = old_edges[old_removed]
    removed_values_plot = old_values[old_removed]

    vmin = float(min(old_values_plot.min(), new_values_plot.min()))
    vmax = float(max(old_values_plot.max(), new_values_plot.max()))
    cmap = mpl.colormaps.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.9, bottom=0.2, top=0.81, wspace=0.24)

    draw_directed_edges(
        axes[0],
        old_edges_plot,
        old_values_plot,
        pos,
        cmap,
        norm,
        scaled_widths(old_values_plot, vmin, vmax),
        old_edge_set,
        alpha=0.9,
        zorder=2,
        mutation_scale=7.0,
    )
    draw_directed_edges(
        axes[0],
        removed_edges_plot,
        removed_values_plot,
        pos,
        cmap,
        norm,
        np.full(len(removed_edges_plot), 2.2, dtype=float),
        old_edge_set,
        color="#d62728",
        linestyle=(0, (4, 2)),
        alpha=0.98,
        zorder=3,
        mutation_scale=8.5,
    )
    draw_nodes(axes[0], pos, node_ids, args.label_nodes)
    setup_axis(axes[0], "Original network")
    add_panel_note(
        axes[0],
        [
            f"links: {len(old_edges)}",
            f"removed directed links: {int(old_removed.sum())}",
        ],
    )

    draw_directed_edges(
        axes[1],
        new_edges_plot,
        new_values_plot,
        pos,
        cmap,
        norm,
        scaled_widths(new_values_plot, vmin, vmax),
        new_edge_set,
        alpha=0.9,
        zorder=2,
        mutation_scale=7.0,
    )
    draw_nodes(axes[1], pos, node_ids, args.label_nodes)
    setup_axis(axes[1], "Reconfigured network")
    add_panel_note(
        axes[1],
        [
            f"links: {len(new_edges)}",
        ],
    )

    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=axes.ravel().tolist(), shrink=0.84, pad=0.025)
    label = {
        "capacity": "Link capacity",
        "speed": "Link speed",
        "length": "Link length",
    }[args.attribute]
    cbar.set_label(label)

    axes[0].plot(
        [],
        [],
        color="#d62728",
        linewidth=1.8,
        linestyle=(0, (4, 2)),
        label="removed directed link",
    )
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8)

    network_name = {
        "siouxfalls": "Sioux Falls",
        "ema": "EMA",
    }.get(args.dataset, str(getattr(data, "network_name", args.dataset)))
    fig.suptitle(f"{network_name} network reconfiguration", fontsize=13)

    output = Path(args.output).resolve() if args.output else default_output_path(
        root, args.dataset, args.split, args.sample_index, args.attribute
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
