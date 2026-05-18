"""Plot the new-edge prediction penalty for ST-PINN GatedGCN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "Sioux Falls": "results/ours/network-pairs-topology-stpinn_gatedgcn_siouxfalls_10000/0/summary.json",
    "EMA network": "results/ours/network-pairs-topology-stpinn_gatedgcn_ema_10000/0/summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot new-edge WMAPE penalty relative to retained-edge WMAPE."
    )
    parser.add_argument(
        "--output",
        default="results/edge_type/new_edge_penalty.png",
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_edge_wmapes(root: Path, relative_path: str) -> tuple[float, float]:
    path = root / relative_path
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    real_metrics = summary["test_metrics"]["real"]
    old_wmape = float(real_metrics["old_edges"]["wmape"])
    new_wmape = float(real_metrics["new_edges"]["wmape"])
    return old_wmape, new_wmape


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for network, path in RUNS.items():
        old_wmape, new_wmape = load_edge_wmapes(root, path)
        rows.append(
            {
                "network": network,
                "old_wmape": old_wmape,
                "new_wmape": new_wmape,
                "ratio": new_wmape / old_wmape,
            }
        )
    return rows


def plot(rows: list[dict], output: Path, dpi: int) -> None:
    networks = [row["network"] for row in rows]
    ratios = np.array([row["ratio"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(networks))
    colors = ["#b45f3c", "#4f7f9f"]

    bars = ax.bar(x, ratios, width=0.52, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)))

    ax.set_xticks(x)
    ax.set_xticklabels(networks)
    ax.set_ylabel("New-edge / retained-edge WMAPE")
    ax.set_ylim(0.0, max(1.8, float(ratios.max()) + 0.2))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, row in zip(bars, rows):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.04,
            f"{height:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.08,
            f"old {row['old_wmape'] * 100:.1f}%\nnew {row['new_wmape'] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    ax.text(
        0.98,
        1.01,
        "ratio = 1",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    rows = collect_rows(root)
    output = (root / args.output).resolve()
    plot(rows, output, args.dpi)
    print(output)
    for row in rows:
        print(
            f"{row['network']}: ratio={row['ratio']:.3f}, "
            f"old WMAPE={row['old_wmape'] * 100:.2f}%, "
            f"new WMAPE={row['new_wmape'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
