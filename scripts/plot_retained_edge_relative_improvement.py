"""Plot retained-edge relative improvement of ST-PINN over the best baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "Sioux Falls": {
        "stpinn": "results/ours/network-pairs-topology-stpinn_gatedgcn_siouxfalls_10000/0/summary.json",
        "baselines": {
            "MLP": "results/convergence_curves/conv_mlp_siouxfalls_10000/summary.json",
            "Single-topology GatedGCN": "results/convergence_curves/conv_gated_siouxfalls_10000/summary.json",
            "Node-centric GNN": "results/convergence_curves/conv_node_siouxfalls_10000/summary.json",
        },
    },
    "EMA network": {
        "stpinn": "results/ours/network-pairs-topology-stpinn_gatedgcn_ema_10000/0/summary.json",
        "baselines": {
            "MLP": "results/convergence_curves/conv_mlp_ema_10000/summary.json",
            "Single-topology GatedGCN": "results/convergence_curves/conv_gated_ema_10000/summary.json",
            "Node-centric GNN": "results/convergence_curves/conv_node_ema_10000/summary.json",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot retained-edge relative improvement for the Results chapter."
    )
    parser.add_argument(
        "--output",
        default="results/edge_type/retained_edge_relative_improvement.png",
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_old_wmape(root: Path, relative_path: str) -> float:
    path = root / relative_path
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return float(summary["test_metrics"]["real"]["old_edges"]["wmape"])


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for network, spec in RUNS.items():
        stpinn_wmape = load_old_wmape(root, spec["stpinn"])
        baseline_values = {
            name: load_old_wmape(root, path)
            for name, path in spec["baselines"].items()
        }
        best_baseline = min(baseline_values, key=baseline_values.get)
        best_baseline_wmape = baseline_values[best_baseline]
        improvement = (best_baseline_wmape - stpinn_wmape) / best_baseline_wmape * 100.0
        rows.append(
            {
                "network": network,
                "best_baseline": best_baseline,
                "baseline_wmape": best_baseline_wmape,
                "stpinn_wmape": stpinn_wmape,
                "improvement": improvement,
            }
        )
    return rows


def plot(rows: list[dict], output: Path, dpi: int) -> None:
    networks = [row["network"] for row in rows]
    improvements = np.array([row["improvement"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(networks))
    colors = ["#2f6f8f", "#5f8f3f"]

    bars = ax.bar(x, improvements, width=0.52, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(networks)
    ax.set_ylabel("Retained-edge WMAPE reduction (%)")
    ax.set_ylim(0.0, max(40.0, float(improvements.max()) + 6.0))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, row in zip(bars, rows):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            1.2,
            f"vs. {row['best_baseline']}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    caption = (
        "Relative reduction in retained-edge WMAPE achieved by ST-PINN GatedGCN "
        "against the strongest baseline on each network."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

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
            f"{row['network']}: improvement={row['improvement']:.2f}%, "
            f"ST-PINN old WMAPE={row['stpinn_wmape'] * 100:.2f}%, "
            f"{row['best_baseline']} old WMAPE={row['baseline_wmape'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
