#!/usr/bin/env python
"""Combine the three network-level SUE warm-start benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NETWORKS = ("siouxfalls", "ema", "anaheim")
DISPLAY = {"siouxfalls": "SiouxFalls", "ema": "EMA", "anaheim": "Anaheim"}
METHODS = ("cold", "old_flow", "predicted")
LABELS = {"cold": "Cold start", "old_flow": "Old-flow warm start", "predicted": "Predicted-flow warm start"}
COLORS = {"cold": "#6C757D", "old_flow": "#F8961E", "predicted": "#277DA1"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    comparisons = []
    for network_key in NETWORKS:
        path = root / network_key / "benchmark" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark summary: {path}")
        payload = load_json(path)
        methods = payload["methods"]
        for method in METHODS:
            values = methods[method]
            rows.append(
                {
                    "network": DISPLAY[network_key],
                    "method": method,
                    "num_graphs": payload["num_graphs_benchmarked"],
                    "num_paired_converged": payload["num_paired_converged"],
                    "convergence_rate": values["convergence_rate"],
                    "mean_iterations": values["mean_iterations"],
                    "median_iterations": values["median_iterations"],
                    "p95_iterations": values["p95_iterations"],
                    "mean_network_loading_calls": values["mean_network_loading_calls"],
                    "mean_solver_sec": values["mean_solver_sec"],
                    "median_solver_sec": values["median_solver_sec"],
                    "p95_solver_sec": values["p95_solver_sec"],
                    "mean_end_to_end_sec": values["mean_end_to_end_sec"],
                    "solver_speedup_vs_cold": values["solver_speedup_vs_cold"],
                    "end_to_end_speedup_vs_cold": values["end_to_end_speedup_vs_cold"],
                    "iteration_reduction_vs_cold": values["iteration_reduction_vs_cold"],
                    "mean_initial_flow_gap": values["mean_initial_flow_gap"],
                    "mean_initial_wmape_to_dataset": values["mean_initial_wmape_to_dataset"],
                    "mean_final_relative_l2_to_cold": values["mean_final_relative_l2_to_cold"],
                }
            )
        for method in ("old_flow", "predicted"):
            values = methods[method]
            comparisons.append(
                {
                    "network": DISPLAY[network_key],
                    "method": method,
                    "num_paired_converged": payload["num_paired_converged"],
                    "iteration_reduction_percent": 100.0 * values["iteration_reduction_vs_cold"],
                    "solver_speedup": values["solver_speedup_vs_cold"],
                    "end_to_end_speedup": values["end_to_end_speedup_vs_cold"],
                    "convergence_rate": values["convergence_rate"],
                    "final_relative_l2_to_cold": values["mean_final_relative_l2_to_cold"],
                }
            )
    return rows, comparisons


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Network | Initialization | Paired converged | Mean iterations | Reduction (%) | Mean end-to-end (s) | Speedup | Convergence (%) | Final RelL2 to cold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['network']} | {LABELS[row['method']]} | {row['num_paired_converged']} | "
            f"{row['mean_iterations']:.2f} | {100.0 * row['iteration_reduction_vs_cold']:.2f} | "
            f"{row['mean_end_to_end_sec']:.4f} | {row['end_to_end_speedup_vs_cold']:.3f}x | "
            f"{100.0 * row['convergence_rate']:.2f} | {row['mean_final_relative_l2_to_cold']:.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    networks = [DISPLAY[key] for key in NETWORKS]
    x = np.arange(len(networks), dtype=np.float64)
    width = 0.24
    panels = [
        ("mean_iterations", "Mean outer iterations", False),
        ("mean_end_to_end_sec", "Mean end-to-end time (s)", True),
        ("end_to_end_speedup_vs_cold", "End-to-end speedup vs cold", False),
        ("convergence_rate", "Convergence rate", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    for axis, (field, ylabel, log_scale) in zip(axes.flat, panels):
        for method_index, method in enumerate(METHODS):
            selected = [next(row for row in rows if row["network"] == network and row["method"] == method) for network in networks]
            values = [row[field] for row in selected]
            axis.bar(
                x + (method_index - 1) * width,
                values,
                width,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(networks)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
        if field == "end_to_end_speedup_vs_cold":
            axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
        if field == "convergence_rate":
            axis.set_ylim(0.0, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/sue_warmstart", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    output = root / "summary"
    rows, comparisons = collect(root)
    write_csv(output / "sue_warmstart_all_methods.csv", rows)
    write_csv(output / "sue_warmstart_comparison.csv", comparisons)
    write_markdown(output / "sue_warmstart_table.md", rows)
    plot(output / "sue_warmstart_comparison.png", rows)
    print(f"Summary outputs: {output}")


if __name__ == "__main__":
    main()
