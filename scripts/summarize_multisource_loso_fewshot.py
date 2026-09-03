#!/usr/bin/env python
"""Summarize LOSO few-shot results and draw the four primary metric curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NETWORK_ORDER = ["SiouxFalls", "EMA", "Anaheim"]
COLORS = {
    "SiouxFalls": "#277DA1",
    "EMA": "#43AA8B",
    "Anaheim": "#F8961E",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(results_root: Path) -> list[dict]:
    rows = []
    found_targets = set()
    for path in sorted(results_root.glob("target_*/summary.json")):
        payload = load_json(path)
        target = str(payload["target"])
        found_targets.add(target)
        sources = "+".join(payload["sources"])
        for key, result in payload["k_results"].items():
            test = result["test"]
            metrics = test["metrics_real"]
            constraint = test["constraint_violation"]
            rows.append(
                {
                    "target": target,
                    "sources": sources,
                    "k": int(key),
                    "all_edge_wmape": float(metrics["all_edges"]["wmape"]),
                    "new_edge_wmape": float(metrics["new_edges"]["wmape"]),
                    "r2": float(metrics["all_edges"]["r2"]),
                    "relcon": float(constraint["relcon"]),
                    "relcon_p95": float(constraint["relcon_p95"]),
                    "milliseconds_per_graph": float(test["timing"]["forward_milliseconds_per_graph"]),
                    "source": str(path),
                }
            )
    missing = [network for network in NETWORK_ORDER if network not in found_targets]
    if missing:
        raise FileNotFoundError(
            "Cannot summarize an incomplete LOSO sweep. Missing target summaries: " + ", ".join(missing)
        )
    rows.sort(key=lambda row: (NETWORK_ORDER.index(row["target"]), row["k"]))
    return rows


def validate_k_values(rows: list[dict], expected: list[int]) -> None:
    for network in NETWORK_ORDER:
        available = sorted(row["k"] for row in rows if row["target"] == network)
        if available != expected:
            raise RuntimeError(f"{network}: expected k={expected}, found k={available}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict], k_values: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("all_edge_wmape", "All-edge WMAPE", False),
        ("new_edge_wmape", "New-edge WMAPE", False),
        ("r2", r"$R^2$", True),
        ("relcon", "RelCon", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (field, ylabel, higher_is_better) in zip(axes.flat, panels):
        for network in NETWORK_ORDER:
            selected = sorted(
                (row for row in rows if row["target"] == network),
                key=lambda row: row["k"],
            )
            axis.plot(
                [row["k"] for row in selected],
                [row[field] for row in selected],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=network,
                color=COLORS[network],
            )
        axis.set_xscale("symlog", linthresh=50, linscale=1.0, base=10)
        axis.set_xticks(k_values)
        axis.set_xticklabels([str(value) for value in k_values], rotation=30)
        axis.set_xlabel("Number of labeled target graphs (k)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if higher_is_better:
            axis.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.4)
    axes[0, 0].legend(frameon=False)
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Target | k | All-edge WMAPE (%) | New-edge WMAPE (%) | R2 | RelCon |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['k']} | "
            f"{100.0 * row['all_edge_wmape']:.2f} | "
            f"{100.0 * row['new_edge_wmape']:.2f} | "
            f"{row['r2']:.4f} | {row['relcon']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/multisource_loso_fewshot")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[0, 50, 100, 250, 500, 1000, 2000, 4000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).expanduser().resolve()
    output_dir = root / "summary"
    rows = collect_rows(root)
    expected = sorted(set(args.k_values))
    validate_k_values(rows, expected)
    csv_path = output_dir / "multisource_loso_fewshot_results.csv"
    figure_path = output_dir / "fewshot_adaptation_curves.png"
    markdown_path = output_dir / "multisource_loso_fewshot_table.md"
    write_csv(csv_path, rows)
    plot_curves(figure_path, rows, expected)
    write_markdown(markdown_path, rows)
    print(f"CSV      : {csv_path}")
    print(f"Figure   : {figure_path}")
    print(f"PDF      : {figure_path.with_suffix('.pdf')}")
    print(f"Markdown : {markdown_path}")


if __name__ == "__main__":
    main()
