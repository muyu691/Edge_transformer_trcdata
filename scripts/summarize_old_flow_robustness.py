#!/usr/bin/env python
"""Merge clean and corrupted old-flow evaluations into CSV and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NETWORKS = {
    "SiouxFalls": "siouxfalls",
    "EMA": "ema",
    "Anaheim": "anaheim",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(network: str, mode: str, level: float, test: dict, source: Path) -> dict:
    real = test["metrics"]["real"]
    constraint = test["constraint_violation"]
    timing = test.get("timing", {})
    return {
        "network": network,
        "perturbation": mode,
        "level": float(level),
        "level_percent": float(level) * 100.0,
        "wmape": float(real["all_edges"]["wmape"]),
        "rmse": float(real["all_edges"]["rmse"]),
        "r2": float(real["all_edges"]["r2"]),
        "old_edge_wmape": float(real["old_edges"]["wmape"]),
        "new_edge_wmape": float(real["new_edges"]["wmape"]),
        "relcon": float(constraint["relcon"]),
        "relcon_p95": float(constraint["relcon_p95"]),
        "milliseconds_per_graph": float(timing.get("forward_milliseconds_per_graph", 0.0)),
        "source": str(source),
    }


def clean_rows(ours_root: Path) -> list[dict]:
    rows = []
    for network, key in NETWORKS.items():
        summary_path = (
            ours_root
            / f"network-pairs-topology-ours_edge_transformer_{key}_10000"
            / "0"
            / "summary.json"
        )
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing clean our-model summary: {summary_path}")
        payload = load_json(summary_path)
        clean_test = {
            "metrics": payload["test_metrics"],
            "constraint_violation": payload["constraint_violation"],
            "timing": payload.get("test_time", {}),
        }
        # The same completed clean evaluation is the zero point of both curves.
        rows.append(metric_row(network, "missing", 0.0, clean_test, summary_path))
        rows.append(metric_row(network, "gaussian_noise", 0.0, clean_test, summary_path))
    return rows


def robustness_rows(results_root: Path) -> list[dict]:
    rows = []
    paths = sorted(results_root.rglob("old_flow_robustness_*.json"))
    if not paths:
        raise FileNotFoundError(f"No robustness JSON files found below {results_root}")
    for path in paths:
        payload = load_json(path)
        perturbation = payload["perturbation"]
        rows.append(
            metric_row(
                network=str(payload["network"]),
                mode=str(perturbation["mode"]),
                level=float(perturbation["requested_level"]),
                test=payload["test"],
                source=path,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_expected_conditions(rows: list[dict]) -> None:
    expected = {
        "missing": {0.0, 0.15, 0.30},
        "gaussian_noise": {0.0, 0.05, 0.15},
    }
    missing_conditions = []
    for network in NETWORKS:
        for mode, levels in expected.items():
            available = {
                round(float(row["level"]), 8)
                for row in rows
                if row["network"] == network and row["perturbation"] == mode
            }
            for level in levels:
                if round(level, 8) not in available:
                    missing_conditions.append(f"{network}:{mode}:{level}")
    if missing_conditions:
        raise RuntimeError(
            "Cannot summarize an incomplete robustness sweep. Missing: "
            + ", ".join(missing_conditions)
        )


def plot_results(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    modes = ["missing", "gaussian_noise"]
    xlabels = ["Missing old-flow observations (%)", "Gaussian noise sigma (%)"]
    colors = {"SiouxFalls": "#277DA1", "EMA": "#43AA8B", "Anaheim": "#F8961E"}

    for column, (mode, xlabel) in enumerate(zip(modes, xlabels)):
        for network in NETWORKS:
            selected = sorted(
                (row for row in rows if row["network"] == network and row["perturbation"] == mode),
                key=lambda row: row["level"],
            )
            if not selected:
                continue
            x = [row["level_percent"] for row in selected]
            axes[0, column].plot(
                x,
                [row["wmape"] for row in selected],
                marker="o",
                linewidth=1.8,
                label=network,
                color=colors[network],
            )
            axes[1, column].plot(
                x,
                [row["relcon"] for row in selected],
                marker="o",
                linewidth=1.8,
                label=network,
                color=colors[network],
            )

        axes[0, column].set_xlabel(xlabel)
        axes[0, column].set_ylabel("WMAPE")
        axes[1, column].set_xlabel(xlabel)
        axes[1, column].set_ylabel("RelCon")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[0, column].legend(frameon=False)

    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/old_flow_robustness")
    parser.add_argument("--ours-root", default="results/ours")
    parser.add_argument("--output-dir", default="results/old_flow_robustness/summary")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    ours_root = Path(args.ours_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = clean_rows(ours_root) + robustness_rows(results_root)
    rows.sort(key=lambda row: (row["network"], row["perturbation"], row["level"]))
    validate_expected_conditions(rows)

    csv_path = output_dir / "old_flow_robustness_results.csv"
    figure_path = output_dir / "old_flow_robustness_curves.png"
    write_csv(csv_path, rows)
    print(f"CSV    : {csv_path}")
    if not args.no_plot:
        plot_results(figure_path, rows)
        print(f"Figure : {figure_path}")
        print(f"PDF    : {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
