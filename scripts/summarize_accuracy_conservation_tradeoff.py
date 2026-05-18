"""Collect accuracy-conservation sweep summaries into one CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ST-PINN GatedGCN accuracy-conservation sweep results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/accuracy_conservation_tradeoff",
        help="Directory containing GraphGym run folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/accuracy_conservation_tradeoff/accuracy_conservation_tradeoff.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_name(path: Path) -> str:
    try:
        return path.parents[1].name
    except IndexError:
        return path.parent.name


def main() -> None:
    args = parse_args()
    root = Path(args.results_dir)
    summaries = sorted(root.glob("*/0/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json files found under {root}")

    rows = []
    for summary_path in summaries:
        item = read_summary(summary_path)
        cfg = item.get("config", {})
        model_cfg = cfg.get("model", {})
        test_metrics = item.get("test_metrics", {})
        real = test_metrics.get("real", {})
        all_edges = real.get("all_edges", {})
        old_edges = real.get("old_edges", {})
        new_edges = real.get("new_edges", {})
        con = item.get("constraint_violation", {})
        timing = item.get("test_time", {})

        rows.append(
            {
                "run": run_name(summary_path),
                "network": item.get("network_name", ""),
                "lambda_new_final": model_cfg.get("lambda_new_final", ""),
                "lambda_con": model_cfg.get("lambda_con", ""),
                "lambda_con_schedule": model_cfg.get("lambda_con_schedule", ""),
                "best_epoch": item.get("best_epoch", ""),
                "r2": all_edges.get("r2", ""),
                "rmse": all_edges.get("rmse", ""),
                "wmape": all_edges.get("wmape", ""),
                "old_wmape": old_edges.get("wmape", ""),
                "new_wmape": new_edges.get("wmape", ""),
                "relcon": con.get("relcon", ""),
                "relcon_p95": con.get("relcon_p95", ""),
                "time_ms_per_graph": timing.get("forward_milliseconds_per_graph", ""),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "network",
        "lambda_new_final",
        "lambda_con",
        "lambda_con_schedule",
        "best_epoch",
        "r2",
        "rmse",
        "wmape",
        "old_wmape",
        "new_wmape",
        "relcon",
        "relcon_p95",
        "time_ms_per_graph",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
