#!/usr/bin/env python3
"""Summarize ST-PINN GatedGCN training time from final run summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "Sioux Falls": Path(
        "results/ours/network-pairs-topology-stpinn_gatedgcn_siouxfalls_10000/0/summary.json"
    ),
    "EMA": Path(
        "results/ours/network-pairs-topology-stpinn_gatedgcn_ema_10000/0/summary.json"
    ),
}


def format_duration(seconds: float) -> str:
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    if hours >= 1.0:
        return f"{hours:.2f} h"
    return f"{minutes:.1f} min"


def load_row(project_root: Path, network: str, rel_path: Path) -> dict:
    summary_path = project_root / rel_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    training_time = summary.get("training_time", {})
    total_seconds = float(training_time["total_seconds"])
    num_epochs = int(training_time["num_epochs"])
    avg_epoch_seconds = float(training_time["average_epoch_seconds"])
    metadata = summary.get("dataset_metadata", {})
    splits = metadata.get("splits", {})

    return {
        "network": network,
        "model": "ST-PINN GatedGCN",
        "params": int(summary.get("params", 0)),
        "device": summary.get("device_used", ""),
        "num_epochs": num_epochs,
        "best_epoch": int(summary.get("best_epoch", -1)),
        "train_graphs": int(splits.get("train", 0)),
        "val_graphs": int(splits.get("val", 0)),
        "test_graphs": int(splits.get("test", 0)),
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60.0,
        "total_hours": total_seconds / 3600.0,
        "average_epoch_seconds": avg_epoch_seconds,
        "formatted_time": format_duration(total_seconds),
        "summary_path": str(rel_path),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "network",
        "model",
        "params",
        "device",
        "num_epochs",
        "best_epoch",
        "train_graphs",
        "val_graphs",
        "test_graphs",
        "total_seconds",
        "total_minutes",
        "total_hours",
        "average_epoch_seconds",
        "formatted_time",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training time of the ST-PINN GatedGCN model.}",
        r"\label{tab:stpinn-training-time}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Network & Epochs & Training time & Time per epoch & Parameters \\",
        r"\midrule",
    ]
    for row in rows:
        params_k = row["params"] / 1000.0
        lines.append(
            f"{row['network']} & "
            f"{row['num_epochs']} & "
            f"{row['formatted_time']} & "
            f"{row['average_epoch_seconds']:.2f} s & "
            f"{params_k:.1f}k \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ST-PINN GatedGCN training time for Sioux Falls and EMA."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing results/ours.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <project-root>/results/training_time.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir or (project_root / "results" / "training_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_row(project_root, network, rel_path) for network, rel_path in RUNS.items()]

    json_path = output_dir / "stpinn_training_time_summary.json"
    csv_path = output_dir / "stpinn_training_time_summary.csv"
    tex_path = output_dir / "stpinn_training_time_table.tex"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)

    print("ST-PINN GatedGCN training time summary")
    for row in rows:
        print(
            f"- {row['network']}: {row['formatted_time']} "
            f"({row['total_seconds']:.1f} s), "
            f"{row['average_epoch_seconds']:.2f} s/epoch, "
            f"{row['num_epochs']} epochs"
        )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
