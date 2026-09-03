#!/usr/bin/env python
"""Compare multi-source-pretrained and random-initialized few-shot curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def metrics_from_result(result: dict) -> dict:
    test = result["test"]
    metrics = test["metrics_real"]
    constraint = test["constraint_violation"]
    return {
        "all_edge_wmape": float(metrics["all_edges"]["wmape"]),
        "new_edge_wmape": float(metrics["new_edges"]["wmape"]),
        "r2": float(metrics["all_edges"]["r2"]),
        "relcon": float(constraint["relcon"]),
    }


def collect_rows(random_root: Path, transfer_root: Path, expected_k: list[int]) -> tuple[list[dict], list[dict]]:
    long_rows = []
    comparison_rows = []
    for target in NETWORK_ORDER:
        random_path = random_root / f"target_{target.lower()}" / "summary.json"
        transfer_path = transfer_root / f"target_{target.lower()}" / "summary.json"
        if not random_path.exists() or not transfer_path.exists():
            raise FileNotFoundError(
                f"Missing paired summary for {target}: random={random_path.exists()}, "
                f"transfer={transfer_path.exists()}"
            )
        random_summary = load_json(random_path)
        transfer_summary = load_json(transfer_path)
        nested_path = transfer_path.parent / "fewshot_nested_indices.npz"
        if not nested_path.exists():
            raise FileNotFoundError(f"Missing transfer nested subsets: {nested_path}")
        nested_order = np.load(nested_path)["order"].astype(np.int64)
        random_k = sorted(int(value) for value in random_summary["k_values"])
        transfer_k = sorted(int(value) for value in transfer_summary["k_values"])
        if random_k != expected_k or transfer_k != expected_k:
            raise RuntimeError(
                f"{target}: expected k={expected_k}, random={random_k}, transfer={transfer_k}"
            )
        if random_summary["transfer_protocol_signature"] != transfer_summary["protocol_signature"]:
            raise RuntimeError(f"{target}: random result points to a different transfer protocol.")

        for k in expected_k:
            random_result = random_summary["k_results"][str(k)]
            transfer_result = transfer_summary["k_results"][str(k)]
            random_metrics = metrics_from_result(random_result)
            transfer_metrics = metrics_from_result(transfer_result)
            expected_subset_hash = hashlib.sha256(nested_order[:k].tobytes()).hexdigest()
            if random_result["subset_sha256"] != expected_subset_hash:
                raise RuntimeError(f"{target}, k={k}: target subsets differ.")
            random_steps = int(random_result["expected_transfer_optimizer_steps"])
            transfer_adaptation = transfer_result.get("adaptation")
            transfer_steps = 0 if transfer_adaptation is None else int(transfer_adaptation["optimizer_steps"])
            actual_adaptation = random_result.get("adaptation")
            actual_steps = 0 if actual_adaptation is None else int(actual_adaptation["optimizer_steps"])
            if random_steps != transfer_steps or actual_steps != transfer_steps:
                raise RuntimeError(
                    f"{target}, k={k}: optimizer steps differ: "
                    f"expected={random_steps}, random={actual_steps}, transfer={transfer_steps}"
                )

            for initialization, values in (
                ("Multi-source pretraining", transfer_metrics),
                ("Random initialization", random_metrics),
            ):
                long_rows.append(
                    {
                        "target": target,
                        "k": k,
                        "initialization": initialization,
                        "optimizer_steps": transfer_steps,
                        **values,
                    }
                )

            random_all = random_metrics["all_edge_wmape"]
            random_new = random_metrics["new_edge_wmape"]
            comparison_rows.append(
                {
                    "target": target,
                    "k": k,
                    "optimizer_steps": transfer_steps,
                    "transfer_all_edge_wmape": transfer_metrics["all_edge_wmape"],
                    "random_all_edge_wmape": random_all,
                    "all_edge_wmape_gain": random_all - transfer_metrics["all_edge_wmape"],
                    "all_edge_relative_reduction": (
                        1.0 - transfer_metrics["all_edge_wmape"] / random_all
                        if random_all > 0.0
                        else float("nan")
                    ),
                    "transfer_new_edge_wmape": transfer_metrics["new_edge_wmape"],
                    "random_new_edge_wmape": random_new,
                    "new_edge_wmape_gain": random_new - transfer_metrics["new_edge_wmape"],
                    "new_edge_relative_reduction": (
                        1.0 - transfer_metrics["new_edge_wmape"] / random_new
                        if random_new > 0.0
                        else float("nan")
                    ),
                    "transfer_r2": transfer_metrics["r2"],
                    "random_r2": random_metrics["r2"],
                    "r2_gain": transfer_metrics["r2"] - random_metrics["r2"],
                    "transfer_relcon": transfer_metrics["relcon"],
                    "random_relcon": random_metrics["relcon"],
                    "relcon_gain": random_metrics["relcon"] - transfer_metrics["relcon"],
                }
            )
    return long_rows, comparison_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "Positive WMAPE/RelCon gains and positive R2 gain mean multi-source pretraining is better.",
        "",
        "| Target | k | Steps | Transfer WMAPE (%) | Random WMAPE (%) | Relative reduction (%) | R2 gain | RelCon gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['k']} | {row['optimizer_steps']} | "
            f"{100.0 * row['transfer_all_edge_wmape']:.2f} | "
            f"{100.0 * row['random_all_edge_wmape']:.2f} | "
            f"{100.0 * row['all_edge_relative_reduction']:.2f} | "
            f"{row['r2_gain']:.4f} | {row['relcon_gain']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(path: Path, rows: list[dict], k_values: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("all_edge_wmape", "All-edge WMAPE"),
        ("new_edge_wmape", "New-edge WMAPE"),
        ("r2", r"$R^2$"),
        ("relcon", "RelCon"),
    ]
    styles = {
        "Multi-source pretraining": "-",
        "Random initialization": "--",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for axis, (field, ylabel) in zip(axes.flat, panels):
        for target in NETWORK_ORDER:
            for initialization, linestyle in styles.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["target"] == target and row["initialization"] == initialization
                    ),
                    key=lambda row: row["k"],
                )
                axis.plot(
                    [row["k"] for row in selected],
                    [row[field] for row in selected],
                    color=COLORS[target],
                    linestyle=linestyle,
                    marker="o" if initialization == "Multi-source pretraining" else "s",
                    linewidth=1.8,
                    markersize=4.2,
                    label=f"{target}: {initialization}",
                )
        axis.set_xscale("symlog", linthresh=50, linscale=1.0, base=10)
        axis.set_xticks(k_values)
        axis.set_xticklabels([str(value) for value in k_values], rotation=30)
        axis.set_xlabel("Number of labeled target graphs (k)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=False,
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-root", default="results/random_init_fewshot")
    parser.add_argument("--transfer-root", default="results/multisource_loso_fewshot")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[0, 50, 100, 250, 500, 1000, 2000, 4000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_root = Path(args.random_root).expanduser().resolve()
    transfer_root = Path(args.transfer_root).expanduser().resolve()
    output_dir = random_root / "summary"
    expected_k = sorted(set(int(value) for value in args.k_values))
    rows, comparison_rows = collect_rows(random_root, transfer_root, expected_k)
    long_csv = output_dir / "random_and_transfer_fewshot_results.csv"
    comparison_csv = output_dir / "transfer_vs_random_comparison.csv"
    markdown_path = output_dir / "transfer_vs_random_table.md"
    figure_path = output_dir / "transfer_vs_random_fewshot_curves.png"
    write_csv(long_csv, rows)
    write_csv(comparison_csv, comparison_rows)
    write_markdown(markdown_path, comparison_rows)
    plot_curves(figure_path, rows, expected_k)
    print(f"Results    : {long_csv}")
    print(f"Comparison : {comparison_csv}")
    print(f"Markdown   : {markdown_path}")
    print(f"Figure     : {figure_path}")
    print(f"PDF        : {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
