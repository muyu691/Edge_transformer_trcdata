"""Plot accuracy-conservation trade-off figures from completed sweep runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


LAMBDA_ORDER = [0.0, 0.01, 0.05, 0.1, 0.2]
NETWORK_ORDER = ["SiouxFalls", "EMA"]
NETWORK_LABELS = {
    "SiouxFalls": "Sioux Falls",
    "EMA": "EMA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis figures for the accuracy-conservation sweep."
    )
    parser.add_argument(
        "--results-dir",
        default="results/accuracy_conservation_tradeoff",
        help="Directory containing the completed trade-off runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/accuracy_conservation_tradeoff/figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def to_float(value) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def read_csv_rows(results_dir: Path) -> list[dict]:
    csv_path = results_dir / "accuracy_conservation_tradeoff.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV summary: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    for row in rows:
        row["lambda_con"] = to_float(row["lambda_con"])
        row["rmse"] = to_float(row["rmse"])
        row["wmape"] = to_float(row["wmape"])
        row["relcon"] = to_float(row["relcon"])
        row["relcon_p95"] = to_float(row.get("relcon_p95", ""))
    return rows


def read_history(run_dir: Path) -> list[dict]:
    path = run_dir / "0" / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("epochs", [])


def read_summary(run_dir: Path) -> dict:
    path = run_dir / "0" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_runs(results_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if not (run_dir / "0" / "summary.json").exists():
            continue
        summary = read_summary(run_dir)
        cfg = summary.get("config", {})
        model_cfg = cfg.get("model", {})
        network = summary.get("network_name", "")
        lambda_con = float(model_cfg.get("lambda_con"))
        runs.append(
            {
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "network": network,
                "lambda_con": lambda_con,
                "history": read_history(run_dir),
            }
        )
    return runs


def sorted_network_rows(rows: list[dict], network: str) -> list[dict]:
    subset = [row for row in rows if row["network"] == network]
    return sorted(subset, key=lambda row: row["lambda_con"])


def lambda_label(value: float) -> str:
    return "0" if abs(value) < 1e-12 else f"{value:g}"


def save_figure(fig, output_prefix: Path, dpi: int) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def plot_tradeoff(rows: list[dict], output_dir: Path, dpi: int, metric: str) -> None:
    ylabel = "WMAPE (%)" if metric == "wmape" else "RMSE"
    yscale = 100.0 if metric == "wmape" else 1.0
    cmap = plt.get_cmap("viridis")
    color_values = {value: cmap(idx / (len(LAMBDA_ORDER) - 1)) for idx, value in enumerate(LAMBDA_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=False)
    for ax, network in zip(axes, NETWORK_ORDER):
        subset = sorted_network_rows(rows, network)
        x = np.array([row["relcon"] for row in subset], dtype=float)
        y = np.array([row[metric] * yscale for row in subset], dtype=float)

        ax.plot(x, y, color="#4b5563", linewidth=1.0, alpha=0.55, zorder=1)
        for row in subset:
            lam = row["lambda_con"]
            ax.scatter(
                row["relcon"],
                row[metric] * yscale,
                s=72,
                color=color_values.get(lam, "#2f9e44"),
                edgecolor="#222222",
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                rf"$\lambda_{{con}}={lambda_label(lam)}$",
                (row["relcon"], row[metric] * yscale),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        best_idx = int(np.nanargmin(y))
        ax.scatter(
            x[best_idx],
            y[best_idx],
            s=150,
            facecolor="none",
            edgecolor="#d9480f",
            linewidth=1.6,
            zorder=4,
        )
        ax.set_title(NETWORK_LABELS.get(network, network))
        ax.set_xlabel("RelCon")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    save_figure(fig, output_dir / f"accuracy_conservation_tradeoff_{metric}", dpi)


def metric_from_epoch(epoch_record: dict, split: str, metric: str) -> float | None:
    split_payload = epoch_record.get(split)
    if not isinstance(split_payload, dict):
        return None
    keys = [f"{metric}_real", metric]
    for key in keys:
        value = split_payload.get(key)
        if value is not None:
            return float(value)
    return None


def plot_epoch_curves(runs: list[dict], output_dir: Path, dpi: int, metric: str, split: str = "val") -> None:
    ylabel = "Validation WMAPE (%)" if metric == "wmape" else "Validation RMSE"
    yscale = 100.0 if metric == "wmape" else 1.0
    cmap = plt.get_cmap("viridis")
    color_values = {value: cmap(idx / (len(LAMBDA_ORDER) - 1)) for idx, value in enumerate(LAMBDA_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=False)
    for ax, network in zip(axes, NETWORK_ORDER):
        subset = sorted(
            [run for run in runs if run["network"] == network],
            key=lambda run: run["lambda_con"],
        )
        for run in subset:
            epochs = []
            values = []
            for item in run["history"]:
                value = metric_from_epoch(item, split=split, metric=metric)
                if value is None:
                    continue
                epochs.append(int(item.get("epoch", len(epochs))))
                values.append(value * yscale)
            if not epochs:
                continue
            lam = run["lambda_con"]
            ax.plot(
                epochs,
                values,
                linewidth=1.5,
                color=color_values.get(lam, "#2f9e44"),
                label=rf"$\lambda_{{con}}={lambda_label(lam)}$",
            )

        ax.set_title(NETWORK_LABELS.get(network, network))
        ax.set_xlabel("Epoch")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir / f"accuracy_conservation_epoch_{split}_{metric}", dpi)


def main() -> None:
    args = parse_args()
    root = project_root()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    rows = read_csv_rows(results_dir)
    runs = collect_runs(results_dir)

    plot_tradeoff(rows, output_dir, args.dpi, metric="wmape")
    plot_tradeoff(rows, output_dir, args.dpi, metric="rmse")
    plot_epoch_curves(runs, output_dir, args.dpi, metric="wmape", split="val")
    plot_epoch_curves(runs, output_dir, args.dpi, metric="rmse", split="val")


if __name__ == "__main__":
    main()
