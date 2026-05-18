"""Plot ablation impact on WMAPE and RelCon relative to the full ST-PINN GatedGCN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = [
    ("wo_old_flow", "w/o old\nflow"),
    ("new_attr_only", "new attr\nonly"),
    ("no_rho_injection", "w/o residual\ninjection"),
    ("no_global_message", "w/o global\nmessage"),
    ("unshared_diffusion_cells", "unshared\ncells"),
    ("no_lwr_recurrence", "w/o recurrent\npressure"),
]

NETWORK_ORDER = ["SiouxFalls", "EMA"]
NETWORK_LABELS = {"SiouxFalls": "Sioux Falls", "EMA": "EMA"}
NETWORK_COLORS = {"SiouxFalls": "#2a9d8f", "EMA": "#e9c46a"}


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ablation impact bar charts.")
    parser.add_argument(
        "--ablation-dir",
        default="results/ablation",
        help="Directory containing ablation run folders.",
    )
    parser.add_argument(
        "--ours-dir",
        default="results/ours",
        help="Directory containing full ST-PINN GatedGCN runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablation/figures",
        help="Directory for generated figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_from_summary(summary: dict) -> dict[str, float | str | int]:
    real_all = summary["test_metrics"]["real"]["all_edges"]
    return {
        "network": summary["network_name"],
        "wmape": float(real_all["wmape"]),
        "relcon": float(summary["constraint_violation"]["relcon"]),
        "rmse": float(real_all["rmse"]),
        "params": int(summary.get("params", 0)),
    }


def find_full_model_summaries(ours_dir: Path) -> dict[str, dict]:
    full = {}
    for summary_path in ours_dir.glob("network-pairs-topology-stpinn_gatedgcn_*_10000/0/summary.json"):
        summary = read_json(summary_path)
        item = metrics_from_summary(summary)
        full[str(item["network"])] = item
    missing = [network for network in NETWORK_ORDER if network not in full]
    if missing:
        raise FileNotFoundError(f"Missing full model summaries for: {', '.join(missing)}")
    return full


def variant_key_from_name(name: str) -> str | None:
    for key, _ in VARIANT_ORDER:
        if key in name:
            return key
    return None


def collect_ablation_metrics(ablation_dir: Path, full: dict[str, dict]) -> list[dict]:
    rows = []
    for summary_path in sorted(ablation_dir.glob("*/0/summary.json")):
        run_name = summary_path.parents[1].name
        variant_key = variant_key_from_name(run_name)
        if variant_key is None:
            continue
        summary = read_json(summary_path)
        item = metrics_from_summary(summary)
        network = str(item["network"])
        base = full[network]
        rows.append(
            {
                "run": run_name,
                "network": network,
                "variant_key": variant_key,
                "wmape": item["wmape"],
                "relcon": item["relcon"],
                "delta_wmape_pp": (float(item["wmape"]) - float(base["wmape"])) * 100.0,
                "delta_relcon": float(item["relcon"]) - float(base["relcon"]),
            }
        )

    expected = {(network, key) for network in NETWORK_ORDER for key, _ in VARIANT_ORDER}
    observed = {(row["network"], row["variant_key"]) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise FileNotFoundError(f"Missing ablation summaries for: {missing}")
    return rows


def save_delta_csv(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ablation_impact_delta.csv"
    fields = [
        "network",
        "variant_key",
        "wmape",
        "relcon",
        "delta_wmape_pp",
        "delta_relcon",
        "run",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def value_lookup(rows: list[dict], network: str, variant_key: str, metric: str) -> float:
    for row in rows:
        if row["network"] == network and row["variant_key"] == variant_key:
            return float(row[metric])
    raise KeyError((network, variant_key, metric))


def set_symmetric_ylim(ax, values: list[float], pad: float = 1.15) -> None:
    max_abs = max(abs(value) for value in values)
    if max_abs <= 1e-12:
        max_abs = 1.0
    ax.set_ylim(-max_abs * pad, max_abs * pad)


def plot_ablation_impact(rows: list[dict], output_dir: Path, dpi: int) -> None:
    variant_keys = [key for key, _ in VARIANT_ORDER]
    labels = [label for _, label in VARIANT_ORDER]
    x = np.arange(len(variant_keys), dtype=float)
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)

    metrics = [
        ("delta_wmape_pp", "WMAPE change over full model\n(percentage points)"),
        ("delta_relcon", "RelCon change over full model"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        all_values = []
        for idx, network in enumerate(NETWORK_ORDER):
            offset = (idx - 0.5) * width
            values = [value_lookup(rows, network, key, metric) for key in variant_keys]
            all_values.extend(values)
            ax.bar(
                x + offset,
                values,
                width=width,
                label=NETWORK_LABELS[network],
                color=NETWORK_COLORS[network],
                edgecolor="#222222",
                linewidth=0.55,
                alpha=0.95,
            )

        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        set_symmetric_ylim(ax, all_values)

    axes[0].set_title("Prediction accuracy")
    axes[1].set_title("Physical consistency")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "ablation_impact_wmape_relcon.png"
    pdf_path = output_dir / "ablation_impact_wmape_relcon.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    args = parse_args()
    root = project_root()
    ablation_dir = Path(args.ablation_dir)
    ours_dir = Path(args.ours_dir)
    output_dir = Path(args.output_dir)
    if not ablation_dir.is_absolute():
        ablation_dir = root / ablation_dir
    if not ours_dir.is_absolute():
        ours_dir = root / ours_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    full = find_full_model_summaries(ours_dir)
    rows = collect_ablation_metrics(ablation_dir, full)
    save_delta_csv(rows, output_dir)
    plot_ablation_impact(rows, output_dir, args.dpi)


if __name__ == "__main__":
    main()
