from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss and convergence curves from history.json files."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form LABEL=PATH, where PATH points to a run directory or history.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store generated figures and summary JSON.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Convergence Curves",
        help="Figure title prefix.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return safe or "run"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_history_path(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        candidate = path / "history.json"
        if candidate.exists():
            return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(f"Could not resolve history path from {path_text}")


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec '{spec}'. Expected LABEL=PATH.")
    label, path_text = spec.split("=", 1)
    return label.strip(), _resolve_history_path(path_text.strip())


def _get_nested(obj, path, default=None):
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            return default
    return cur


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _extract_split_loss(split_obj) -> float | None:
    if split_obj is None:
        return None
    if isinstance(split_obj, dict):
        if "loss_total" in split_obj:
            return _to_float(split_obj.get("loss_total"))
        if "loss" in split_obj and not isinstance(split_obj.get("loss"), dict):
            return _to_float(split_obj.get("loss"))
    return None


def _extract_split_rmse_norm(split_obj) -> float | None:
    if split_obj is None:
        return None
    if isinstance(split_obj, dict):
        if "rmse_norm" in split_obj:
            return _to_float(split_obj.get("rmse_norm"))
        metrics_obj = split_obj.get("metrics")
        if isinstance(metrics_obj, dict):
            return _to_float(
                _get_nested(metrics_obj, ["normalized", "all_edges", "rmse"])
            )
    return None


def _extract_field(split_obj, key: str) -> float | None:
    if split_obj is None or not isinstance(split_obj, dict):
        return None
    return _to_float(split_obj.get(key))


def _load_history_series(history_path: Path) -> dict:
    payload = _load_json(history_path)
    entries = payload.get("epochs", [])
    series = {
        "epochs": [],
        "epoch_seconds": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_rmse_norm": [],
        "val_rmse_norm": [],
        "test_rmse_norm": [],
        "train_loss_old": [],
        "train_loss_new": [],
        "val_loss_old": [],
        "val_loss_new": [],
        "val_loss_con": [],
        "val_lambda_con": [],
        "val_lambda_new_current": [],
        "val_rho_terminal_abs_mean": [],
        "val_rho_terminal_abs_max": [],
    }

    for entry in entries:
        train_obj = entry.get("train")
        val_obj = entry.get("val")
        test_obj = entry.get("test")

        if val_obj is None and ("val_loss" in entry or "val_metrics" in entry):
            val_obj = {
                "loss_total": _get_nested(entry, ["val_loss", "loss_total"]),
                "metrics": entry.get("val_metrics"),
            }
        if test_obj is None and ("test_loss" in entry or "test_metrics" in entry):
            test_obj = {
                "loss_total": _get_nested(entry, ["test_loss", "loss_total"]),
                "metrics": entry.get("test_metrics"),
            }

        series["epochs"].append(int(entry.get("epoch", len(series["epochs"]))))
        series["epoch_seconds"].append(_to_float(entry.get("epoch_seconds")))
        series["train_loss"].append(_extract_split_loss(train_obj))
        series["val_loss"].append(_extract_split_loss(val_obj))
        series["test_loss"].append(_extract_split_loss(test_obj))
        series["train_rmse_norm"].append(_extract_split_rmse_norm(train_obj))
        series["val_rmse_norm"].append(_extract_split_rmse_norm(val_obj))
        series["test_rmse_norm"].append(_extract_split_rmse_norm(test_obj))
        series["train_loss_old"].append(_extract_field(train_obj, "loss_old"))
        series["train_loss_new"].append(_extract_field(train_obj, "loss_new"))
        series["val_loss_old"].append(_extract_field(val_obj, "loss_old"))
        series["val_loss_new"].append(_extract_field(val_obj, "loss_new"))
        series["val_loss_con"].append(_extract_field(val_obj, "loss_con"))
        series["val_lambda_con"].append(_extract_field(val_obj, "lambda_con"))
        series["val_lambda_new_current"].append(_extract_field(val_obj, "lambda_new_current"))
        series["val_rho_terminal_abs_mean"].append(_extract_field(val_obj, "rho_terminal_abs_mean"))
        series["val_rho_terminal_abs_max"].append(_extract_field(val_obj, "rho_terminal_abs_max"))

    return series


def _valid_xy(xs, ys):
    out_x = []
    out_y = []
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and math.isnan(y)):
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def _plot_overlay(output_dir: Path, runs: list[dict], title: str, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("train_loss", "Train Loss", axes[0]),
        ("val_loss", "Validation Loss", axes[1]),
        ("val_rmse_norm", "Validation Norm RMSE", axes[2]),
    ]
    for key, ylabel, axis in panels:
        for run in runs:
            xs, ys = _valid_xy(run["series"]["epochs"], run["series"][key])
            if xs:
                axis.plot(xs, ys, linewidth=1.8, label=run["label"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / "overlay_core_metrics.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_diagnostics(output_dir: Path, run: dict, dpi: int) -> None:
    label = run["label"]
    safe_label = _safe_label(label)
    series = run["series"]
    epochs = series["epochs"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for key, name in (("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")):
        xs, ys = _valid_xy(epochs, series[key])
        if xs:
            ax.plot(xs, ys, linewidth=1.8, label=name)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    for key, name in (("train_rmse_norm", "train"), ("val_rmse_norm", "val"), ("test_rmse_norm", "test")):
        xs, ys = _valid_xy(epochs, series[key])
        if xs:
            ax.plot(xs, ys, linewidth=1.8, label=name)
    ax.set_title("Normalized RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    component_keys = [
        ("train_loss_old", "train loss_old"),
        ("train_loss_new", "train loss_new"),
        ("val_loss_old", "val loss_old"),
        ("val_loss_new", "val loss_new"),
        ("val_loss_con", "val loss_con"),
    ]
    plotted_any = False
    for key, name in component_keys:
        xs, ys = _valid_xy(epochs, series[key])
        if xs:
            plotted_any = True
            ax.plot(xs, ys, linewidth=1.6, label=name)
    ax.set_title("Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No component history", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    left_plotted = False
    right_plotted = False
    for key, name in (("val_lambda_new_current", "lambda_new_current"), ("val_lambda_con", "lambda_con")):
        xs, ys = _valid_xy(epochs, series[key])
        if xs:
            left_plotted = True
            ax.plot(xs, ys, linewidth=1.8, label=name)
    for key, name in (("val_rho_terminal_abs_mean", "rho_abs_mean"), ("val_rho_terminal_abs_max", "rho_abs_max")):
        xs, ys = _valid_xy(epochs, series[key])
        if xs:
            right_plotted = True
            ax2.plot(xs, ys, linewidth=1.4, linestyle="--", label=name)
    ax.set_title("Lambda / Terminal Rho")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lambda")
    ax2.set_ylabel("Real-space rho residual")
    ax.grid(True, alpha=0.3)
    if left_plotted or right_plotted:
        handles_1, labels_1 = ax.get_legend_handles_labels()
        handles_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No lambda/rho history", ha="center", va="center", transform=ax.transAxes)

    figure.suptitle(f"Diagnostics: {label}")
    figure.tight_layout()
    figure.savefig(output_dir / f"diagnostics_{safe_label}.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _build_summary(runs: list[dict]) -> dict:
    summary = {"runs": []}
    for run in runs:
        epochs = run["series"]["epochs"]
        val_rmse = run["series"]["val_rmse_norm"]
        val_loss = run["series"]["val_loss"]

        best_rmse_epoch = None
        best_rmse_value = None
        best_loss_epoch = None
        best_loss_value = None

        valid_rmse = [(epoch, value) for epoch, value in zip(epochs, val_rmse) if value is not None]
        if valid_rmse:
            best_rmse_epoch, best_rmse_value = min(valid_rmse, key=lambda item: item[1])

        valid_loss = [(epoch, value) for epoch, value in zip(epochs, val_loss) if value is not None]
        if valid_loss:
            best_loss_epoch, best_loss_value = min(valid_loss, key=lambda item: item[1])

        summary["runs"].append(
            {
                "label": run["label"],
                "history_path": str(run["history_path"]),
                "best_val_rmse_norm_epoch": best_rmse_epoch,
                "best_val_rmse_norm": best_rmse_value,
                "best_val_loss_epoch": best_loss_epoch,
                "best_val_loss": best_loss_value,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in args.run:
        label, history_path = _parse_run_spec(spec)
        runs.append(
            {
                "label": label,
                "history_path": history_path,
                "series": _load_history_series(history_path),
            }
        )

    _plot_overlay(output_dir, runs, args.title, args.dpi)
    for run in runs:
        _plot_diagnostics(output_dir, run, args.dpi)

    summary = _build_summary(runs)
    (output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
