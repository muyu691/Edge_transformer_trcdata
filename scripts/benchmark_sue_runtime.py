#!/usr/bin/env python3
"""Benchmark SUE solve time on the test split of network-pair datasets.

The measured time is the wall-clock time of solving SUE on each reconfigured
graph G'. Dataset loading, split construction, and file writing are excluded
from the per-graph timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREATE_DATA_DIR = PROJECT_ROOT / "create_sioux_data"
sys.path.insert(0, str(CREATE_DATA_DIR))

from solve_network_pairs import solve_single_graph_sue  # noqa: E402


DATASET_PKLS = {
    "siouxfalls": PROJECT_ROOT
    / "create_sioux_data"
    / "processed_data"
    / "siouxfalls_pairs_newpolicy_lhs"
    / "network_pairs_dataset.pkl",
    "ema": PROJECT_ROOT
    / "create_sioux_data"
    / "processed_data"
    / "ema_pairs_newpolicy_lhs"
    / "network_pairs_dataset.pkl",
    "anaheim": PROJECT_ROOT
    / "create_sioux_data"
    / "processed_data"
    / "anaheim_pairs_newpolicy_lhs"
    / "network_pairs_dataset.pkl",
}

DISPLAY_NAMES = {
    "siouxfalls": "SiouxFalls",
    "ema": "EMA",
    "anaheim": "Anaheim",
}


def split_indices(
    num_samples: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the train/validation/test split used by build_network_pairs_dataset.py."""
    if num_samples <= 0:
        raise ValueError("No valid samples found in the input pickle.")

    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(num_samples)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    if n_train == 0:
        n_train = 1
    if val_ratio > 0 and num_samples - n_train >= 2 and n_val == 0:
        n_val = 1
    if n_train + n_val > num_samples:
        n_val = max(0, num_samples - n_train)

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train : n_train + n_val]
    test_idx = all_idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def load_pairs(input_pkl: Path) -> list[dict[str, Any]]:
    with input_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if "pairs" in payload:
            pairs = payload.get("pairs")
        elif "completed_pairs" in payload:
            pairs = payload.get("completed_pairs")
        else:
            pairs = None
    else:
        pairs = payload
    if not isinstance(pairs, list):
        raise ValueError(
            f"Could not read pairs from {input_pkl}. Expected a list, "
            "a {'pairs': ...} dataset, or a {'completed_pairs': ...} checkpoint."
        )
    return pairs


def summarize_success(records: list[dict[str, Any]]) -> dict[str, float]:
    success_times = np.array(
        [float(row["elapsed_sec"]) for row in records if row["status"] == "ok"],
        dtype=np.float64,
    )
    if success_times.size == 0:
        raise RuntimeError("No successful SUE solves were recorded.")

    return {
        "total_sue_time_sec": float(success_times.sum()),
        "mean_sue_time_sec": float(success_times.mean()),
        "median_sue_time_sec": float(np.median(success_times)),
        "std_sue_time_sec": float(success_times.std()),
        "min_sue_time_sec": float(success_times.min()),
        "max_sue_time_sec": float(success_times.max()),
        "p25_sue_time_sec": float(np.percentile(success_times, 25)),
        "p75_sue_time_sec": float(np.percentile(success_times, 75)),
        "p95_sue_time_sec": float(np.percentile(success_times, 95)),
        "mean_sue_time_ms": float(success_times.mean() * 1000.0),
        "median_sue_time_ms": float(np.median(success_times) * 1000.0),
        "p95_sue_time_ms": float(np.percentile(success_times, 95) * 1000.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "network_name": row["network_name"],
                    "rank": int(row["rank"]),
                    "pair_index": int(row["pair_index"]),
                    "num_edges": int(row["num_edges"]),
                    "elapsed_sec": float(row["elapsed_sec"]),
                    "elapsed_ms": float(row["elapsed_ms"]),
                    "status": row["status"],
                    "loading_warning": _to_bool(row["loading_warning"]),
                    "used_flow_iter": int(row["used_flow_iter"]),
                    "flow_sum": float(row["flow_sum"]),
                    "error": row.get("error", ""),
                }
            )
    return rows


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: (int(row["rank"]), int(row["pair_index"])))


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset_key = args.network_name.strip().lower()
    if dataset_key not in DATASET_PKLS:
        raise ValueError(
            f"Unsupported network_name={args.network_name}. "
            f"Use one of: {', '.join(sorted(DATASET_PKLS))}."
        )

    input_pkl = Path(args.input_pkl) if args.input_pkl else DATASET_PKLS[dataset_key]
    input_pkl = input_pkl.expanduser().resolve()
    if not input_pkl.exists():
        raise FileNotFoundError(f"Input pickle not found: {input_pkl}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    network_display = DISPLAY_NAMES[dataset_key]
    run_name = args.run_name or f"{dataset_key}_sue_runtime"
    per_graph_csv = output_dir / f"{run_name}_per_graph.csv"
    summary_json = output_dir / f"{run_name}_summary.json"
    summary_csv = output_dir / f"{run_name}_summary.csv"

    print("=" * 72)
    print("SUE Runtime Benchmark")
    print("=" * 72)
    print(f"Network          : {network_display}")
    print(f"Input pickle     : {input_pkl}")
    print(f"Output directory : {output_dir}")
    print(f"Run name         : {run_name}")
    print(f"Started          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    pairs = load_pairs(input_pkl)
    _, _, test_idx = split_indices(
        len(pairs),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.num_test_graphs > 0:
        test_idx = test_idx[: args.num_test_graphs]

    target_pair_indices = {int(pair_idx) for pair_idx in test_idx}
    records: list[dict[str, Any]] = []
    if args.resume:
        existing_records = [
            row
            for row in read_csv_records(per_graph_csv)
            if int(row["pair_index"]) in target_pair_indices
        ]
        if existing_records:
            records.extend(existing_records)
            print(
                f"Resuming from {per_graph_csv}: "
                f"{len(existing_records)} existing records loaded."
            )

    completed_pair_indices = {int(row["pair_index"]) for row in records}
    new_records_since_checkpoint = 0

    for rank, pair_idx in enumerate(test_idx, start=1):
        if int(pair_idx) in completed_pair_indices:
            continue

        pair = pairs[int(pair_idx)]
        graph = pair["G_prime"]
        od_matrix = pair["od_matrix"]
        num_edges = int(graph.number_of_edges())

        status = "ok"
        error = ""
        loading_warning = False
        used_flow_iter = args.flow_iter
        flow_sum = float("nan")

        start = time.perf_counter()
        try:
            flows, loading_warning, used_flow_iter = solve_single_graph_sue(
                graph,
                od_matrix,
                node_ids=pair.get("node_ids"),
                centroid_nodes=pair.get("centroid_nodes"),
                max_iter=args.max_iter,
                convergence_threshold=args.convergence_threshold,
                theta=args.theta,
                value_iter=args.value_iter,
                value_tol=args.value_tol,
                flow_iter=args.flow_iter,
                flow_tol=args.flow_tol,
                retry_flow_iter=args.retry_flow_iter,
            )
            flow_sum = float(np.sum(flows))
        except Exception as exc:  # Keep benchmarking the remaining graphs.
            status = "failed"
            error = repr(exc)
        elapsed_sec = time.perf_counter() - start

        records.append(
            {
                "network_name": network_display,
                "rank": rank,
                "pair_index": int(pair_idx),
                "num_edges": num_edges,
                "elapsed_sec": elapsed_sec,
                "elapsed_ms": elapsed_sec * 1000.0,
                "status": status,
                "loading_warning": bool(loading_warning),
                "used_flow_iter": int(used_flow_iter),
                "flow_sum": flow_sum,
                "error": error,
            }
        )
        completed_pair_indices.add(int(pair_idx))
        new_records_since_checkpoint += 1

        if rank == 1 or rank % args.print_every == 0 or rank == len(test_idx):
            ok_times = [r["elapsed_sec"] for r in records if r["status"] == "ok"]
            mean_msg = np.mean(ok_times) if ok_times else float("nan")
            print(
                f"[{rank:>5d}/{len(test_idx)}] "
                f"last={elapsed_sec:.3f}s mean_ok={mean_msg:.3f}s status={status}"
            )

        if (
            args.checkpoint_every > 0
            and new_records_since_checkpoint >= args.checkpoint_every
        ):
            records = sort_records(records)
            write_csv(per_graph_csv, records)
            print(f"[checkpoint] wrote {len(records)} records to {per_graph_csv}")
            new_records_since_checkpoint = 0

    records = sort_records(records)
    if records:
        write_csv(per_graph_csv, records)

    stats = summarize_success(records)
    success_records = [row for row in records if row["status"] == "ok"]
    edge_counts = np.array([int(row["num_edges"]) for row in success_records], dtype=np.int64)
    warning_count = sum(1 for row in success_records if row["loading_warning"])

    summary = {
        "network_name": network_display,
        "run_name": run_name,
        "input_pkl": str(input_pkl),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_total_pairs": len(pairs),
        "num_test_available": int(len(split_indices(len(pairs), args.train_ratio, args.val_ratio, args.seed)[2])),
        "num_graphs_requested": int(args.num_test_graphs),
        "num_graphs_benchmarked": int(len(records)),
        "num_success": int(len(success_records)),
        "num_failed": int(len(records) - len(success_records)),
        "loading_warning_count": int(warning_count),
        "mean_num_edges": float(edge_counts.mean()) if edge_counts.size else float("nan"),
        "min_num_edges": int(edge_counts.min()) if edge_counts.size else 0,
        "max_num_edges": int(edge_counts.max()) if edge_counts.size else 0,
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "max_iter": int(args.max_iter),
        "convergence_threshold": float(args.convergence_threshold),
        "theta": float(args.theta),
        "value_iter": int(args.value_iter),
        "value_tol": float(args.value_tol),
        "flow_iter": int(args.flow_iter),
        "flow_tol": float(args.flow_tol),
        "retry_flow_iter": int(args.retry_flow_iter),
        **stats,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(summary_csv, [summary])

    print("=" * 72)
    print(f"Graphs solved     : {summary['num_success']} / {summary['num_graphs_benchmarked']}")
    print(f"Mean per graph    : {summary['mean_sue_time_sec']:.4f} s ({summary['mean_sue_time_ms']:.2f} ms)")
    print(f"Median per graph  : {summary['median_sue_time_sec']:.4f} s ({summary['median_sue_time_ms']:.2f} ms)")
    print(f"P95 per graph     : {summary['p95_sue_time_sec']:.4f} s ({summary['p95_sue_time_ms']:.2f} ms)")
    print(f"Per-graph CSV     : {per_graph_csv}")
    print(f"Summary JSON      : {summary_json}")
    print(f"Summary CSV       : {summary_csv}")
    print("=" * 72)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure SUE inference time per reconfigured graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--network_name", choices=sorted(DATASET_PKLS), required=True)
    parser.add_argument("--input_pkl", default="", help="Override the raw network-pairs pickle path.")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "results" / "sue_runtime"))
    parser.add_argument("--run_name", default="")
    parser.add_argument("--num_test_graphs", type=int, default=0, help="0 means all test graphs.")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--convergence_threshold", type=float, default=1e-5)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--value_iter", type=int, default=250)
    parser.add_argument("--value_tol", type=float, default=1e-8)
    parser.add_argument("--flow_iter", type=int, default=500)
    parser.add_argument("--flow_tol", type=float, default=1e-9)
    parser.add_argument("--retry_flow_iter", type=int, default=2000)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Write the per-graph CSV every N newly solved graphs; <=0 disables checkpointing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing per-graph CSV and skip already benchmarked pair_index rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
