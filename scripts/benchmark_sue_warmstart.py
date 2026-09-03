#!/usr/bin/env python
"""Paired legacy-SUE benchmark with a prior-preserving predicted warm start."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREATE_DATA_DIR = PROJECT_ROOT / "create_sioux_data"
sys.path.insert(0, str(CREATE_DATA_DIR))

from solve_network_pairs import solve_single_graph_sue  # noqa: E402


METHODS = ("cold", "old_flow", "predicted_warm")
_WORKER_PAIRS = None
_WORKER_PAIR_TO_PREDICTION = None
_WORKER_SOLVER_ARGS = None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    os.replace(temporary, path)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        pairs = payload.get("pairs", payload.get("completed_pairs"))
    else:
        pairs = payload
    if not isinstance(pairs, list):
        raise ValueError(f"Could not read a list of network pairs from {path}")
    return pairs


def load_prediction_store(path: Path) -> dict:
    with np.load(path) as payload:
        store = {key: payload[key] for key in payload.files}
    required = {"pair_indices", "offsets", "edge_counts", "predicted_flows", "true_flows"}
    missing = required - set(store)
    if missing:
        raise ValueError(f"Prediction file is missing arrays: {sorted(missing)}")
    num_graphs = int(store["pair_indices"].size)
    if store["offsets"].shape != (num_graphs + 1,):
        raise ValueError("Prediction offsets have an invalid shape.")
    if store["edge_counts"].shape != (num_graphs,):
        raise ValueError("Prediction edge_counts have an invalid shape.")
    if int(store["offsets"][-1]) != int(store["predicted_flows"].size):
        raise ValueError("Prediction offsets do not span predicted_flows.")
    if store["true_flows"].shape != store["predicted_flows"].shape:
        raise ValueError("Predicted and true flattened flow arrays have different shapes.")
    return store


def unpack_prediction_store(store: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for rank, pair_index in enumerate(store["pair_indices"]):
        start = int(store["offsets"][rank])
        end = int(store["offsets"][rank + 1])
        result[int(pair_index)] = (
            np.asarray(store["predicted_flows"][start:end], dtype=np.float64),
            np.asarray(store["true_flows"][start:end], dtype=np.float64),
        )
    if len(result) != int(store["pair_indices"].size):
        raise ValueError("Prediction pair_indices contains duplicates.")
    return result


def old_flow_projection(pair: dict) -> np.ndarray:
    old_edges = pair.get("edge_list_old", list(pair["G"].edges()))
    new_edges = pair.get("edge_list_new", list(pair["G_prime"].edges()))
    old_flows = np.asarray(pair["flows_old"], dtype=np.float64).reshape(-1)
    if len(old_edges) != old_flows.size:
        raise ValueError("flows_old length does not match edge_list_old.")
    old_by_edge = {tuple(edge): float(flow) for edge, flow in zip(old_edges, old_flows)}
    return np.asarray([old_by_edge.get(tuple(edge), 0.0) for edge in new_edges], dtype=np.float64)


def wmape(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.abs(lhs - rhs).sum() / max(np.abs(rhs).sum(), 1e-12))


def relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-12))


def conservation_relative_residual(pair: dict, flows: np.ndarray) -> float:
    """Measure aggregate node-balance error against the pair's OD matrix."""
    node_ids = tuple(int(value) for value in pair.get("node_ids", sorted(pair["G_prime"].nodes())))
    centroid_nodes = tuple(int(value) for value in pair.get("centroid_nodes", ()))
    od_matrix = np.asarray(pair["od_matrix"], dtype=np.float64)
    if len(centroid_nodes) != od_matrix.shape[0]:
        raise ValueError("centroid_nodes does not match the OD matrix dimension.")

    edges = pair.get("edge_list_new", list(pair["G_prime"].edges()))
    values = np.asarray(flows, dtype=np.float64).reshape(-1)
    if len(edges) != values.size:
        raise ValueError("Flow length does not match edge_list_new.")

    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    actual = np.zeros(len(node_ids), dtype=np.float64)
    for (u, v), flow in zip(edges, values):
        actual[node_to_index[int(u)]] -= flow
        actual[node_to_index[int(v)]] += flow

    expected = np.zeros(len(node_ids), dtype=np.float64)
    centroid_balance = od_matrix.sum(axis=0) - od_matrix.sum(axis=1)
    for node_id, balance in zip(centroid_nodes, centroid_balance):
        expected[node_to_index[node_id]] = balance
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))


def protocol_signature(args: argparse.Namespace, prediction_metadata: dict) -> str:
    payload = {
        "task": "sue-warmstart-legacy-prior-preserving-v3",
        "network": args.network,
        "input_pkl": str(args.input_pkl),
        "prediction_checkpoint_sha256": prediction_metadata["checkpoint_sha256"],
        "num_test_graphs": args.num_test_graphs,
        "max_iter": args.max_iter,
        "convergence_threshold": args.convergence_threshold,
        "theta": args.theta,
        "value_iter": args.value_iter,
        "value_tol": args.value_tol,
        "flow_iter": args.flow_iter,
        "flow_tol": args.flow_tol,
        "retry_flow_iter": args.retry_flow_iter,
        "sue_loading_protocol": args.sue_loading_protocol,
        "initial_loading_protocol": args.initial_loading_protocol,
        "methods": METHODS,
        "step_rule": args.step_rule,
        "residual_step_exponent": args.residual_step_exponent,
        "residual_step_min": args.residual_step_min,
        "residual_step_max": args.residual_step_max,
        "stop_before_update": args.stop_before_update,
        "step_warmup_iters": args.step_warmup_iters,
        "step_warmup_max": args.step_warmup_max,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _init_worker(pairs, pair_to_prediction, solver_args) -> None:
    global _WORKER_PAIRS, _WORKER_PAIR_TO_PREDICTION, _WORKER_SOLVER_ARGS
    _WORKER_PAIRS = pairs
    _WORKER_PAIR_TO_PREDICTION = pair_to_prediction
    _WORKER_SOLVER_ARGS = solver_args


def empty_record(network: str, rank: int, pair_index: int, method: str, protocol: str) -> dict:
    return {
        "protocol_signature": protocol,
        "network": network,
        "rank": int(rank),
        "pair_index": int(pair_index),
        "method": method,
        "num_edges": 0,
        "status": "failed",
        "converged": False,
        "iterations": 0,
        "network_loading_calls": 0,
        "elapsed_solver_sec": math.nan,
        "initialization_preparation_sec": math.nan,
        "elapsed_end_to_end_sec": math.nan,
        "initial_wmape_to_dataset": math.nan,
        "initial_relative_l2_to_dataset": math.nan,
        "final_wmape_to_dataset": math.nan,
        "final_relative_l2_to_dataset": math.nan,
        "final_wmape_to_cold": math.nan,
        "final_relative_l2_to_cold": math.nan,
        "initial_flow_gap": math.nan,
        "initial_cost_gap": math.nan,
        "final_flow_gap": math.nan,
        "final_cost_gap": math.nan,
        "final_update_gap": math.nan,
        "final_convergence_metric": math.nan,
        "loading_warning": False,
        "used_flow_iter": 0,
        "negative_initial_flow_count": 0,
        "initial_flow_mode": "",
        "initial_step_size": math.nan,
        "initial_prior_weight": math.nan,
        "mean_step_size": math.nan,
        "initial_conservation_residual": math.nan,
        "final_conservation_residual": math.nan,
        "error": "",
    }


def _run_pair(task: tuple[int, int, tuple[str, ...], str, float]) -> list[dict]:
    rank, pair_index, requested_methods, protocol, prediction_sec_per_graph = task
    pair = _WORKER_PAIRS[pair_index]
    predicted_raw, _ = _WORKER_PAIR_TO_PREDICTION[pair_index]
    reference = np.asarray(pair["flows_new"], dtype=np.float64).reshape(-1)
    graph = pair["G_prime"]
    od_matrix = pair["od_matrix"]
    num_edges = int(graph.number_of_edges())

    method_order = list(METHODS)
    shift = (rank - 1) % len(method_order)
    method_order = method_order[shift:] + method_order[:shift]
    method_order = [method for method in method_order if method in requested_methods]
    records_by_method = {}
    final_flows = {}

    for method in method_order:
        record = empty_record(_WORKER_SOLVER_ARGS["network"], rank, pair_index, method, protocol)
        record["num_edges"] = num_edges
        try:
            prep_start = time.perf_counter()
            if method == "cold":
                initial = None
                initial_flow_mode = "direct"
            elif method == "old_flow":
                initial = old_flow_projection(pair)
                initial_flow_mode = "direct"
            elif method == "predicted_warm":
                initial = predicted_raw.copy()
                initial_flow_mode = "direct"
            else:
                raise ValueError(f"Unsupported benchmark method: {method!r}")
            preparation_sec = time.perf_counter() - prep_start

            start = time.perf_counter()
            flows, loading_warning, used_flow_iter, diagnostics = solve_single_graph_sue(
                graph,
                od_matrix,
                node_ids=pair.get("node_ids"),
                centroid_nodes=pair.get("centroid_nodes"),
                max_iter=_WORKER_SOLVER_ARGS["max_iter"],
                convergence_threshold=_WORKER_SOLVER_ARGS["convergence_threshold"],
                theta=_WORKER_SOLVER_ARGS["theta"],
                value_iter=_WORKER_SOLVER_ARGS["value_iter"],
                value_tol=_WORKER_SOLVER_ARGS["value_tol"],
                flow_iter=_WORKER_SOLVER_ARGS["flow_iter"],
                flow_tol=_WORKER_SOLVER_ARGS["flow_tol"],
                retry_flow_iter=_WORKER_SOLVER_ARGS["retry_flow_iter"],
                initial_flows=initial,
                return_diagnostics=True,
                loading_protocol=_WORKER_SOLVER_ARGS["sue_loading_protocol"],
                initial_flow_mode=initial_flow_mode,
                initial_loading_protocol=_WORKER_SOLVER_ARGS["initial_loading_protocol"],
                step_rule=_WORKER_SOLVER_ARGS["step_rule"],
                residual_step_exponent=_WORKER_SOLVER_ARGS["residual_step_exponent"],
                residual_step_min=_WORKER_SOLVER_ARGS["residual_step_min"],
                residual_step_max=_WORKER_SOLVER_ARGS["residual_step_max"],
                stop_before_update=_WORKER_SOLVER_ARGS["stop_before_update"],
                step_warmup_iters=_WORKER_SOLVER_ARGS["step_warmup_iters"],
                step_warmup_max=_WORKER_SOLVER_ARGS["step_warmup_max"],
            )
            elapsed_solver = time.perf_counter() - start
            model_time = prediction_sec_per_graph if method.startswith("predicted") else 0.0
            initial_metric_flow = np.asarray(
                diagnostics.pop("initial_state_flows"), dtype=np.float64
            ).reshape(-1)
            record.update(
                {
                    "status": "ok",
                    "converged": bool(diagnostics["converged"]),
                    "iterations": int(diagnostics["iterations"]),
                    "network_loading_calls": int(diagnostics["network_loading_calls"]),
                    "elapsed_solver_sec": float(elapsed_solver),
                    "initialization_preparation_sec": float(preparation_sec),
                    "elapsed_end_to_end_sec": float(elapsed_solver + preparation_sec + model_time),
                    "final_wmape_to_dataset": wmape(flows, reference),
                    "final_relative_l2_to_dataset": relative_l2(flows, reference),
                    "initial_flow_gap": float(diagnostics["initial_flow_gap"]),
                    "initial_cost_gap": float(diagnostics["initial_cost_gap"]),
                    "final_flow_gap": float(diagnostics["final_flow_gap"]),
                    "final_cost_gap": float(diagnostics["final_cost_gap"]),
                    "final_update_gap": float(diagnostics["final_update_gap"]),
                    "final_convergence_metric": float(diagnostics["final_convergence_metric"]),
                    "loading_warning": bool(loading_warning),
                    "used_flow_iter": int(used_flow_iter),
                    "negative_initial_flow_count": int(diagnostics["negative_initial_flow_count"]),
                    "initial_flow_mode": str(diagnostics["initial_flow_mode"]),
                    "initial_step_size": float(diagnostics["initial_step_size"]),
                    "initial_prior_weight": float(diagnostics["initial_prior_weight"]),
                    "mean_step_size": float(diagnostics["mean_step_size"]),
                    "initial_conservation_residual": conservation_relative_residual(
                        pair, initial_metric_flow
                    ),
                    "final_conservation_residual": conservation_relative_residual(pair, flows),
                }
            )
            record["initial_wmape_to_dataset"] = wmape(initial_metric_flow, reference)
            record["initial_relative_l2_to_dataset"] = relative_l2(initial_metric_flow, reference)
            final_flows[method] = flows
        except Exception as exc:
            record["error"] = repr(exc)
        records_by_method[method] = record

    if "cold" in final_flows:
        cold_final = final_flows["cold"]
        for method, flows in final_flows.items():
            records_by_method[method]["final_wmape_to_cold"] = wmape(flows, cold_final)
            records_by_method[method]["final_relative_l2_to_cold"] = relative_l2(flows, cold_final)
    return [records_by_method[method] for method in requested_methods]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def summarize(records: list[dict], prediction_metadata: dict, protocol: str, args) -> dict:
    normalized = []
    for row in records:
        converted = dict(row)
        for field in ("rank", "pair_index", "num_edges", "iterations", "network_loading_calls"):
            converted[field] = int(row[field])
        converted["converged"] = as_bool(row["converged"])
        for field in (
            "elapsed_solver_sec",
            "initialization_preparation_sec",
            "elapsed_end_to_end_sec",
            "initial_wmape_to_dataset",
            "initial_relative_l2_to_dataset",
            "final_wmape_to_dataset",
            "final_relative_l2_to_dataset",
            "final_wmape_to_cold",
            "final_relative_l2_to_cold",
            "initial_flow_gap",
            "final_convergence_metric",
            "initial_step_size",
            "initial_prior_weight",
            "mean_step_size",
            "initial_conservation_residual",
            "final_conservation_residual",
        ):
            converted[field] = float(row[field])
        normalized.append(converted)

    by_pair = {}
    for row in normalized:
        by_pair.setdefault(row["pair_index"], {})[row["method"]] = row
    paired_successful_indices = [
        pair_index
        for pair_index, methods in by_pair.items()
        if all(method in methods and methods[method]["status"] == "ok" for method in METHODS)
    ]
    if not paired_successful_indices:
        raise RuntimeError("No graph completed successfully under all initialization methods.")
    paired_converged_indices = [
        pair_index
        for pair_index, methods in by_pair.items()
        if all(
            method in methods
            and methods[method]["status"] == "ok"
            and methods[method]["converged"]
            for method in METHODS
        )
    ]

    method_summaries = {}
    cold_rows = [by_pair[index]["cold"] for index in paired_successful_indices]
    cold_mean_time = float(np.mean([row["elapsed_solver_sec"] for row in cold_rows]))
    cold_mean_end = float(np.mean([row["elapsed_end_to_end_sec"] for row in cold_rows]))
    cold_mean_iter = float(np.mean([row["iterations"] for row in cold_rows]))
    for method in METHODS:
        all_rows = [row for row in normalized if row["method"] == method]
        paired = [by_pair[index][method] for index in paired_successful_indices]
        pairwise_converged_indices = [
            index
            for index in paired_successful_indices
            if by_pair[index]["cold"]["converged"] and by_pair[index][method]["converged"]
        ]
        solver_times = np.asarray([row["elapsed_solver_sec"] for row in paired], dtype=np.float64)
        end_times = np.asarray([row["elapsed_end_to_end_sec"] for row in paired], dtype=np.float64)
        iterations = np.asarray([row["iterations"] for row in paired], dtype=np.float64)
        loading_calls = np.asarray([row["network_loading_calls"] for row in paired], dtype=np.float64)
        cold_iterations = np.asarray(
            [by_pair[index]["cold"]["iterations"] for index in paired_successful_indices],
            dtype=np.float64,
        )
        iteration_savings = cold_iterations - iterations
        if pairwise_converged_indices:
            pairwise_cold_iterations = np.asarray(
                [by_pair[index]["cold"]["iterations"] for index in pairwise_converged_indices],
                dtype=np.float64,
            )
            pairwise_method_iterations = np.asarray(
                [by_pair[index][method]["iterations"] for index in pairwise_converged_indices],
                dtype=np.float64,
            )
            converged_iteration_reduction = float(
                1.0 - pairwise_method_iterations.mean() / pairwise_cold_iterations.mean()
            )
        else:
            converged_iteration_reduction = math.nan
        method_summaries[method] = {
            "num_records": int(len(all_rows)),
            "num_success": int(sum(row["status"] == "ok" for row in all_rows)),
            "num_converged": int(sum(row["status"] == "ok" and row["converged"] for row in all_rows)),
            "convergence_rate": float(
                sum(row["status"] == "ok" and row["converged"] for row in all_rows)
                / max(len(all_rows), 1)
            ),
            "paired_comparison_count": int(len(paired)),
            "paired_converged_count": int(len(paired_converged_indices)),
            "mean_iterations": float(iterations.mean()),
            "median_iterations": float(np.median(iterations)),
            "p95_iterations": float(np.percentile(iterations, 95)),
            "mean_network_loading_calls": float(loading_calls.mean()),
            "mean_solver_sec": float(solver_times.mean()),
            "median_solver_sec": float(np.median(solver_times)),
            "p95_solver_sec": float(np.percentile(solver_times, 95)),
            "mean_end_to_end_sec": float(end_times.mean()),
            "solver_speedup_vs_cold": float(cold_mean_time / solver_times.mean()),
            "end_to_end_speedup_vs_cold": float(cold_mean_end / end_times.mean()),
            "iteration_reduction_vs_cold": float(1.0 - iterations.mean() / cold_mean_iter),
            "mean_iterations_saved_vs_cold": float(iteration_savings.mean()),
            "median_iterations_saved_vs_cold": float(np.median(iteration_savings)),
            "fraction_fewer_iterations_than_cold": float(np.mean(iteration_savings > 0.0)),
            "pairwise_both_converged_count": int(len(pairwise_converged_indices)),
            "iteration_reduction_vs_cold_both_converged": converged_iteration_reduction,
            "mean_initial_flow_gap": finite_mean(row["initial_flow_gap"] for row in paired),
            "mean_final_convergence_metric": float(
                np.mean([row["final_convergence_metric"] for row in paired])
            ),
            "mean_final_wmape_to_dataset": finite_mean(
                row["final_wmape_to_dataset"] for row in paired
            ),
            "mean_final_relative_l2_to_cold": float(
                np.mean([row["final_relative_l2_to_cold"] for row in paired])
            ),
            "mean_initial_wmape_to_dataset": finite_mean(
                row["initial_wmape_to_dataset"] for row in paired
            ),
            "mean_initial_conservation_residual": finite_mean(
                row["initial_conservation_residual"] for row in paired
            ),
            "mean_final_conservation_residual": finite_mean(
                row["final_conservation_residual"] for row in paired
            ),
            "mean_initial_step_size": finite_mean(row["initial_step_size"] for row in paired),
            "mean_initial_prior_weight": finite_mean(
                row["initial_prior_weight"] for row in paired
            ),
            "mean_step_size": finite_mean(row["mean_step_size"] for row in paired),
        }

    return {
        "task": "sue-warmstart-benchmark",
        "protocol_signature": protocol,
        "network": args.network,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_pkl": str(args.input_pkl),
        "prediction_file": str(args.predictions),
        "checkpoint": prediction_metadata["checkpoint"],
        "checkpoint_sha256": prediction_metadata["checkpoint_sha256"],
        "prediction_forward_milliseconds_per_graph": prediction_metadata[
            "forward_milliseconds_per_graph"
        ],
        "num_graphs_requested": int(args.num_test_graphs),
        "num_graphs_benchmarked": int(len(by_pair)),
        "num_paired_successful": int(len(paired_successful_indices)),
        "num_paired_converged": int(len(paired_converged_indices)),
        "num_workers": int(args.num_workers),
        "solver": {
            "max_iter": int(args.max_iter),
            "convergence_threshold": float(args.convergence_threshold),
            "theta": float(args.theta),
            "value_iter": int(args.value_iter),
            "value_tol": float(args.value_tol),
            "flow_iter": int(args.flow_iter),
            "flow_tol": float(args.flow_tol),
            "retry_flow_iter": int(args.retry_flow_iter),
            "loading_protocol": args.sue_loading_protocol,
            "initial_loading_protocol": args.initial_loading_protocol,
            "step_rule": args.step_rule,
            "residual_step_exponent": float(args.residual_step_exponent),
            "residual_step_min": float(args.residual_step_min),
            "residual_step_max": float(args.residual_step_max),
            "stop_before_update": bool(args.stop_before_update),
            "iteration_definition": "number of outer flow updates",
            "step_warmup_iters": int(args.step_warmup_iters),
            "step_warmup_max": float(args.step_warmup_max),
        },
        "methods": method_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, choices=("SiouxFalls", "EMA", "Anaheim"))
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prediction-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-test-graphs", type=int, default=0, help="0 uses every exported graph.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--value-iter", type=int, default=250)
    parser.add_argument("--value-tol", type=float, default=1e-8)
    parser.add_argument("--flow-iter", type=int, default=500)
    parser.add_argument("--flow-tol", type=float, default=1e-9)
    parser.add_argument("--retry-flow-iter", type=int, default=2000)
    parser.add_argument(
        "--sue-loading-protocol",
        choices=(
            "reasonable_links",
            "stable_reasonable_links",
            "legacy_unrestricted",
            "stable_unrestricted",
        ),
        default="reasonable_links",
    )
    parser.add_argument(
        "--initial-loading-protocol",
        choices=(
            "legacy_unrestricted",
            "stable_unrestricted",
            "reasonable_links",
            "stable_reasonable_links",
        ),
        default="stable_unrestricted",
        help="Loading used to construct the cold initial state.",
    )
    parser.add_argument(
        "--step-rule",
        choices=("msa_sr", "residual_aware"),
        default="msa_sr",
    )
    parser.add_argument("--residual-step-exponent", type=float, default=0.5)
    parser.add_argument("--residual-step-min", type=float, default=0.02)
    parser.add_argument("--residual-step-max", type=float, default=0.80)
    parser.add_argument("--step-warmup-iters", type=int, default=0)
    parser.add_argument("--step-warmup-max", type=float, default=1.0)
    parser.add_argument(
        "--stop-before-update",
        action="store_true",
        help="Return with zero updates when the supplied state already meets tolerance.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_pkl = args.input_pkl.expanduser().resolve()
    args.predictions = args.predictions.expanduser().resolve()
    args.prediction_metadata = args.prediction_metadata.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_metadata = load_json(args.prediction_metadata)
    if str(prediction_metadata["network"]).lower() != args.network.lower():
        raise ValueError("Prediction metadata network does not match --network.")
    protocol = protocol_signature(args, prediction_metadata)
    per_graph_path = args.output_dir / "per_graph.csv"
    summary_path = args.output_dir / "summary.json"

    print(f"Loading raw pairs: {args.input_pkl}")
    pairs = load_pairs(args.input_pkl)
    prediction_store = load_prediction_store(args.predictions)
    pair_to_prediction = unpack_prediction_store(prediction_store)
    pair_indices = [int(value) for value in prediction_store["pair_indices"]]
    if args.num_test_graphs > 0:
        pair_indices = pair_indices[: args.num_test_graphs]

    for pair_index in pair_indices:
        if pair_index < 0 or pair_index >= len(pairs):
            raise IndexError(f"Prediction pair index {pair_index} is outside raw dataset.")
        predicted, exported_true = pair_to_prediction[pair_index]
        raw_true = np.asarray(pairs[pair_index]["flows_new"], dtype=np.float64).reshape(-1)
        if predicted.size != int(pairs[pair_index]["G_prime"].number_of_edges()):
            raise RuntimeError(f"Edge-count mismatch for pair_index={pair_index}.")
        if not np.allclose(exported_true, raw_true, rtol=2e-4, atol=0.25):
            raise RuntimeError(
                f"PyG/raw edge ordering or label mismatch for pair_index={pair_index}."
            )
    print(f"Validated prediction alignment for {len(pair_indices)} test graphs.")

    records = read_csv(per_graph_path) if args.resume else []
    if records and any(row.get("protocol_signature") != protocol for row in records):
        raise RuntimeError("Existing per_graph.csv was produced by another protocol.")
    target_set = set(pair_indices)
    records = [row for row in records if int(row["pair_index"]) in target_set]
    completed = {(int(row["pair_index"]), row["method"]) for row in records}
    tasks = []
    prediction_sec = float(prediction_metadata["forward_milliseconds_per_graph"]) / 1000.0
    for rank, pair_index in enumerate(pair_indices, start=1):
        missing = tuple(method for method in METHODS if (pair_index, method) not in completed)
        if missing:
            tasks.append((rank, pair_index, missing, protocol, prediction_sec))

    solver_args = {
        "network": args.network,
        "max_iter": args.max_iter,
        "convergence_threshold": args.convergence_threshold,
        "theta": args.theta,
        "value_iter": args.value_iter,
        "value_tol": args.value_tol,
        "flow_iter": args.flow_iter,
        "flow_tol": args.flow_tol,
        "retry_flow_iter": args.retry_flow_iter,
        "sue_loading_protocol": args.sue_loading_protocol,
        "initial_loading_protocol": args.initial_loading_protocol,
        "step_rule": args.step_rule,
        "residual_step_exponent": args.residual_step_exponent,
        "residual_step_min": args.residual_step_min,
        "residual_step_max": args.residual_step_max,
        "stop_before_update": args.stop_before_update,
        "step_warmup_iters": args.step_warmup_iters,
        "step_warmup_max": args.step_warmup_max,
    }
    print(
        f"Running {len(tasks)} remaining paired graphs with {args.num_workers} worker(s); "
        f"protocol={protocol}"
    )
    completed_since_write = 0

    def consume(new_rows: list[dict]) -> None:
        nonlocal completed_since_write, records
        records.extend(new_rows)
        completed_since_write += 1
        if args.checkpoint_every > 0 and completed_since_write >= args.checkpoint_every:
            records.sort(key=lambda row: (int(row["rank"]), METHODS.index(row["method"])))
            write_csv(per_graph_path, records)
            completed_since_write = 0
            print(f"[checkpoint] {len(records)} method records")

    if args.num_workers <= 1:
        _init_worker(pairs, pair_to_prediction, solver_args)
        for task_index, task in enumerate(tasks, start=1):
            consume(_run_pair(task))
            if task_index == 1 or task_index % 10 == 0 or task_index == len(tasks):
                print(f"[{task_index}/{len(tasks)}] pair_index={task[1]}")
    elif tasks:
        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(pairs, pair_to_prediction, solver_args),
        ) as executor:
            future_to_task = {executor.submit(_run_pair, task): task for task in tasks}
            for done_count, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                consume(future.result())
                if done_count == 1 or done_count % 10 == 0 or done_count == len(tasks):
                    print(f"[{done_count}/{len(tasks)}] pair_index={task[1]}")

    records.sort(key=lambda row: (int(row["rank"]), METHODS.index(row["method"])))
    write_csv(per_graph_path, records)
    summary = summarize(records, prediction_metadata, protocol, args)
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"Per-graph results: {per_graph_path}")
    print(f"Summary          : {summary_path}")


if __name__ == "__main__":
    main()
