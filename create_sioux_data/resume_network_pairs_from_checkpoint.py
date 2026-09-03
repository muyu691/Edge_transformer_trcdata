#!/usr/bin/env python3
"""Resume network-pair generation from a second-SUE checkpoint.

The original generator saves checkpoints as:
  {"completed_pairs": [...], "failed_indices": [...]}

Older checkpoints do not store the original sample index in each pair. This
script reconstructs the deterministic scenario list and recovers completed
indices by matching OD matrices, then solves only the remaining samples.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from network_registry import resolve_network_spec  # noqa: E402
from solve_network_pairs import (  # noqa: E402
    _save_final_dataset,
    generate_network_pairs,
    load_network_data,
    load_scenarios,
    solve_single_graph_sue,
)


def _od_key(od_matrix: np.ndarray) -> tuple:
    arr = np.ascontiguousarray(od_matrix)
    digest = hashlib.sha1(arr.view(np.uint8)).hexdigest()
    return arr.shape, str(arr.dtype), digest


def _load_checkpoint(
    path: Path,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "completed_pairs" in payload:
        return (
            list(payload.get("completed_pairs", [])),
            list(payload.get("failed_indices", [])),
            list(payload.get("failure_records", [])),
        )
    if isinstance(payload, dict) and "pairs" in payload:
        return (
            list(payload.get("pairs", [])),
            list(payload.get("failed_indices", [])),
            list(payload.get("failure_records", [])),
        )
    if isinstance(payload, list):
        return payload, [], []
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _recover_completed_by_index(
    completed_pairs: list[dict[str, Any]],
    od_matrices: np.ndarray,
) -> dict[int, dict[str, Any]]:
    od_to_index: dict[tuple, int] = {}
    for idx, od_matrix in enumerate(od_matrices):
        key = _od_key(od_matrix)
        if key in od_to_index:
            raise ValueError("Duplicate OD matrix key found; cannot infer checkpoint indices safely.")
        od_to_index[key] = idx

    completed_by_index: dict[int, dict[str, Any]] = {}
    unmatched = 0
    for pair in completed_pairs:
        if "sample_idx" in pair:
            idx = int(pair["sample_idx"])
        else:
            key = _od_key(pair["od_matrix"])
            idx = od_to_index.get(key, -1)
        if idx < 0:
            unmatched += 1
            continue
        pair["sample_idx"] = idx
        completed_by_index[idx] = pair

    if unmatched:
        raise ValueError(f"Could not recover indices for {unmatched} checkpoint pairs.")
    return completed_by_index


def _build_completed_pair(sample_idx: int, pair: dict[str, Any], solver_params: dict[str, Any]) -> dict[str, Any]:
    G_prime = pair["G_prime"]
    flows_new, retried, used_flow_iter = solve_single_graph_sue(
        G_prime,
        pair["od_matrix"],
        node_ids=pair.get("node_ids"),
        centroid_nodes=pair.get("centroid_nodes"),
        **solver_params,
    )
    edge_list_old = list(pair["G"].edges())
    edge_list_new = list(G_prime.edges())
    completed_pair = {
        "sample_idx": int(sample_idx),
        "od_matrix": pair["od_matrix"],
        "G": pair["G"],
        "G_prime": G_prime,
        "mutation_type": pair["mutation_type"],
        "mutation_info": pair["mutation_info"],
        "flows_old": pair["flows_old"],
        "flows_new": flows_new,
        "edge_list_old": edge_list_old,
        "edge_list_new": edge_list_new,
        "network_name": pair.get("network_name", "Unknown"),
        "node_ids": tuple(pair.get("node_ids", tuple(sorted(pair["G"].nodes())))),
        "centroid_nodes": tuple(pair.get("centroid_nodes", tuple())),
        "node_id_offset": pair.get("node_id_offset", 1),
    }
    return {
        "index": int(sample_idx),
        "pair": completed_pair,
        "retried": bool(retried),
        "used_flow_iter": int(used_flow_iter),
        "error": "",
        "mutation_type": pair["mutation_type"],
    }


def _solve_entry(
    sample_idx: int,
    pair: dict[str, Any],
    solver_params: dict[str, Any],
    result_queue,
) -> None:
    try:
        result_queue.put(_build_completed_pair(sample_idx, pair, solver_params))
    except Exception as exc:  # Keep the parent alive and mark the sample failed.
        result_queue.put(
            {
                "index": int(sample_idx),
                "pair": None,
                "retried": False,
                "used_flow_iter": int(solver_params.get("flow_iter", 0)),
                "error": repr(exc),
                "mutation_type": pair.get("mutation_type", "unknown"),
            }
        )


def _save_resume_checkpoint(
    completed_by_index: dict[int, dict[str, Any]],
    failed_indices: set[int],
    failure_records: list[dict[str, Any]],
    base_path: Path,
    processed_count: int,
) -> None:
    path = Path(f"{base_path}.resume_{processed_count}.pkl")
    pairs = [completed_by_index[i] for i in sorted(completed_by_index)]
    payload = {
        "completed_pairs": pairs,
        "failed_indices": sorted(failed_indices),
        "failure_records": list(failure_records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[Resume checkpoint] saved {len(pairs)} pairs to {path}", flush=True)


def solve_missing_with_timeouts(
    scenario_pairs_by_index: dict[int, dict[str, Any]],
    missing_indices: list[int],
    completed_by_index: dict[int, dict[str, Any]],
    failed_indices: set[int],
    failure_records: list[dict[str, Any]],
    solver_params: dict[str, Any],
    num_workers: int,
    timeout_sec: int,
    checkpoint_path: Path,
    checkpoint_interval: int,
) -> None:
    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(ctx_name)
    result_queue = ctx.Queue()
    pending = list(missing_indices)
    running: dict[int, tuple[Any, float]] = {}
    processed_missing = 0

    progress = tqdm(total=len(missing_indices), desc="Resume second SUE")

    def launch_next() -> None:
        if not pending:
            return
        idx = pending.pop(0)
        proc = ctx.Process(
            target=_solve_entry,
            args=(idx, scenario_pairs_by_index[idx], solver_params, result_queue),
        )
        proc.daemon = False
        proc.start()
        running[idx] = (proc, time.monotonic())

    for _ in range(max(1, num_workers)):
        launch_next()

    while running:
        consumed = False
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break
            consumed = True
            idx = int(result["index"])
            proc_start = running.pop(idx, None)
            if proc_start is not None:
                proc_start[0].join(timeout=1)

            if result["error"]:
                failed_indices.add(idx)
                failure_records.append(
                    {
                        "index": idx,
                        "stage": "new_sue",
                        "reason": result["error"],
                        "mutation_type": result["mutation_type"],
                    }
                )
                print(f"\n[Failed] sample {idx}: {result['error']}", flush=True)
            else:
                completed_by_index[idx] = result["pair"]
                if result["retried"]:
                    print(
                        f"\n[Retry succeeded] sample {idx} accepted after stronger "
                        f"Markov-logit limits (flow_iter={result['used_flow_iter']}).",
                        flush=True,
                    )

            processed_missing += 1
            progress.update(1)
            if checkpoint_interval > 0 and processed_missing % checkpoint_interval == 0:
                _save_resume_checkpoint(
                    completed_by_index,
                    failed_indices,
                    failure_records,
                    checkpoint_path,
                    processed_missing,
                )
            launch_next()

        now = time.monotonic()
        timed_out = []
        for idx, (proc, start) in list(running.items()):
            if timeout_sec > 0 and now - start > timeout_sec:
                timed_out.append(idx)
                proc.terminate()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                running.pop(idx, None)
                failed_indices.add(idx)
                failure_records.append(
                    {
                        "index": idx,
                        "stage": "new_sue",
                        "reason": f"Solver exceeded timeout_sec={timeout_sec}.",
                        "mutation_type": scenario_pairs_by_index[idx].get(
                            "mutation_type", "unknown"
                        ),
                    }
                )
                processed_missing += 1
                progress.update(1)
                print(f"\n[Timeout] sample {idx} exceeded {timeout_sec}s and was marked failed.", flush=True)
                launch_next()

        if timed_out and checkpoint_interval > 0:
            _save_resume_checkpoint(
                completed_by_index,
                failed_indices,
                failure_records,
                checkpoint_path,
                processed_missing,
            )
        if not consumed and not timed_out:
            time.sleep(2)

    progress.close()
    _save_resume_checkpoint(
        completed_by_index,
        failed_indices,
        failure_records,
        checkpoint_path,
        processed_missing,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume second-SUE network-pair generation from checkpoint.")
    parser.add_argument("--network_name", default="Anaheim")
    parser.add_argument("--dataset_root", default="../anaheim_data")
    parser.add_argument("--network_file", default="")
    parser.add_argument("--od_file", default="")
    parser.add_argument("--parser", default="tntp")
    parser.add_argument("--node_id_offset", type=int, default=1)
    parser.add_argument("--centroid_nodes", default="")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--convergence_threshold", type=float, default=1e-5)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--value_iter", type=int, default=250)
    parser.add_argument("--value_tol", type=float, default=1e-8)
    parser.add_argument("--flow_iter", type=int, default=500)
    parser.add_argument("--flow_tol", type=float, default=1e-9)
    parser.add_argument("--retry_flow_iter", type=int, default=2000)
    parser.add_argument("--retry_max_iter", type=int, default=300)
    parser.add_argument("--retry_value_iter", type=int, default=500)
    parser.add_argument(
        "--sue_loading_protocol",
        default="reasonable_links",
        choices=["stable_unrestricted", "reasonable_links", "legacy_unrestricted"],
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timeout_sec", type=int, default=3600)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    network_spec = resolve_network_spec(
        network_name=args.network_name,
        dataset_root=args.dataset_root,
        network_file=args.network_file,
        od_file=args.od_file,
        parser=args.parser,
        node_id_offset=args.node_id_offset,
        centroid_nodes=args.centroid_nodes,
    )
    network_data = load_network_data(network_spec)
    G_topo = network_data.graph

    scenarios_path = output_dir / "base_scenarios.npz"
    flows_old_path = output_dir / "flows_old.npy"
    if not scenarios_path.exists() or not flows_old_path.exists():
        raise FileNotFoundError("base_scenarios.npz and flows_old.npy are required for resume.")

    od_matrices, capacities, speeds = load_scenarios(str(scenarios_path))
    flows_old = np.load(flows_old_path)
    if od_matrices.shape[0] != args.num_samples or flows_old.shape[0] != args.num_samples:
        raise ValueError(
            f"Resume files do not match num_samples={args.num_samples}: "
            f"od={od_matrices.shape}, flows_old={flows_old.shape}"
        )

    print("=" * 72)
    print("Resume network-pair generation")
    print("=" * 72)
    print(f"network_name    : {network_data.network_name}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"output_dir      : {output_dir}")
    print(f"num_samples     : {args.num_samples}")
    print(f"num_workers     : {args.num_workers}")
    print(f"timeout_sec     : {args.timeout_sec}")
    print("=" * 72)

    scenario_pairs, mutation_failures = generate_network_pairs(
        G_topo=G_topo,
        od_matrices=od_matrices,
        capacities=capacities,
        speeds=speeds,
        seed=args.seed,
        network_name=network_data.network_name,
        node_ids=network_data.node_ids,
        centroid_nodes=network_data.centroid_nodes,
        node_id_offset=network_data.node_id_offset,
        mutation_policy=network_spec.mutation_policy,
    )
    scenario_pairs_by_index = {}
    invalid_old_indices = set()
    for pair in scenario_pairs:
        idx = int(pair["sample_idx"])
        flow_row = flows_old[idx]
        if (not np.all(np.isfinite(flow_row)) or np.any(flow_row < 0.0)
                or (od_matrices[idx].sum() > 0.0 and flow_row.sum() <= 0.0)):
            invalid_old_indices.add(idx)
            continue
        pair["flows_old"] = flow_row.copy()
        scenario_pairs_by_index[idx] = pair

    checkpoint_pairs, checkpoint_failed, failure_records = _load_checkpoint(checkpoint_path)
    completed_by_index = _recover_completed_by_index(checkpoint_pairs, od_matrices)
    failed_indices = set(int(i) for i in checkpoint_failed)
    failed_indices.update(invalid_old_indices)
    failed_indices.update(int(record["index"]) for record in mutation_failures)
    recorded_failure_indices = {
        int(record["index"])
        for record in failure_records
        if "index" in record
    }
    for idx in sorted(invalid_old_indices - recorded_failure_indices):
        failure_records.append(
            {
                "index": idx,
                "stage": "old_sue",
                "reason": "Loaded flows_old row is invalid or non-converged.",
            }
        )
    failure_records.extend(
        {"stage": "reconfiguration", **record}
        for record in mutation_failures
        if int(record["index"]) not in recorded_failure_indices
    )
    done_indices = set(completed_by_index) | failed_indices
    missing_indices = [idx for idx in sorted(scenario_pairs_by_index) if idx not in done_indices]

    print(f"checkpoint completed : {len(completed_by_index)}")
    print(f"checkpoint failed    : {len(failed_indices)}")
    print(f"missing to solve     : {len(missing_indices)}")

    solver_params = {
        "max_iter": args.max_iter,
        "convergence_threshold": args.convergence_threshold,
        "theta": args.theta,
        "value_iter": args.value_iter,
        "value_tol": args.value_tol,
        "flow_iter": args.flow_iter,
        "flow_tol": args.flow_tol,
        "retry_flow_iter": args.retry_flow_iter,
        "retry_max_iter": args.retry_max_iter,
        "retry_value_iter": args.retry_value_iter,
        "loading_protocol": args.sue_loading_protocol,
    }
    solve_missing_with_timeouts(
        scenario_pairs_by_index=scenario_pairs_by_index,
        missing_indices=missing_indices,
        completed_by_index=completed_by_index,
        failed_indices=failed_indices,
        failure_records=failure_records,
        solver_params=solver_params,
        num_workers=max(1, args.num_workers),
        timeout_sec=args.timeout_sec,
        checkpoint_path=output_dir / "pairs_completed.pkl",
        checkpoint_interval=args.checkpoint_interval,
    )

    completed_pairs = [completed_by_index[i] for i in sorted(completed_by_index)]
    failed_sorted = sorted(failed_indices)
    output_path = output_dir / "network_pairs_dataset.pkl"
    _save_final_dataset(
        completed_pairs,
        failed_sorted,
        str(output_path),
        failure_records=failure_records,
    )

    print("=" * 72)
    print("Resume complete")
    print(f"valid pairs    : {len(completed_pairs)}")
    print(f"failed/skipped : {len(failed_sorted)}")
    print(f"output         : {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
