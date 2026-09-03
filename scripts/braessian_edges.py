#!/usr/bin/env python3
"""Rank Braessian edge removals with the Edge Transformer and hybrid SUE checks.

The experiment has three deliberately separated stages:

1. ``score`` builds every feasible single-edge-removal candidate on held-out EMA
   scenarios and predicts the post-removal link flow in GPU batches.
2. ``oracle`` uses the existing cold-start SUE solver for a reproducible hybrid
   manifest: exhaustive checks on a random audit subset and only model top-k
   candidates on all remaining scenarios.
3. ``summarize`` reports full-ranking accuracy only on the exhaustive audit
   subset, and deployable top-k screening utility over every scenario.

No SUE implementation is changed here.  The oracle calls the project's
``solve_single_graph_sue`` with the selected existing loading protocol.
"""

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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREATE_DATA_DIR = PROJECT_ROOT / "create_sioux_data"
for path in (PROJECT_ROOT, CREATE_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from solve_network_pairs import solve_single_graph_sue  # noqa: E402
from sue_solver import bpr_travel_time  # noqa: E402


SCORE_FIELDS = (
    "protocol_signature",
    "row_id",
    "scenario_rank",
    "pair_index",
    "candidate_rank",
    "model_rank",
    "edge_u",
    "edge_v",
    "num_edges_original",
    "num_edges_after_removal",
    "baseline_tstt",
    "model_removed_tstt",
    "model_benefit",
    "model_relative_benefit",
    "predicted_total_flow",
    "negative_prediction_count",
    "negative_prediction_fraction",
    "old_flow",
    "vc_ratio",
    "link_tstt",
    "marginal_congestion",
    "score_random",
    "score_low_flow",
    "score_low_vc",
    "score_high_flow",
    "score_high_vc",
    "score_high_link_tstt",
    "score_high_marginal_congestion",
    "model_forward_sec",
)

ORACLE_FIELDS = (
    "oracle_protocol_signature",
    "score_protocol_signature",
    "row_id",
    "scenario_rank",
    "pair_index",
    "candidate_rank",
    "edge_u",
    "edge_v",
    "evaluation_scope",
    "model_rank",
    "status",
    "converged",
    "loading_warning",
    "used_flow_iter",
    "iterations",
    "network_loading_calls",
    "final_flow_gap",
    "final_cost_gap",
    "final_update_gap",
    "final_convergence_metric",
    "elapsed_solver_sec",
    "oracle_removed_tstt",
    "oracle_benefit",
    "oracle_relative_benefit",
    "error",
)

SCORE_TIMING_FIELDS = (
    "scenario_rank",
    "pair_index",
    "num_candidates",
    "tensor_build_sec",
    "batch_update_sec",
    "host_to_device_sec",
    "model_forward_sec",
    "tstt_and_ranking_sec",
    "serialization_sec",
    "scenario_wall_sec",
)

METHOD_SCORES = {
    "edge_transformer": "model_relative_benefit",
    "low_old_flow": "score_low_flow",
    "low_vc_ratio": "score_low_vc",
    "high_old_flow": "score_high_flow",
    "high_vc_ratio": "score_high_vc",
    "high_link_tstt": "score_high_link_tstt",
    "high_marginal_congestion": "score_high_marginal_congestion",
    "random": "score_random",
}

CANDIDATE_KEY_FIELDS = (
    "scenario_rank",
    "pair_index",
    "candidate_rank",
    "edge_u",
    "edge_v",
)

_ORACLE_PAIRS: list[dict[str, Any]] | None = None
_ORACLE_ARGS: dict[str, Any] | None = None
_ORACLE_SIGNATURE = ""
_SCORE_SIGNATURE = ""


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_marker(
    marker_path: Path,
    input_pkl: Path,
    dataset_dir: Path | None = None,
) -> dict[str, str]:
    """Require a passed reconstruction audit bound to these exact EMA inputs."""
    marker = marker_path.expanduser().resolve()
    expected_input = input_pkl.expanduser().resolve()
    if not marker.exists():
        raise FileNotFoundError(
            f"Validated raw-pair marker is missing: {marker}. "
            "Run run_reconstruct_ema_raw_from_pyg_vera_cpu.sh first."
        )
    values: dict[str, str] = {}
    for line in marker.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if "input_pkl" not in values:
        raise RuntimeError(f"Validation marker has no input_pkl binding: {marker}")
    marked_input = Path(values["input_pkl"]).expanduser().resolve()
    if marked_input != expected_input:
        raise RuntimeError(
            f"Validation marker input mismatch: marker={marked_input}, requested={expected_input}."
        )
    if dataset_dir is not None:
        expected_dataset = dataset_dir.expanduser().resolve()
        marked_dataset_value = values.get("pyg_dir", values.get("dataset_dir", ""))
        if not marked_dataset_value:
            raise RuntimeError(f"Validation marker has no pyg_dir binding: {marker}")
        marked_dataset = Path(marked_dataset_value).expanduser().resolve()
        if marked_dataset != expected_dataset:
            raise RuntimeError(
                "Validation marker PyG-directory mismatch: "
                f"marker={marked_dataset}, requested={expected_dataset}."
            )
    validation_report = marker.parent / "raw_pairs_validation.json"
    if not validation_report.exists():
        raise FileNotFoundError(
            f"Validation marker exists but its audit report is missing: {validation_report}"
        )
    report = load_json(validation_report)
    if report.get("passed") is not True:
        raise RuntimeError(f"Raw-pair validation report is not passed: {validation_report}")
    values["marker_sha256"] = sha256_file(marker)
    values["validation_report_sha256"] = sha256_file(validation_report)
    return values


def short_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def as_int(row: dict, key: str) -> int:
    return int(row[key])


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def candidate_key(row: dict) -> tuple[int, int, int, int, int]:
    return tuple(as_int(row, key) for key in CANDIDATE_KEY_FIELDS)


def graph_arrays(graph: nx.DiGraph) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    edges = [(int(u), int(v)) for u, v in graph.edges()]
    capacities = np.asarray(
        [graph[u][v]["capacity"] for u, v in edges], dtype=np.float64
    )
    free_flow_times_list = []
    for u, v in edges:
        attributes = graph[u][v]
        if "free_flow_time" in attributes:
            free_flow_times_list.append(float(attributes["free_flow_time"]))
        else:
            free_flow_times_list.append(
                float(attributes["length"])
                / max(float(attributes["speed"]), 1e-12)
                * 60.0
            )
    free_flow_times = np.asarray(free_flow_times_list, dtype=np.float64)
    return edges, capacities, free_flow_times


def total_system_travel_time(graph: nx.DiGraph, flows: np.ndarray) -> float:
    """Return TSTT in vehicle-minutes/hour for the project's minute-based BPR cost."""
    _, capacities, free_flow_times = graph_arrays(graph)
    values = np.asarray(flows, dtype=np.float64).reshape(-1)
    if values.shape != capacities.shape:
        raise ValueError(
            f"Flow length {values.size} does not match graph edges {capacities.size}."
        )
    travel_times = bpr_travel_time(values, capacities, free_flow_times)
    return float(np.dot(values, travel_times))


def total_system_travel_time_arrays(
    flows: np.ndarray,
    capacities: np.ndarray,
    free_flow_times: np.ndarray,
) -> np.ndarray:
    """Vectorized TSTT over one or more equally-sized candidate networks."""
    values = np.asarray(flows, dtype=np.float64)
    capacity_values = np.asarray(capacities, dtype=np.float64)
    t0_values = np.asarray(free_flow_times, dtype=np.float64)
    if values.shape != capacity_values.shape or values.shape != t0_values.shape:
        raise ValueError(
            "Flow/capacity/free-flow-time arrays must have identical shapes: "
            f"{values.shape}, {capacity_values.shape}, {t0_values.shape}."
        )
    travel_times = bpr_travel_time(values, capacity_values, t0_values)
    return np.sum(values * travel_times, axis=-1)


def stable_model_order(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -as_float(row, "model_relative_benefit"),
            as_int(row, "candidate_rank"),
            as_int(row, "row_id"),
        ),
    )


def build_hybrid_manifest(
    score_rows: list[dict],
    num_audit_scenarios: int,
    audit_seed: int,
    final_check_top_k: int,
    num_shards: int,
) -> tuple[list[dict], list[int], dict[int, int]]:
    """Select exhaustive audit rows and model-top-k deployment rows deterministically."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in score_rows:
        grouped[as_int(row, "scenario_rank")].append(row)
    scenario_ranks = sorted(grouped)
    if not scenario_ranks:
        raise ValueError("Cannot build a hybrid manifest from an empty score table.")
    if not 0 <= num_audit_scenarios <= len(scenario_ranks):
        raise ValueError("--audit-scenarios must be between 0 and the scenario count.")
    if final_check_top_k <= 0:
        raise ValueError("--final-check-k must be positive.")
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")

    rng = np.random.default_rng(audit_seed)
    audit_ranks = sorted(
        int(value)
        for value in rng.permutation(np.asarray(scenario_ranks, dtype=np.int64))[
            :num_audit_scenarios
        ]
    )
    audit_set = set(audit_ranks)
    audit_order = [
        int(value)
        for value in np.random.default_rng(audit_seed + 1).permutation(audit_ranks)
    ]
    deployment_ranks = [rank for rank in scenario_ranks if rank not in audit_set]
    deployment_order = [
        int(value)
        for value in np.random.default_rng(audit_seed + 2).permutation(deployment_ranks)
    ]
    shard_assignment: dict[int, int] = {}
    for index, rank in enumerate(audit_order):
        shard_assignment[rank] = index % num_shards
    for index, rank in enumerate(deployment_order):
        shard_assignment[rank] = index % num_shards

    selected = []
    for scenario_rank in scenario_ranks:
        rows = grouped[scenario_rank]
        ordered = stable_model_order(rows)
        if final_check_top_k > len(ordered):
            raise ValueError(
                f"final_check_top_k={final_check_top_k} exceeds candidates={len(ordered)}."
            )
        model_rank_by_id = {
            as_int(row, "row_id"): rank
            for rank, row in enumerate(ordered, start=1)
        }
        if scenario_rank in audit_set:
            chosen = rows
            scope = "exhaustive_audit"
        else:
            chosen = ordered[:final_check_top_k]
            scope = "model_top_k"
        for row in chosen:
            item = dict(row)
            item["evaluation_scope"] = scope
            item["model_rank"] = model_rank_by_id[as_int(row, "row_id")]
            item["hybrid_shard_id"] = shard_assignment[scenario_rank]
            selected.append(item)
    selected.sort(key=lambda row: as_int(row, "row_id"))
    return selected, audit_ranks, shard_assignment


def validate_base_pair(pair: dict, pair_index: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    if str(pair.get("network_name", "")).lower() != "ema":
        raise ValueError(f"pair_index={pair_index} is not an EMA pair.")
    graph = pair["G"]
    edges = [(int(u), int(v)) for u, v in graph.edges()]
    declared = [tuple(map(int, edge)) for edge in pair.get("edge_list_old", edges)]
    if edges != declared:
        raise ValueError(
            f"pair_index={pair_index}: list(G.edges()) does not match edge_list_old."
        )
    flows = np.asarray(pair["flows_old"], dtype=np.float64).reshape(-1)
    if flows.size != len(edges):
        raise ValueError(
            f"pair_index={pair_index}: flows_old={flows.size}, edges={len(edges)}."
        )
    if not np.all(np.isfinite(flows)):
        raise ValueError(f"pair_index={pair_index}: flows_old contains NaN/Inf.")
    return edges, flows


def feasible_single_edge_removals(graph: nx.DiGraph) -> list[tuple[int, int]]:
    """Keep directed edges whose removal preserves strong connectivity."""
    feasible = []
    for u, v in graph.edges():
        candidate = graph.copy()
        candidate.remove_edge(u, v)
        if nx.is_strongly_connected(candidate):
            feasible.append((int(u), int(v)))
    return feasible


def choose_scenarios(
    test_indices: np.ndarray,
    num_scenarios: int,
    selection_seed: int,
) -> np.ndarray:
    values = np.asarray(test_indices, dtype=np.int64).reshape(-1)
    if num_scenarios <= 0:
        return values.copy()
    if num_scenarios > values.size:
        raise ValueError(
            f"Requested {num_scenarios} scenarios, but test split has {values.size}."
        )
    rng = np.random.default_rng(selection_seed)
    return rng.permutation(values)[:num_scenarios]


def build_identity_template_pair(pair: dict) -> dict:
    """Build one base PyG record; deletion candidates are created by tensor masks."""
    old_edges = [(int(u), int(v)) for u, v in pair["G"].edges()]
    return {
        "od_matrix": np.asarray(pair["od_matrix"], dtype=np.float64),
        "G": pair["G"],
        "G_prime": pair["G"],
        "flows_old": np.asarray(pair["flows_old"], dtype=np.float64),
        # The model returns batch.y for GraphGym compatibility but never uses it as input.
        "flows_new": np.zeros(len(old_edges), dtype=np.float64),
        "edge_list_old": old_edges,
        "edge_list_new": old_edges,
        "mutation_type": "identity_template",
        "mutation_info": {"type": "identity_template"},
        "network_name": "EMA",
        "node_ids": tuple(int(value) for value in pair["node_ids"]),
        "centroid_nodes": tuple(int(value) for value in pair["centroid_nodes"]),
        "node_id_offset": int(pair.get("node_id_offset", 1)),
    }


def load_scalers(dataset_dir: Path):
    with (dataset_dir / "scalers" / "attr_scaler.pkl").open("rb") as handle:
        attr_scaler = pickle.load(handle)
    with (dataset_dir / "scalers" / "flow_scaler.pkl").open("rb") as handle:
        flow_scaler = pickle.load(handle)
    return attr_scaler, flow_scaler


def score_candidates(args: argparse.Namespace) -> None:
    process_start = time.perf_counter()
    # Heavy ML imports stay out of CPU-only oracle/summary workers.
    import torch
    from torch_geometric.data import Batch, Data
    from torch_geometric.graphgym.model_builder import create_model

    from create_sioux_data.build_network_pairs_dataset import (
        build_single_data_object,
        extract_edge_attrs,
    )
    from scripts.export_sue_warmstart_predictions import (
        configure,
        resolve_checkpoint,
        state_dict_from_checkpoint,
        torch_load,
    )

    args.input_pkl = args.input_pkl.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    cfg_path = Path(args.cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing inference config: {cfg_path}")
    args.cfg = str(cfg_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_scenarios < 0:
        raise ValueError("--num-scenarios must be 0 (all test scenarios) or positive.")
    marker_values = validate_raw_marker(
        args.validation_marker, args.input_pkl, args.dataset_dir
    )

    checkpoint_path, training_summary = resolve_checkpoint(args.run_dir, args.checkpoint)
    trained_network = str(training_summary.get("network_name", ""))
    if trained_network.lower() != "ema":
        raise ValueError(f"Expected an EMA checkpoint, got {trained_network!r}.")

    split_path = args.dataset_dir / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing EMA split indices: {split_path}")
    dataset_meta_path = args.dataset_dir / "dataset_meta.json"
    if not dataset_meta_path.exists():
        raise FileNotFoundError(f"Missing EMA dataset metadata: {dataset_meta_path}")
    dataset_meta = load_json(dataset_meta_path)
    expected_dataset_values = {
        "network_name": "EMA",
        "num_nodes": 74,
        "num_edges_old": 258,
        "od_dim": 74,
    }
    for key, expected in expected_dataset_values.items():
        if dataset_meta.get(key) != expected:
            raise ValueError(
                f"EMA dataset metadata {key}={dataset_meta.get(key)!r}, expected {expected!r}."
            )
    with np.load(split_path) as split_file:
        test_indices = np.asarray(split_file["test_idx"], dtype=np.int64)
    selected_indices = choose_scenarios(
        test_indices,
        num_scenarios=args.num_scenarios,
        selection_seed=args.selection_seed,
    )
    if selected_indices.size == 0:
        raise RuntimeError("The selected EMA test-scenario set is empty.")

    print(f"Loading raw EMA pairs: {args.input_pkl}")
    pairs = load_pairs(args.input_pkl)
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(pairs)):
        raise IndexError("Selected test indices fall outside the raw pair list.")
    first_pair = pairs[int(selected_indices[0])]
    first_edges, _ = validate_base_pair(first_pair, int(selected_indices[0]))
    first_node_ids = tuple(int(value) for value in first_pair["node_ids"])
    first_centroid_nodes = tuple(int(value) for value in first_pair["centroid_nodes"])
    feasible_edges = feasible_single_edge_removals(first_pair["G"])
    infeasible_edges = [edge for edge in first_edges if edge not in set(feasible_edges)]
    if not feasible_edges:
        raise RuntimeError("No feasible single-edge removal candidates were found.")

    protocol_payload = {
        "task": "braessian-edge-model-ranking-v3-masked-batch",
        "network": "EMA",
        "input_pkl": str(args.input_pkl),
        "dataset_dir": str(args.dataset_dir),
        "input_pkl_sha256": sha256_file(args.input_pkl),
        "validation_marker_sha256": marker_values["marker_sha256"],
        "validation_report_sha256": marker_values["validation_report_sha256"],
        "attr_scaler_sha256": sha256_file(
            args.dataset_dir / "scalers" / "attr_scaler.pkl"
        ),
        "flow_scaler_sha256": sha256_file(
            args.dataset_dir / "scalers" / "flow_scaler.pkl"
        ),
        "split_indices_sha256": sha256_file(split_path),
        "dataset_meta_sha256": sha256_file(dataset_meta_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "inference_config": str(cfg_path),
        "inference_config_sha256": sha256_file(cfg_path),
        "saved_topology_config": training_summary.get("config", {}).get(
            "topology_gnn", {}
        ),
        "selected_pair_indices": selected_indices.tolist(),
        "feasible_edges": [list(edge) for edge in feasible_edges],
        "prediction_flow_policy": args.prediction_flow_policy,
        "selection_seed": int(args.selection_seed),
        "inference_seed": int(args.seed),
        "requested_device": args.device,
        "batch_size": int(args.batch_size),
        "candidate_construction": "cached_edge_mask_batch_templates",
        "score_storage": "atomic_scenario_parts_then_single_merge",
        "random_baseline_rule": "pcg64_seedsequence(selection_seed,scenario_rank,pair_index)",
    }
    signature = short_signature(protocol_payload)
    ranking_path = args.output_dir / "model_rankings.csv"
    metadata_path = args.output_dir / "ranking_metadata.json"
    parts_dir = args.output_dir / "model_ranking_parts"
    timing_parts_dir = args.output_dir / "score_timing_parts"
    timing_path = args.output_dir / "score_scenario_timings.csv"
    parts_dir.mkdir(parents=True, exist_ok=True)
    timing_parts_dir.mkdir(parents=True, exist_ok=True)

    forward_seconds = 0.0
    previous_wall = 0.0
    existing_metadata: dict = {}
    if args.resume and metadata_path.exists():
        existing_metadata = load_json(metadata_path)
        if existing_metadata.get("protocol_signature") != signature:
            raise RuntimeError("Existing model ranking uses another protocol.")
        previous_wall = float(existing_metadata.get("wall_seconds_total", 0.0))
    elif not args.resume and any(parts_dir.glob("scenario_*.csv")):
        raise RuntimeError(
            f"Score part files already exist under {parts_dir}; use --resume or a fresh output dir."
        )

    completed_ranks = set()
    timing_records: dict[int, dict] = {}
    negative_prediction_total = 0
    for scenario_rank, pair_index_value in enumerate(selected_indices, start=1):
        part_path = parts_dir / f"scenario_{scenario_rank:04d}.csv"
        timing_part_path = timing_parts_dir / f"scenario_{scenario_rank:04d}.json"
        if not args.resume or not part_path.exists() or not timing_part_path.exists():
            continue
        rows = read_csv(part_path)
        expected_pair = int(selected_indices[scenario_rank - 1])
        expected_manifest = {
            (candidate_rank, edge[0], edge[1])
            for candidate_rank, edge in enumerate(feasible_edges, start=1)
        }
        actual_manifest = {
            (
                as_int(row, "candidate_rank"),
                as_int(row, "edge_u"),
                as_int(row, "edge_v"),
            )
            for row in rows
        }
        expected_ids = {
            (scenario_rank - 1) * len(feasible_edges) + candidate_rank
            for candidate_rank in range(1, len(feasible_edges) + 1)
        }
        actual_ids = {as_int(row, "row_id") for row in rows}
        stable_rank_by_id = {
            as_int(row, "row_id"): rank
            for rank, row in enumerate(stable_model_order(rows), start=1)
        }
        timing_record = load_json(timing_part_path)
        if (
            len(rows) == len(feasible_edges)
            and len(actual_ids) == len(rows)
            and actual_ids == expected_ids
            and actual_manifest == expected_manifest
            and all(as_int(row, "scenario_rank") == scenario_rank for row in rows)
            and all(as_int(row, "pair_index") == expected_pair for row in rows)
            and all(row.get("protocol_signature") == signature for row in rows)
            and {as_int(row, "model_rank") for row in rows}
            == set(range(1, len(feasible_edges) + 1))
            and all(
                as_int(row, "model_rank")
                == stable_rank_by_id[as_int(row, "row_id")]
                for row in rows
            )
            and as_int(timing_record, "scenario_rank") == scenario_rank
            and as_int(timing_record, "pair_index") == expected_pair
            and as_int(timing_record, "num_candidates") == len(feasible_edges)
            and timing_record.get("protocol_signature") == signature
        ):
            completed_ranks.add(scenario_rank)
            forward_seconds += sum(as_float(row, "model_forward_sec") for row in rows)
            negative_prediction_total += sum(
                as_int(row, "negative_prediction_count") for row in rows
            )
            timing_records[scenario_rank] = timing_record

    if (
        len(completed_ranks) == len(selected_indices)
        and existing_metadata.get("completed", False)
        and existing_metadata.get("final_merge_completed", False)
        and ranking_path.exists()
        and timing_path.exists()
    ):
        final_rows = read_csv(ranking_path)
        final_timing = read_csv(timing_path)
        expected_rows = len(selected_indices) * len(feasible_edges)
        final_ids = [as_int(row, "row_id") for row in final_rows]
        final_by_scenario: dict[int, list[dict]] = defaultdict(list)
        for row in final_rows:
            final_by_scenario[as_int(row, "scenario_rank")].append(row)
        final_is_exact = (
            len(final_rows) == expected_rows
            and len(set(final_ids)) == len(final_ids)
            and set(final_ids) == set(range(1, expected_rows + 1))
            and len({candidate_key(row) for row in final_rows}) == expected_rows
            and all(row.get("protocol_signature") == signature for row in final_rows)
            and len(final_timing) == len(selected_indices)
            and {as_int(row, "scenario_rank") for row in final_timing}
            == set(range(1, len(selected_indices) + 1))
            and all(
                as_int(row, "pair_index")
                == int(selected_indices[as_int(row, "scenario_rank") - 1])
                and as_int(row, "num_candidates") == len(feasible_edges)
                for row in final_timing
            )
            and all(
                as_int(row, "model_rank") == rank
                for rows in final_by_scenario.values()
                for rank, row in enumerate(stable_model_order(rows), start=1)
            )
        )
        if final_is_exact:
            print("Score output is already complete and exactly matches all scenario parts.")
            print(json.dumps(existing_metadata, indent=2))
            return

    attr_scaler, flow_scaler = load_scalers(args.dataset_dir)
    device = configure(args, training_summary)
    model = create_model().to(device)
    checkpoint = torch_load(checkpoint_path)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    inner = model.model if hasattr(model, "model") else model
    flow_mean = float(inner.flow_mean.detach().cpu().item())
    flow_std = float(inner.flow_std.detach().cpu().item())
    scaler_mean = float(np.asarray(flow_scaler.mean_).reshape(-1)[0])
    scaler_std = float(np.asarray(flow_scaler.scale_).reshape(-1)[0])
    if not np.allclose([flow_mean, flow_std], [scaler_mean, scaler_std], rtol=1e-6):
        raise RuntimeError(
            "Checkpoint flow normalization does not match the EMA flow scaler: "
            f"checkpoint=({flow_mean}, {flow_std}), scaler=({scaler_mean}, {scaler_std})."
        )

    initial_metadata = {
        **protocol_payload,
        "protocol_signature": signature,
        "checkpoint": str(checkpoint_path),
        "num_scenarios_requested": int(len(selected_indices)),
        "num_edges_original": int(len(first_edges)),
        "num_feasible_candidates_per_scenario": int(len(feasible_edges)),
        "num_infeasible_edges": int(len(infeasible_edges)),
        "infeasible_edges": [list(edge) for edge in infeasible_edges],
        "batch_size": int(args.batch_size),
        "device": str(device),
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "forward_seconds_total": forward_seconds,
        "wall_seconds_total": previous_wall,
        "num_scenarios_completed": len(completed_ranks),
        "num_candidate_rows": len(completed_ranks) * len(feasible_edges),
        "negative_prediction_count": int(negative_prediction_total),
        "ranking_file": str(ranking_path),
        "ranking_parts_dir": str(parts_dir),
        "scenario_timing_file": str(timing_path),
    }
    save_json_atomic(metadata_path, initial_metadata)

    edge_position = {edge: index for index, edge in enumerate(first_edges)}
    all_edge_positions = np.arange(len(first_edges), dtype=np.int64)
    keep_indices_by_edge = {
        edge: np.delete(all_edge_positions, edge_position[edge])
        for edge in feasible_edges
    }
    template_pair = build_identity_template_pair(first_pair)
    base_data = build_single_data_object(template_pair, attr_scaler, flow_scaler)
    if int(base_data.edge_index_new.shape[1]) != len(first_edges):
        raise RuntimeError("Identity template does not preserve the complete EMA edge list.")

    template_chunks = []
    for chunk_start in range(0, len(feasible_edges), args.batch_size):
        chunk_edges = feasible_edges[chunk_start : chunk_start + args.batch_size]
        keep_matrix = np.stack(
            [keep_indices_by_edge[edge] for edge in chunk_edges], axis=0
        )
        data_list = []
        for keep_indices in keep_matrix:
            keep_tensor = torch.from_numpy(keep_indices).long()
            data_list.append(
                Data(
                    x=base_data.x,
                    edge_index_old=base_data.edge_index_old,
                    edge_attr_old=base_data.edge_attr_old,
                    flow_old=base_data.flow_old,
                    edge_index_new=base_data.edge_index_new.index_select(1, keep_tensor),
                    edge_attr_new=base_data.edge_attr_new.index_select(0, keep_tensor),
                    y=base_data.y.index_select(0, keep_tensor),
                    non_centroid_mask=base_data.non_centroid_mask,
                    net_demand=base_data.net_demand,
                    new_edge_mask=torch.zeros(len(keep_indices), dtype=torch.bool),
                    node_ids=base_data.node_ids,
                    num_nodes=int(base_data.num_nodes),
                    num_edges_old=len(first_edges),
                    num_edges_new=len(keep_indices),
                    centroid_count=int(base_data.centroid_count),
                    od_dim=int(base_data.od_dim),
                    mutation_type="single_edge_removal",
                    network_name="EMA",
                )
            )
        template_batch = Batch.from_data_list(data_list)
        template_batch.split = "test"
        template_batch = template_batch.to(device)
        template_chunks.append(
            {
                "chunk_start": chunk_start,
                "edges": chunk_edges,
                "keep_indices": keep_matrix,
                "keep_flat_device": torch.from_numpy(keep_matrix.reshape(-1))
                .long()
                .to(device),
                "batch": template_batch,
            }
        )

    node_count = int(base_data.num_nodes)
    source_indices = base_data.edge_index_old[0].detach().cpu().numpy()
    target_indices = base_data.edge_index_old[1].detach().cpu().numpy()
    wall_start = process_start
    with torch.inference_mode():
        for scenario_rank, pair_index_value in enumerate(selected_indices, start=1):
            if scenario_rank in completed_ranks:
                print(f"[{scenario_rank}/{len(selected_indices)}] resume skip pair={pair_index_value}")
                continue
            scenario_wall_start = time.perf_counter()
            pair_index = int(pair_index_value)
            pair = pairs[pair_index]
            edge_list, old_flows = validate_base_pair(pair, pair_index)
            if edge_list != first_edges:
                raise RuntimeError(
                    "EMA base topology changed across scenarios; candidate manifest would drift."
                )
            if (
                tuple(int(value) for value in pair["node_ids"]) != first_node_ids
                or tuple(int(value) for value in pair["centroid_nodes"])
                != first_centroid_nodes
            ):
                raise RuntimeError(
                    "EMA node/centroid indexing changed across scenarios; cached templates are unsafe."
                )
            baseline_tstt = total_system_travel_time(pair["G"], old_flows)
            _, original_capacities, original_t0 = graph_arrays(pair["G"])
            safe_original_capacities = np.maximum(original_capacities, 1e-12)
            original_travel_times = bpr_travel_time(
                old_flows, safe_original_capacities, original_t0
            )
            original_link_tstt = old_flows * original_travel_times
            original_marginal_congestion = (
                original_t0
                * 0.15
                * 4.0
                * (np.maximum(old_flows, 0.0) / safe_original_capacities) ** 4
            )
            random_scores = np.random.default_rng(
                np.random.SeedSequence(
                    [int(args.selection_seed), int(scenario_rank), int(pair_index)]
                )
            ).random(len(feasible_edges))
            scenario_rows: list[dict] = []

            tensor_build_start = time.perf_counter()
            raw_attr_old = extract_edge_attrs(pair["G"], edge_list)
            norm_attr_old = attr_scaler.transform(raw_attr_old).astype(np.float32)
            norm_flow_old = flow_scaler.transform(old_flows.reshape(-1, 1)).astype(
                np.float32
            )
            net_demand_np = np.zeros(node_count, dtype=np.float64)
            np.add.at(net_demand_np, target_indices, old_flows)
            np.add.at(net_demand_np, source_indices, -old_flows)
            net_demand_np = net_demand_np.astype(np.float32)
            tensor_build_seconds = time.perf_counter() - tensor_build_start

            h2d_start = time.perf_counter()
            attr_device = torch.from_numpy(norm_attr_old).to(device, non_blocking=True)
            flow_device = torch.from_numpy(norm_flow_old).to(device, non_blocking=True)
            demand_device = torch.from_numpy(net_demand_np).to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - h2d_start
            batch_update_seconds = 0.0
            scenario_forward_seconds = 0.0
            tstt_and_ranking_seconds = 0.0

            for template in template_chunks:
                chunk_start = int(template["chunk_start"])
                chunk_edges = template["edges"]
                keep_matrix = template["keep_indices"]
                batch = template["batch"]
                chunk_size = len(chunk_edges)

                update_start = time.perf_counter()
                batch.edge_attr_old.view(chunk_size, len(first_edges), -1).copy_(
                    attr_device.unsqueeze(0).expand(chunk_size, -1, -1)
                )
                batch.flow_old.view(chunk_size, len(first_edges), -1).copy_(
                    flow_device.unsqueeze(0).expand(chunk_size, -1, -1)
                )
                batch.edge_attr_new.copy_(
                    attr_device.index_select(0, template["keep_flat_device"])
                )
                batch.net_demand.view(chunk_size, node_count).copy_(
                    demand_device.unsqueeze(0).expand(chunk_size, -1)
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                batch_update_seconds += time.perf_counter() - update_start

                forward_start = time.perf_counter()
                pred_scaled, _ = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_forward = time.perf_counter() - forward_start
                forward_seconds += elapsed_forward
                scenario_forward_seconds += elapsed_forward
                pred_real = (
                    pred_scaled * flow_std + flow_mean
                ).view(chunk_size, len(first_edges) - 1).detach().cpu().numpy().astype(
                    np.float64
                )
                if not np.all(np.isfinite(pred_real)):
                    raise FloatingPointError(
                        f"pair={pair_index}: candidate predictions contain NaN/Inf."
                    )

                postprocess_start = time.perf_counter()
                negative_counts = np.count_nonzero(pred_real < 0.0, axis=1)
                predicted_used = (
                    np.maximum(pred_real, 0.0)
                    if args.prediction_flow_policy == "clip_zero"
                    else pred_real
                )
                candidate_capacities = original_capacities[keep_matrix]
                candidate_t0 = original_t0[keep_matrix]
                removed_tstt_values = total_system_travel_time_arrays(
                    predicted_used, candidate_capacities, candidate_t0
                )
                benefit_values = baseline_tstt - removed_tstt_values
                relative_values = benefit_values / max(abs(baseline_tstt), 1e-12)

                for offset, edge in enumerate(chunk_edges):
                    local_index = chunk_start + offset + 1
                    removed_position = edge_position[edge]
                    flow = float(old_flows[removed_position])
                    capacity = float(safe_original_capacities[removed_position])
                    link_tstt = float(original_link_tstt[removed_position])
                    marginal_congestion = float(
                        original_marginal_congestion[removed_position]
                    )
                    row_id = (scenario_rank - 1) * len(feasible_edges) + local_index
                    scenario_rows.append(
                        {
                            "protocol_signature": signature,
                            "row_id": int(row_id),
                            "scenario_rank": int(scenario_rank),
                            "pair_index": int(pair_index),
                            "candidate_rank": int(local_index),
                            "model_rank": 0,
                            "edge_u": int(edge[0]),
                            "edge_v": int(edge[1]),
                            "num_edges_original": int(len(edge_list)),
                            "num_edges_after_removal": int(len(edge_list) - 1),
                            "baseline_tstt": float(baseline_tstt),
                            "model_removed_tstt": float(removed_tstt_values[offset]),
                            "model_benefit": float(benefit_values[offset]),
                            "model_relative_benefit": float(relative_values[offset]),
                            "predicted_total_flow": float(predicted_used[offset].sum()),
                            "negative_prediction_count": int(negative_counts[offset]),
                            "negative_prediction_fraction": float(
                                negative_counts[offset] / max(pred_real.shape[1], 1)
                            ),
                            "old_flow": flow,
                            "vc_ratio": float(flow / capacity),
                            "link_tstt": float(link_tstt),
                            "marginal_congestion": float(marginal_congestion),
                            "score_random": float(random_scores[local_index - 1]),
                            "score_low_flow": -flow,
                            "score_low_vc": float(-flow / capacity),
                            "score_high_flow": flow,
                            "score_high_vc": float(flow / capacity),
                            "score_high_link_tstt": float(link_tstt),
                            "score_high_marginal_congestion": float(
                                marginal_congestion
                            ),
                            "model_forward_sec": float(
                                elapsed_forward / max(chunk_size, 1)
                            ),
                        }
                    )
                tstt_and_ranking_seconds += time.perf_counter() - postprocess_start

            rank_start = time.perf_counter()
            for model_rank, row in enumerate(stable_model_order(scenario_rows), start=1):
                row["model_rank"] = model_rank
            scenario_rows.sort(key=lambda row: as_int(row, "row_id"))
            tstt_and_ranking_seconds += time.perf_counter() - rank_start

            part_path = parts_dir / f"scenario_{scenario_rank:04d}.csv"
            timing_part_path = timing_parts_dir / f"scenario_{scenario_rank:04d}.json"
            serialization_start = time.perf_counter()
            write_csv_atomic(part_path, scenario_rows, SCORE_FIELDS)
            serialization_seconds = time.perf_counter() - serialization_start
            timing_record = {
                "protocol_signature": signature,
                "scenario_rank": int(scenario_rank),
                "pair_index": int(pair_index),
                "num_candidates": int(len(scenario_rows)),
                "tensor_build_sec": float(tensor_build_seconds),
                "batch_update_sec": float(batch_update_seconds),
                "host_to_device_sec": float(h2d_seconds),
                "model_forward_sec": float(scenario_forward_seconds),
                "tstt_and_ranking_sec": float(tstt_and_ranking_seconds),
                "serialization_sec": float(serialization_seconds),
                "scenario_wall_sec": float(time.perf_counter() - scenario_wall_start),
            }
            save_json_atomic(timing_part_path, timing_record)
            timing_records[scenario_rank] = timing_record
            completed_ranks.add(scenario_rank)
            negative_prediction_total += sum(
                as_int(row, "negative_prediction_count") for row in scenario_rows
            )
            elapsed_wall = previous_wall + time.perf_counter() - wall_start
            metadata = {
                **initial_metadata,
                "forward_seconds_total": float(forward_seconds),
                "wall_seconds_total": float(elapsed_wall),
                "num_scenarios_completed": int(len(completed_ranks)),
                "num_candidate_rows": int(len(completed_ranks) * len(feasible_edges)),
                "negative_prediction_count": int(negative_prediction_total),
            }
            for field in SCORE_TIMING_FIELDS[3:]:
                if field in timing_record:
                    metadata[f"{field}_total"] = float(
                        sum(float(record.get(field, 0.0)) for record in timing_records.values())
                    )
            save_json_atomic(metadata_path, metadata)
            print(
                f"[{scenario_rank}/{len(selected_indices)}] pair={pair_index}, "
                f"candidates={len(scenario_rows)}, completed={len(completed_ranks)}"
            )

    expected_rows = len(selected_indices) * len(feasible_edges)
    all_rows = []
    final_timing_rows = []
    for scenario_rank in range(1, len(selected_indices) + 1):
        part_path = parts_dir / f"scenario_{scenario_rank:04d}.csv"
        timing_part_path = timing_parts_dir / f"scenario_{scenario_rank:04d}.json"
        if not part_path.exists() or not timing_part_path.exists():
            raise RuntimeError(f"Missing completed score part for scenario {scenario_rank}.")
        all_rows.extend(read_csv(part_path))
        timing_record = load_json(timing_part_path)
        if (
            timing_record.get("protocol_signature") != signature
            or as_int(timing_record, "scenario_rank") != scenario_rank
            or as_int(timing_record, "pair_index")
            != int(selected_indices[scenario_rank - 1])
            or as_int(timing_record, "num_candidates") != len(feasible_edges)
        ):
            raise RuntimeError(f"Invalid score timing part: {timing_part_path}")
        final_timing_rows.append(timing_record)
    all_rows.sort(key=lambda row: as_int(row, "row_id"))
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Ranking rows={len(all_rows)}, expected={expected_rows}.")
    row_ids = [as_int(row, "row_id") for row in all_rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(
        range(1, expected_rows + 1)
    ):
        raise RuntimeError("Final ranking row IDs are not the exact candidate manifest.")
    if len({candidate_key(row) for row in all_rows}) != expected_rows:
        raise RuntimeError("Final ranking candidate keys are not unique.")
    if any(row.get("protocol_signature") != signature for row in all_rows):
        raise RuntimeError("Final ranking merge contains another score protocol.")
    final_by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
        final_by_scenario[as_int(row, "scenario_rank")].append(row)
    if any(
        as_int(row, "model_rank") != rank
        for rows in final_by_scenario.values()
        for rank, row in enumerate(stable_model_order(rows), start=1)
    ):
        raise RuntimeError("Final model ranks do not match the stable ranking rule.")
    write_csv_atomic(ranking_path, all_rows, SCORE_FIELDS)
    write_csv_atomic(timing_path, final_timing_rows, SCORE_TIMING_FIELDS)
    metadata = load_json(metadata_path)
    metadata["completed"] = True
    metadata["num_scenarios_completed"] = int(len(selected_indices))
    metadata["num_candidate_rows"] = int(len(all_rows))
    metadata["forward_milliseconds_per_candidate"] = float(
        1000.0 * forward_seconds / max(len(all_rows), 1)
    )
    metadata["wall_seconds_total"] = float(previous_wall + time.perf_counter() - wall_start)
    metadata["final_merge_completed"] = True
    save_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


def _init_oracle_worker(
    pairs: list[dict[str, Any]],
    solver_args: dict[str, Any],
    oracle_signature: str,
    score_signature: str,
) -> None:
    global _ORACLE_PAIRS, _ORACLE_ARGS, _ORACLE_SIGNATURE, _SCORE_SIGNATURE
    _ORACLE_PAIRS = pairs
    _ORACLE_ARGS = solver_args
    _ORACLE_SIGNATURE = oracle_signature
    _SCORE_SIGNATURE = score_signature


def _empty_oracle_row(score_row: dict) -> dict:
    return {
        "oracle_protocol_signature": _ORACLE_SIGNATURE,
        "score_protocol_signature": _SCORE_SIGNATURE,
        "row_id": as_int(score_row, "row_id"),
        "scenario_rank": as_int(score_row, "scenario_rank"),
        "pair_index": as_int(score_row, "pair_index"),
        "candidate_rank": as_int(score_row, "candidate_rank"),
        "edge_u": as_int(score_row, "edge_u"),
        "edge_v": as_int(score_row, "edge_v"),
        "evaluation_scope": str(score_row["evaluation_scope"]),
        "model_rank": as_int(score_row, "model_rank"),
        "status": "failed",
        "converged": False,
        "loading_warning": False,
        "used_flow_iter": 0,
        "iterations": 0,
        "network_loading_calls": 0,
        "final_flow_gap": math.nan,
        "final_cost_gap": math.nan,
        "final_update_gap": math.nan,
        "final_convergence_metric": math.nan,
        "elapsed_solver_sec": math.nan,
        "oracle_removed_tstt": math.nan,
        "oracle_benefit": math.nan,
        "oracle_relative_benefit": math.nan,
        "error": "",
    }


def _oracle_scenario(task: tuple[int, int, list[dict]]) -> list[dict]:
    scenario_rank, pair_index, score_rows = task
    assert _ORACLE_PAIRS is not None and _ORACLE_ARGS is not None
    pair = _ORACLE_PAIRS[pair_index]
    _, old_flows = validate_base_pair(pair, pair_index)
    baseline_values = {round(as_float(row, "baseline_tstt"), 8) for row in score_rows}
    if len(baseline_values) != 1:
        raise RuntimeError(f"scenario={scenario_rank}: inconsistent baseline TSTT values.")
    recorded_baseline = as_float(score_rows[0], "baseline_tstt")
    baseline_tstt = total_system_travel_time(pair["G"], old_flows)
    if not np.isclose(baseline_tstt, recorded_baseline, rtol=1e-10, atol=1e-6):
        raise RuntimeError(
            f"scenario={scenario_rank}: raw-pair baseline TSTT changed between stages "
            f"(raw={baseline_tstt}, ranking={recorded_baseline})."
        )
    results = []
    by_row_id = {}
    for score_row in score_rows:
        result = _empty_oracle_row(score_row)
        edge = (as_int(score_row, "edge_u"), as_int(score_row, "edge_v"))
        try:
            graph_removed = pair["G"].copy()
            if not graph_removed.has_edge(*edge):
                raise KeyError(f"Candidate edge {edge} is absent from pair G.")
            graph_removed.remove_edge(*edge)
            if not nx.is_strongly_connected(graph_removed):
                raise ValueError(f"Candidate edge {edge} does not preserve strong connectivity.")
            start = time.perf_counter()
            flows, loading_warning, used_flow_iter, diagnostics = solve_single_graph_sue(
                graph_removed,
                np.asarray(pair["od_matrix"], dtype=np.float64),
                node_ids=pair.get("node_ids"),
                centroid_nodes=pair.get("centroid_nodes"),
                max_iter=_ORACLE_ARGS["max_iter"],
                convergence_threshold=_ORACLE_ARGS["convergence_threshold"],
                theta=_ORACLE_ARGS["theta"],
                value_iter=_ORACLE_ARGS["value_iter"],
                value_tol=_ORACLE_ARGS["value_tol"],
                flow_iter=_ORACLE_ARGS["flow_iter"],
                flow_tol=_ORACLE_ARGS["flow_tol"],
                retry_flow_iter=_ORACLE_ARGS["retry_flow_iter"],
                initial_flows=None,
                return_diagnostics=True,
                loading_protocol=_ORACLE_ARGS["loading_protocol"],
                initial_flow_mode="direct",
                step_rule=_ORACLE_ARGS["step_rule"],
            )
            elapsed = time.perf_counter() - start
            diagnostics.pop("initial_state_flows", None)
            removed_tstt = total_system_travel_time(graph_removed, flows)
            benefit = baseline_tstt - removed_tstt
            result.update(
                {
                    "status": "ok",
                    "converged": bool(diagnostics["converged"]),
                    "loading_warning": bool(loading_warning),
                    "used_flow_iter": int(used_flow_iter),
                    "iterations": int(diagnostics["iterations"]),
                    "network_loading_calls": int(diagnostics["network_loading_calls"]),
                    "final_flow_gap": float(diagnostics["final_flow_gap"]),
                    "final_cost_gap": float(diagnostics["final_cost_gap"]),
                    "final_update_gap": float(diagnostics["final_update_gap"]),
                    "final_convergence_metric": float(
                        diagnostics["final_convergence_metric"]
                    ),
                    "elapsed_solver_sec": float(elapsed),
                    "oracle_removed_tstt": float(removed_tstt),
                    "oracle_benefit": float(benefit),
                    "oracle_relative_benefit": float(
                        benefit / max(abs(baseline_tstt), 1e-12)
                    ),
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        by_row_id[result["row_id"]] = result
    for score_row in score_rows:
        results.append(by_row_id[as_int(score_row, "row_id")])
    return results


def run_oracle(args: argparse.Namespace) -> None:
    args.input_pkl = args.input_pkl.expanduser().resolve()
    args.ranking_csv = args.ranking_csv.expanduser().resolve()
    args.ranking_metadata = args.ranking_metadata.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards.")
    marker_values = validate_raw_marker(args.validation_marker, args.input_pkl)

    score_metadata = load_json(args.ranking_metadata)
    score_signature = str(score_metadata["protocol_signature"])
    if not score_metadata.get("completed", False):
        raise RuntimeError("Model ranking stage is not marked complete.")
    if Path(score_metadata["input_pkl"]).expanduser().resolve() != args.input_pkl:
        raise RuntimeError("Oracle raw-pair path differs from the score-stage protocol.")
    if score_metadata.get("validation_marker_sha256") != marker_values["marker_sha256"]:
        raise RuntimeError("Oracle validation marker differs from the score-stage marker.")
    if (
        score_metadata.get("validation_report_sha256")
        != marker_values["validation_report_sha256"]
    ):
        raise RuntimeError("Oracle raw-pair validation report differs from the score stage.")
    score_rows = read_csv(args.ranking_csv)
    if len(score_rows) != int(score_metadata["num_candidate_rows"]):
        raise RuntimeError("Ranking CSV row count does not match ranking metadata.")
    if any(row.get("protocol_signature") != score_signature for row in score_rows):
        raise RuntimeError("Ranking CSV protocol does not match ranking metadata.")
    score_ids = [as_int(row, "row_id") for row in score_rows]
    if len(set(score_ids)) != len(score_ids):
        raise RuntimeError("Ranking CSV contains duplicate row IDs.")
    if len({candidate_key(row) for row in score_rows}) != len(score_rows):
        raise RuntimeError("Ranking CSV contains duplicate candidate keys.")

    score_by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in score_rows:
        score_by_scenario[as_int(row, "scenario_rank")].append(row)
    expected_candidates = int(score_metadata["num_feasible_candidates_per_scenario"])
    for scenario_rank, rows in score_by_scenario.items():
        stable_rank_by_id = {
            as_int(row, "row_id"): rank
            for rank, row in enumerate(stable_model_order(rows), start=1)
        }
        if (
            len(rows) != expected_candidates
            or {as_int(row, "candidate_rank") for row in rows}
            != set(range(1, expected_candidates + 1))
            or {as_int(row, "model_rank") for row in rows}
            != set(range(1, expected_candidates + 1))
            or not all(
                np.isfinite(as_float(row, "model_relative_benefit")) for row in rows
            )
            or not all(
                as_int(row, "model_rank")
                == stable_rank_by_id[as_int(row, "row_id")]
                for row in rows
            )
        ):
            raise RuntimeError(f"Invalid score manifest for scenario {scenario_rank}.")

    hybrid_rows, audit_ranks, shard_assignment = build_hybrid_manifest(
        score_rows,
        num_audit_scenarios=args.audit_scenarios,
        audit_seed=args.audit_seed,
        final_check_top_k=args.final_check_k,
        num_shards=args.num_shards,
    )
    hybrid_manifest_payload = [
        [
            as_int(row, "row_id"),
            str(row["evaluation_scope"]),
            as_int(row, "model_rank"),
            as_int(row, "hybrid_shard_id"),
        ]
        for row in hybrid_rows
    ]
    hybrid_manifest_signature = hashlib.sha256(
        json.dumps(hybrid_manifest_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected_rows = [
        row
        for row in hybrid_rows
        if as_int(row, "hybrid_shard_id") == args.shard_id
    ]
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in selected_rows:
        grouped[(as_int(row, "scenario_rank"), as_int(row, "pair_index"))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: as_int(row, "candidate_rank"))

    solver_args = {
        "max_iter": int(args.max_iter),
        "convergence_threshold": float(args.convergence_threshold),
        "theta": float(args.theta),
        "value_iter": int(args.value_iter),
        "value_tol": float(args.value_tol),
        "flow_iter": int(args.flow_iter),
        "flow_tol": float(args.flow_tol),
        "retry_flow_iter": int(args.retry_flow_iter),
        "loading_protocol": args.sue_loading_protocol,
        "step_rule": args.step_rule,
    }
    oracle_payload = {
        "task": "braessian-edge-hybrid-sue-verification-v3",
        "score_protocol_signature": score_signature,
        "num_shards": int(args.num_shards),
        "audit_scenarios": int(args.audit_scenarios),
        "audit_seed": int(args.audit_seed),
        "audit_scenario_ranks": audit_ranks,
        "final_check_top_k": int(args.final_check_k),
        "ranking_rule": "model_relative_benefit_desc,candidate_rank_asc,row_id_asc",
        "hybrid_manifest_signature": hybrid_manifest_signature,
        "num_hybrid_candidate_rows": int(len(hybrid_rows)),
        "num_audit_candidate_rows": int(
            sum(row["evaluation_scope"] == "exhaustive_audit" for row in hybrid_rows)
        ),
        "num_deployment_candidate_rows": int(
            sum(row["evaluation_scope"] == "model_top_k" for row in hybrid_rows)
        ),
        "scenario_shard_assignment": {
            str(rank): int(shard) for rank, shard in sorted(shard_assignment.items())
        },
        "input_pkl": str(args.input_pkl),
        "input_pkl_sha256": score_metadata.get("input_pkl_sha256"),
        "validation_marker_sha256": marker_values["marker_sha256"],
        "validation_report_sha256": marker_values["validation_report_sha256"],
        "solver": solver_args,
    }
    oracle_signature = short_signature(oracle_payload)
    shard_path = args.output_dir / f"oracle_shard_{args.shard_id:03d}.csv"
    shard_metadata_path = args.output_dir / f"oracle_shard_{args.shard_id:03d}_metadata.json"

    existing = read_csv(shard_path) if args.resume else []
    if existing and any(
        row.get("oracle_protocol_signature") != oracle_signature for row in existing
    ):
        raise RuntimeError("Existing oracle shard uses another protocol.")
    existing_by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in existing:
        existing_by_scenario[as_int(row, "scenario_rank")].append(row)
    expected_rows_by_scenario = {
        scenario_rank: rows for (scenario_rank, _), rows in grouped.items()
    }
    completed_scenarios = set()
    for scenario_rank, rows in existing_by_scenario.items():
        expected = expected_rows_by_scenario.get(scenario_rank, [])
        expected_by_id = {as_int(row, "row_id"): row for row in expected}
        expected_ids = {as_int(row, "row_id") for row in expected}
        actual_ids = [as_int(row, "row_id") for row in rows]
        if (
            len(rows) == len(expected)
            and len(set(actual_ids)) == len(actual_ids)
            and set(actual_ids) == expected_ids
            and {candidate_key(row) for row in rows}
            == {candidate_key(row) for row in expected}
            and all(
                str(row.get("evaluation_scope"))
                == str(expected_by_id[as_int(row, "row_id")]["evaluation_scope"])
                and as_int(row, "model_rank")
                == as_int(expected_by_id[as_int(row, "row_id")], "model_rank")
                for row in rows
            )
            and all(row.get("status") == "ok" for row in rows)
        ):
            completed_scenarios.add(scenario_rank)
    kept_existing = [
        row for row in existing if as_int(row, "scenario_rank") in completed_scenarios
    ]
    tasks = [
        (scenario_rank, pair_index, rows)
        for (scenario_rank, pair_index), rows in sorted(grouped.items())
        if scenario_rank not in completed_scenarios
    ]

    print(f"Loading raw pairs for oracle: {args.input_pkl}")
    pairs = load_pairs(args.input_pkl)
    output_rows = list(kept_existing)
    metadata = {
        **oracle_payload,
        "oracle_protocol_signature": oracle_signature,
        "shard_id": int(args.shard_id),
        "num_scenarios_in_shard": int(len(grouped)),
        "num_candidates_in_shard": int(len(selected_rows)),
        "num_audit_scenarios_in_shard": int(
            sum(scenario_rank in set(audit_ranks) for scenario_rank in grouped)
        ),
        "num_deployment_scenarios_in_shard": int(
            sum(scenario_rank not in set(audit_ranks) for scenario_rank in grouped)
        ),
        "num_audit_candidates_in_shard": int(
            sum(row["evaluation_scope"] == "exhaustive_audit" for row in selected_rows)
        ),
        "num_deployment_candidates_in_shard": int(
            sum(row["evaluation_scope"] == "model_top_k" for row in selected_rows)
        ),
        "num_workers": int(args.num_workers),
        "num_scenarios_completed": int(len(completed_scenarios)),
        "num_rows_completed": int(len(output_rows)),
        "completed": False,
        "output_csv": str(shard_path),
    }
    save_json_atomic(shard_metadata_path, metadata)

    _init_oracle_worker(pairs, solver_args, oracle_signature, score_signature)

    def consume(scenario_rows: list[dict]) -> None:
        nonlocal output_rows
        scenario_rank = as_int(scenario_rows[0], "scenario_rank")
        output_rows = [
            row for row in output_rows if as_int(row, "scenario_rank") != scenario_rank
        ]
        output_rows.extend(scenario_rows)
        output_rows.sort(key=lambda row: as_int(row, "row_id"))
        write_csv_atomic(shard_path, output_rows, ORACLE_FIELDS)
        metadata["num_scenarios_completed"] = len(
            {as_int(row, "scenario_rank") for row in output_rows}
        )
        metadata["num_rows_completed"] = len(output_rows)
        metadata["num_failed"] = sum(row["status"] != "ok" for row in output_rows)
        metadata["num_unconverged"] = sum(
            row["status"] == "ok" and not as_bool(row["converged"])
            for row in output_rows
        )
        save_json_atomic(shard_metadata_path, metadata)

    if args.num_workers <= 1:
        for task_index, task in enumerate(tasks, start=1):
            consume(_oracle_scenario(task))
            print(
                f"[{task_index}/{len(tasks)}] shard={args.shard_id}, "
                f"scenario={task[0]}, rows={len(output_rows)}"
            )
    elif tasks:
        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=context,
            initializer=_init_oracle_worker,
            initargs=(pairs, solver_args, oracle_signature, score_signature),
        ) as executor:
            futures = {executor.submit(_oracle_scenario, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                consume(future.result())
                print(
                    f"[{completed}/{len(tasks)}] shard={args.shard_id}, "
                    f"scenario={task[0]}, rows={len(output_rows)}"
                )

    if len(output_rows) != len(selected_rows):
        raise RuntimeError(
            f"Oracle shard rows={len(output_rows)}, expected={len(selected_rows)}."
        )
    expected_ids = {as_int(row, "row_id") for row in selected_rows}
    output_ids = [as_int(row, "row_id") for row in output_rows]
    if len(set(output_ids)) != len(output_ids) or set(output_ids) != expected_ids:
        raise RuntimeError("Final oracle shard row IDs do not match its score manifest.")
    if {candidate_key(row) for row in output_rows} != {
        candidate_key(row) for row in selected_rows
    }:
        raise RuntimeError("Final oracle shard candidate keys do not match its score manifest.")
    selected_by_id = {as_int(row, "row_id"): row for row in selected_rows}
    if any(
        str(row.get("evaluation_scope"))
        != str(selected_by_id[as_int(row, "row_id")]["evaluation_scope"])
        or as_int(row, "model_rank")
        != as_int(selected_by_id[as_int(row, "row_id")], "model_rank")
        for row in output_rows
    ):
        raise RuntimeError("Final oracle shard scope/model ranks do not match the manifest.")
    num_failed = int(sum(row["status"] != "ok" for row in output_rows))
    metadata["completed"] = num_failed == 0
    metadata["num_scenarios_completed"] = int(len(grouped))
    metadata["num_rows_completed"] = int(len(output_rows))
    metadata["num_failed"] = num_failed
    metadata["num_unconverged"] = int(
        sum(
            row["status"] == "ok" and not as_bool(row["converged"])
            for row in output_rows
        )
    )
    save_json_atomic(shard_metadata_path, metadata)
    if num_failed:
        raise RuntimeError(
            f"Oracle shard {args.shard_id} has {num_failed} failed candidates. "
            "Re-submit with --resume; failed scenarios will be recomputed."
        )
    print(json.dumps(metadata, indent=2))


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def rank_correlations(score: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    from scipy.stats import kendalltau, spearmanr

    if score.size < 2 or np.all(score == score[0]) or np.all(truth == truth[0]):
        return math.nan, math.nan
    return float(spearmanr(score, truth).statistic), float(kendalltau(score, truth).statistic)


def ndcg_at_k(score: np.ndarray, truth: np.ndarray, k: int) -> float:
    relevance = np.maximum(np.asarray(truth, dtype=np.float64), 0.0)
    if relevance.size == 0 or relevance.max() <= 0.0:
        return math.nan
    k = min(int(k), relevance.size)
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    predicted_order = np.argsort(-score, kind="mergesort")[:k]
    ideal_order = np.argsort(-relevance, kind="mergesort")[:k]
    dcg = float(np.sum(relevance[predicted_order] * discount))
    ideal = float(np.sum(relevance[ideal_order] * discount))
    return dcg / ideal if ideal > 0.0 else math.nan


def average_precision(score: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def scenario_method_metrics(
    rows: list[dict],
    method: str,
    score_key: str,
    top_k: list[int],
    threshold: float,
) -> dict:
    truth = np.asarray([row["oracle_relative_benefit"] for row in rows], dtype=np.float64)
    score = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    labels = truth > threshold
    order = np.argsort(-score, kind="mergesort")
    oracle_order = np.argsort(-truth, kind="mergesort")
    best_raw_positive = max(0.0, float(np.max(truth)))
    best_meaningful = float(np.max(truth[labels])) if labels.any() else math.nan
    spearman, kendall = rank_correlations(score, truth)
    result = {
        "scenario_rank": int(rows[0]["scenario_rank"]),
        "pair_index": int(rows[0]["pair_index"]),
        "method": method,
        "num_candidates": int(len(rows)),
        "num_braessian": int(labels.sum()),
        "braessian_prevalence": float(labels.mean()),
        "has_braessian": bool(labels.any()),
        "spearman": spearman,
        "kendall": kendall,
        "average_precision": average_precision(score, labels),
        "benefit_mae": float(np.mean(np.abs(score - truth)))
        if method == "edge_transformer"
        else math.nan,
        "best_oracle_relative_benefit": best_raw_positive,
        "best_meaningful_relative_benefit": best_meaningful,
    }
    for k_value in top_k:
        k = min(k_value, len(rows))
        selected = order[:k]
        selected_positive = int(labels[selected].sum())
        best_selected_raw_positive = max(0.0, float(np.max(truth[selected])))
        selected_meaningful_values = truth[selected][labels[selected]]
        best_selected_meaningful = (
            float(np.max(selected_meaningful_values))
            if selected_meaningful_values.size
            else 0.0
        )
        oracle_top = set(oracle_order[:k].tolist())
        predicted_top = set(selected.tolist())
        result[f"ndcg_at_{k_value}"] = ndcg_at_k(score, truth, k)
        result[f"topk_overlap_at_{k_value}"] = float(
            len(oracle_top & predicted_top) / max(k, 1)
        )
        result[f"precision_at_{k_value}"] = float(selected_positive / max(k, 1))
        result[f"recall_at_{k_value}"] = (
            float(selected_positive / labels.sum()) if labels.any() else math.nan
        )
        result[f"success_at_{k_value}"] = float(selected_positive > 0)
        result[f"success_conditional_at_{k_value}"] = (
            float(selected_positive > 0) if labels.any() else math.nan
        )
        result[f"oracle_best_hit_at_{k_value}"] = (
            float(int(oracle_order[0]) in predicted_top) if labels.any() else math.nan
        )
        result[f"benefit_recovery_at_{k_value}"] = (
            float(best_selected_raw_positive / best_raw_positive)
            if best_raw_positive > 0.0
            else math.nan
        )
        result[f"raw_positive_benefit_recovery_at_{k_value}"] = result[
            f"benefit_recovery_at_{k_value}"
        ]
        result[f"meaningful_benefit_recovery_at_{k_value}"] = (
            float(best_selected_meaningful / best_meaningful)
            if labels.any() and best_meaningful > 0.0
            else math.nan
        )
        result[f"regret_at_{k_value}"] = float(
            best_raw_positive - best_selected_raw_positive
        )
        result[f"meaningful_regret_at_{k_value}"] = (
            float(best_meaningful - best_selected_meaningful)
            if labels.any()
            else math.nan
        )
    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    num_bootstrap: int,
) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(values.mean())
    if values.size == 1 or num_bootstrap <= 0:
        return mean, math.nan, math.nan, int(values.size)
    draws = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high), int(values.size)


def collect_oracle_rows(oracle_dir: Path) -> tuple[list[dict[str, str]], list[dict]]:
    csv_paths = sorted(oracle_dir.glob("oracle_shard_*.csv"))
    csv_paths = [path for path in csv_paths if not path.name.endswith("_metadata.csv")]
    if not csv_paths:
        raise FileNotFoundError(f"No oracle_shard_*.csv files found under {oracle_dir}")
    rows = []
    for path in csv_paths:
        rows.extend(read_csv(path))
    metadata = [
        load_json(path) for path in sorted(oracle_dir.glob("oracle_shard_*_metadata.json"))
    ]
    return rows, metadata


def make_plots(
    output_dir: Path,
    merged_rows: list[dict],
    aggregate_lookup: dict[tuple[str, str], float],
    methods: list[str],
    top_k: list[int],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: {exc}")
        return []

    figures = []
    labels = [method.replace("_", "\n") for method in methods]
    spearman = [aggregate_lookup.get((method, "spearman"), math.nan) for method in methods]
    ndcg10_key = f"ndcg_at_{10 if 10 in top_k else top_k[-1]}"
    ndcg = [aggregate_lookup.get((method, ndcg10_key), math.nan) for method in methods]
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, spearman, width, label="Spearman")
    ax.bar(
        x + width / 2,
        ndcg,
        width,
        label=f"NDCG@{ndcg10_key.rsplit('_', 1)[-1]}",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=min(0.0, np.nanmin(spearman) - 0.05), top=1.05)
    ax.set_ylabel("Scenario-macro score")
    ax.set_title("Agreement with SUE ranking on exhaustive audit scenarios")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "ranking_quality.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        recovery = [
            aggregate_lookup.get((method, f"benefit_recovery_at_{k}"), math.nan)
            for k in top_k
        ]
        ax.plot(top_k, recovery, marker="o", label=method.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xticks(top_k, [str(value) for value in top_k])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("SUE final checks after model/heuristic screening (k)")
    ax.set_ylabel("Oracle benefit recovery")
    ax.set_title("Planning benefit versus verification budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "screening_tradeoff.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figures.append(str(path))

    model_x = np.asarray(
        [row["model_relative_benefit"] for row in merged_rows], dtype=np.float64
    )
    oracle_y = np.asarray(
        [row["oracle_relative_benefit"] for row in merged_rows], dtype=np.float64
    )
    if model_x.size > 20000:
        selected = np.random.default_rng(2026).choice(model_x.size, 20000, replace=False)
        model_x, oracle_y = model_x[selected], oracle_y[selected]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(100.0 * model_x, 100.0 * oracle_y, s=6, alpha=0.18, linewidths=0)
    limits = [
        min(float(np.min(100.0 * model_x)), float(np.min(100.0 * oracle_y))),
        max(float(np.max(100.0 * model_x)), float(np.max(100.0 * oracle_y))),
    ]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Predicted TSTT reduction (%)")
    ax.set_ylabel("SUE TSTT reduction (%)")
    ax.set_title("Edge-removal benefit on exhaustive audit scenarios")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "model_vs_oracle_benefit.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figures.append(str(path))
    return figures


def final_check_scenario_metrics(
    score_rows: list[dict],
    evaluated_by_id: dict[int, dict],
    top_k: list[int],
    threshold: float,
    evaluation_scope: str,
) -> dict:
    """Application metrics available without knowing the unevaluated SUE ranking."""
    ordered = stable_model_order(score_rows)
    result = {
        "scenario_rank": as_int(ordered[0], "scenario_rank"),
        "pair_index": as_int(ordered[0], "pair_index"),
        "evaluation_scope": evaluation_scope,
    }
    for k in top_k:
        selected_score_rows = ordered[:k]
        selected_ids = [as_int(row, "row_id") for row in selected_score_rows]
        missing = [row_id for row_id in selected_ids if row_id not in evaluated_by_id]
        if missing:
            raise RuntimeError(
                f"Scenario {result['scenario_rank']} lacks SUE final checks for "
                f"model top-{k}: {missing[:10]}."
            )
        selected = [evaluated_by_id[row_id] for row_id in selected_ids]
        truth = np.asarray(
            [row["oracle_relative_benefit"] for row in selected], dtype=np.float64
        )
        best_index = int(np.argmax(truth))
        best = selected[best_index]
        best_relative = float(truth[best_index])
        best_absolute = float(best["oracle_benefit"])
        raw_positive = best_relative > 0.0
        meaningful = best_relative > threshold
        solver_seconds = float(sum(row["elapsed_solver_sec"] for row in selected))
        meaningful_positions = np.flatnonzero(truth > threshold)
        if meaningful_positions.size:
            first_success_index = int(meaningful_positions[0])
            calls_to_first_success = first_success_index + 1
            adaptive_selected = selected[first_success_index]
            adaptive_relative = float(
                adaptive_selected["oracle_relative_benefit"]
            )
            adaptive_absolute = float(adaptive_selected["oracle_benefit"])
        else:
            first_success_index = len(selected) - 1
            calls_to_first_success = math.nan
            adaptive_selected = None
            adaptive_relative = 0.0
            adaptive_absolute = 0.0
        adaptive_calls = (
            int(calls_to_first_success) if np.isfinite(calls_to_first_success) else len(selected)
        )
        adaptive_seconds = float(
            sum(
                row["elapsed_solver_sec"]
                for row in selected[: first_success_index + 1]
            )
        )
        result.update(
            {
                f"sue_calls_at_{k}": int(len(selected)),
                f"sue_seconds_at_{k}": solver_seconds,
                f"raw_positive_success_at_{k}": float(raw_positive),
                f"meaningful_success_at_{k}": float(meaningful),
                f"no_action_at_{k}": float(not meaningful),
                f"verified_best_relative_benefit_at_{k}": best_relative,
                f"verified_best_absolute_benefit_at_{k}": best_absolute,
                f"implemented_relative_benefit_at_{k}": (
                    best_relative if meaningful else 0.0
                ),
                f"implemented_absolute_benefit_at_{k}": (
                    best_absolute if meaningful else 0.0
                ),
                f"selected_row_id_at_{k}": int(best["row_id"]) if meaningful else 0,
                f"selected_edge_u_at_{k}": int(best["edge_u"]) if meaningful else -1,
                f"selected_edge_v_at_{k}": int(best["edge_v"]) if meaningful else -1,
                f"calls_to_first_success_at_{k}": calls_to_first_success,
                f"adaptive_capped_calls_at_{k}": adaptive_calls,
                f"adaptive_sue_seconds_at_{k}": adaptive_seconds,
                f"adaptive_stopped_early_at_{k}": float(
                    meaningful and adaptive_calls < len(selected)
                ),
                f"adaptive_implemented_relative_benefit_at_{k}": adaptive_relative,
                f"adaptive_implemented_absolute_benefit_at_{k}": adaptive_absolute,
                f"adaptive_selected_row_id_at_{k}": (
                    int(adaptive_selected["row_id"]) if adaptive_selected else 0
                ),
            }
        )
    return result


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if array.size else math.nan


def summarize(args: argparse.Namespace) -> None:
    args.ranking_csv = args.ranking_csv.expanduser().resolve()
    args.ranking_metadata = args.ranking_metadata.expanduser().resolve()
    args.oracle_dir = args.oracle_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    top_k = sorted({int(value) for value in args.top_k.split(",") if int(value) > 0})
    if not top_k:
        raise ValueError("--top-k must contain at least one positive integer.")
    if args.num_bootstrap < 0:
        raise ValueError("--num-bootstrap cannot be negative.")
    thresholds = sorted(
        {float(value) for value in args.braess_thresholds.split(",") if value.strip()}
    )
    primary_threshold = float(args.primary_braess_threshold)
    if primary_threshold < 0.0 or any(value < 0.0 for value in thresholds):
        raise ValueError("Braessian benefit thresholds must be non-negative.")
    if primary_threshold not in thresholds:
        thresholds.append(primary_threshold)
        thresholds.sort()

    ranking_metadata = load_json(args.ranking_metadata)
    score_signature = str(ranking_metadata["protocol_signature"])
    score_rows = read_csv(args.ranking_csv)
    if not ranking_metadata.get("completed", False):
        raise RuntimeError("Model ranking metadata is not marked complete.")
    if len(score_rows) != int(ranking_metadata["num_candidate_rows"]):
        raise RuntimeError("Ranking CSV row count does not match ranking metadata.")
    if any(row.get("protocol_signature") != score_signature for row in score_rows):
        raise RuntimeError("Ranking CSV contains a mismatched protocol.")
    score_ids = [as_int(row, "row_id") for row in score_rows]
    if len(set(score_ids)) != len(score_ids):
        raise RuntimeError("Ranking CSV contains duplicate row IDs.")
    if len({candidate_key(row) for row in score_rows}) != len(score_rows):
        raise RuntimeError("Ranking CSV contains duplicate candidate keys.")

    candidates_per_scenario = int(
        ranking_metadata["num_feasible_candidates_per_scenario"]
    )
    requested_scenarios = int(ranking_metadata["num_scenarios_requested"])
    score_by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in score_rows:
        score_by_scenario[as_int(row, "scenario_rank")].append(row)
    if len(score_by_scenario) != requested_scenarios:
        raise RuntimeError(
            f"Score scenarios={len(score_by_scenario)}, expected={requested_scenarios}."
        )
    for scenario_rank, rows in score_by_scenario.items():
        pair_indices = {as_int(row, "pair_index") for row in rows}
        ordered = stable_model_order(rows)
        model_ranks_match = all(
            as_int(row, "model_rank") == expected_rank
            for expected_rank, row in enumerate(ordered, start=1)
        )
        if (
            len(rows) != candidates_per_scenario
            or len(pair_indices) != 1
            or {as_int(row, "candidate_rank") for row in rows}
            != set(range(1, candidates_per_scenario + 1))
            or {as_int(row, "model_rank") for row in rows}
            != set(range(1, candidates_per_scenario + 1))
            or not model_ranks_match
            or not all(
                np.isfinite(as_float(row, "model_relative_benefit")) for row in rows
            )
        ):
            raise RuntimeError(f"Invalid score manifest for scenario {scenario_rank}.")

    oracle_rows, oracle_metadata = collect_oracle_rows(args.oracle_dir)
    if not oracle_metadata:
        raise RuntimeError("No oracle shard metadata files were found.")
    oracle_metadata.sort(key=lambda item: int(item["shard_id"]))
    policy_keys = (
        "task",
        "score_protocol_signature",
        "num_shards",
        "audit_scenarios",
        "audit_seed",
        "audit_scenario_ranks",
        "final_check_top_k",
        "ranking_rule",
        "hybrid_manifest_signature",
        "num_hybrid_candidate_rows",
        "num_audit_candidate_rows",
        "num_deployment_candidate_rows",
        "scenario_shard_assignment",
        "oracle_protocol_signature",
    )
    for key in policy_keys:
        values = {
            json.dumps(item.get(key), sort_keys=True, separators=(",", ":"))
            for item in oracle_metadata
        }
        if len(values) != 1:
            raise RuntimeError(f"Oracle shard metadata disagree on {key}: {values}")
    policy = oracle_metadata[0]
    if policy.get("task") != "braessian-edge-hybrid-sue-verification-v3":
        raise RuntimeError(
            "Summarize requires the hybrid SUE verification protocol, not the old "
            "all-candidate oracle."
        )
    if str(policy["score_protocol_signature"]) != score_signature:
        raise RuntimeError("Oracle metadata do not correspond to this score protocol.")

    declared_num_shards = int(policy["num_shards"])
    shard_ids = [int(item["shard_id"]) for item in oracle_metadata]
    shard_metadata_complete = (
        len(set(shard_ids)) == len(shard_ids)
        and set(shard_ids) == set(range(declared_num_shards))
        and all(item.get("completed", False) for item in oracle_metadata)
    )
    if not shard_metadata_complete:
        message = (
            "Hybrid SUE shard metadata are incomplete or duplicated: "
            f"found={sorted(shard_ids)}, expected=0..{declared_num_shards - 1}."
        )
        if not args.allow_incomplete:
            raise RuntimeError(message)
        print("WARNING:", message)

    final_check_top_k = int(policy["final_check_top_k"])
    if max(top_k) > final_check_top_k:
        raise ValueError(
            f"Requested top-k={max(top_k)} exceeds the deployment SUE budget "
            f"final_check_top_k={final_check_top_k}. Partial deployment oracle rows "
            "cannot support that metric."
        )
    hybrid_rows, audit_ranks, shard_assignment = build_hybrid_manifest(
        score_rows,
        num_audit_scenarios=int(policy["audit_scenarios"]),
        audit_seed=int(policy["audit_seed"]),
        final_check_top_k=final_check_top_k,
        num_shards=declared_num_shards,
    )
    hybrid_manifest_payload = [
        [
            as_int(row, "row_id"),
            str(row["evaluation_scope"]),
            as_int(row, "model_rank"),
            as_int(row, "hybrid_shard_id"),
        ]
        for row in hybrid_rows
    ]
    hybrid_manifest_signature = hashlib.sha256(
        json.dumps(hybrid_manifest_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_assignment = {
        str(rank): int(shard) for rank, shard in sorted(shard_assignment.items())
    }
    if audit_ranks != [int(value) for value in policy["audit_scenario_ranks"]]:
        raise RuntimeError("Reconstructed audit scenario ranks differ from oracle metadata.")
    if expected_assignment != {
        str(key): int(value)
        for key, value in policy["scenario_shard_assignment"].items()
    }:
        raise RuntimeError("Reconstructed hybrid shard assignment differs from metadata.")
    if hybrid_manifest_signature != str(policy["hybrid_manifest_signature"]):
        raise RuntimeError("Reconstructed hybrid manifest signature differs from metadata.")
    if len(hybrid_rows) != int(policy["num_hybrid_candidate_rows"]):
        raise RuntimeError("Reconstructed hybrid manifest row count differs from metadata.")
    expected_audit_rows = sum(
        row["evaluation_scope"] == "exhaustive_audit" for row in hybrid_rows
    )
    expected_deployment_rows = len(hybrid_rows) - expected_audit_rows
    if (
        expected_audit_rows != int(policy["num_audit_candidate_rows"])
        or expected_deployment_rows != int(policy["num_deployment_candidate_rows"])
    ):
        raise RuntimeError("Hybrid audit/deployment row counts differ from metadata.")
    for item in oracle_metadata:
        shard_id = int(item["shard_id"])
        shard_rows = [
            row
            for row in hybrid_rows
            if as_int(row, "hybrid_shard_id") == shard_id
        ]
        shard_scenarios = {as_int(row, "scenario_rank") for row in shard_rows}
        if int(item["num_candidates_in_shard"]) != len(shard_rows):
            raise RuntimeError(f"Shard {shard_id} candidate count differs from manifest.")
        if int(item["num_scenarios_in_shard"]) != len(shard_scenarios):
            raise RuntimeError(f"Shard {shard_id} scenario count differs from manifest.")

    oracle_signatures = {row.get("oracle_protocol_signature") for row in oracle_rows}
    if len(oracle_signatures) != 1:
        raise RuntimeError(f"Found multiple oracle protocols: {oracle_signatures}")
    metadata_oracle_signatures = {
        item.get("oracle_protocol_signature") for item in oracle_metadata
    }
    if metadata_oracle_signatures != oracle_signatures:
        raise RuntimeError(
            "Oracle CSV and metadata signatures differ: "
            f"rows={oracle_signatures}, metadata={metadata_oracle_signatures}."
        )
    if any(row.get("score_protocol_signature") != score_signature for row in oracle_rows):
        raise RuntimeError("Oracle rows do not correspond to the supplied model ranking.")

    expected_by_id = {as_int(row, "row_id"): row for row in hybrid_rows}
    oracle_by_id: dict[int, dict] = {}
    duplicate_ids = []
    for row in oracle_rows:
        row_id = as_int(row, "row_id")
        if row_id in oracle_by_id:
            duplicate_ids.append(row_id)
        oracle_by_id[row_id] = row
    expected_ids = set(expected_by_id)
    actual_ids = set(oracle_by_id)
    extra = sorted(actual_ids - expected_ids)
    missing = sorted(expected_ids - actual_ids)
    if duplicate_ids or extra:
        raise RuntimeError(
            f"Hybrid oracle join is invalid: duplicates={duplicate_ids[:10]}, "
            f"extra={extra[:10]}."
        )
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Hybrid oracle is missing {len(missing)} manifest rows; "
            f"examples={missing[:10]}."
        )

    merged = []
    failed = []
    unconverged = []
    oracle_ok_count = 0
    for row_id in sorted(expected_ids & actual_ids):
        expected = expected_by_id[row_id]
        oracle_row = oracle_by_id[row_id]
        mismatched_keys = [
            key
            for key in CANDIDATE_KEY_FIELDS
            if as_int(expected, key) != as_int(oracle_row, key)
        ]
        if mismatched_keys:
            raise RuntimeError(
                f"Oracle row_id={row_id} mismatches manifest fields: {mismatched_keys}."
            )
        if (
            str(oracle_row.get("evaluation_scope"))
            != str(expected["evaluation_scope"])
            or as_int(oracle_row, "model_rank") != as_int(expected, "model_rank")
        ):
            raise RuntimeError(
                f"Oracle row_id={row_id} has the wrong evaluation scope/model rank."
            )
        if oracle_row["status"] != "ok":
            failed.append(row_id)
            continue
        oracle_ok_count += 1
        converged = as_bool(oracle_row["converged"])
        if not converged:
            unconverged.append(row_id)
            if args.require_converged:
                continue
        item = {
            "row_id": row_id,
            "scenario_rank": as_int(expected, "scenario_rank"),
            "pair_index": as_int(expected, "pair_index"),
            "candidate_rank": as_int(expected, "candidate_rank"),
            "model_rank": as_int(expected, "model_rank"),
            "edge_u": as_int(expected, "edge_u"),
            "edge_v": as_int(expected, "edge_v"),
            "evaluation_scope": str(expected["evaluation_scope"]),
        }
        for key in (
            "baseline_tstt",
            "model_removed_tstt",
            "model_benefit",
            "model_relative_benefit",
            "score_random",
            "score_low_flow",
            "score_low_vc",
            "score_high_flow",
            "score_high_vc",
            "score_high_link_tstt",
            "score_high_marginal_congestion",
            "model_forward_sec",
        ):
            item[key] = as_float(expected, key)
        for key in (
            "oracle_removed_tstt",
            "oracle_benefit",
            "oracle_relative_benefit",
            "elapsed_solver_sec",
        ):
            item[key] = as_float(oracle_row, key)
        item["converged"] = converged
        merged.append(item)

    if failed and not args.allow_incomplete:
        raise RuntimeError(
            f"Hybrid SUE verification has {len(failed)} failed rows; "
            f"examples={failed[:10]}."
        )
    if args.require_converged and unconverged and not args.allow_incomplete:
        raise RuntimeError(
            f"--require-converged removed {len(unconverged)} rows. Use "
            "--allow-incomplete only for a clearly labelled diagnostic summary."
        )

    usable_by_id = {int(row["row_id"]): row for row in merged}
    expected_by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in hybrid_rows:
        expected_by_scenario[as_int(row, "scenario_rank")].append(row)
    usable_ids = set(usable_by_id)
    complete_ranks = []
    incomplete_ranks = []
    for scenario_rank in sorted(score_by_scenario):
        expected_scenario_ids = {
            as_int(row, "row_id") for row in expected_by_scenario[scenario_rank]
        }
        if expected_scenario_ids <= usable_ids:
            complete_ranks.append(scenario_rank)
        else:
            incomplete_ranks.append(scenario_rank)
    if incomplete_ranks and not args.allow_incomplete:
        raise RuntimeError(
            f"Hybrid verification is incomplete for {len(incomplete_ranks)} scenarios; "
            f"examples={incomplete_ranks[:10]}."
        )
    if not complete_ranks:
        raise RuntimeError("No complete hybrid-verification scenarios are available.")

    audit_set = set(audit_ranks)
    audit_scenarios: dict[int, list[dict]] = {}
    for scenario_rank in audit_ranks:
        if scenario_rank not in complete_ranks:
            continue
        rows = [
            usable_by_id[as_int(row, "row_id")]
            for row in expected_by_scenario[scenario_rank]
        ]
        if (
            len(rows) != candidates_per_scenario
            or {row["candidate_rank"] for row in rows}
            != set(range(1, candidates_per_scenario + 1))
            or {row["evaluation_scope"] for row in rows} != {"exhaustive_audit"}
        ):
            raise RuntimeError(
                f"Audit scenario {scenario_rank} is not an exhaustive candidate set."
            )
        audit_scenarios[scenario_rank] = sorted(
            rows, key=lambda row: row["candidate_rank"]
        )
    if not audit_scenarios:
        raise RuntimeError("No complete exhaustive audit scenario is available.")
    if len(audit_scenarios) != len(audit_ranks) and not args.allow_incomplete:
        raise RuntimeError("Not every registered audit scenario is complete.")

    timing_path = Path(
        ranking_metadata.get(
            "scenario_timing_file",
            args.ranking_csv.parent / "score_scenario_timings.csv",
        )
    ).expanduser()
    if not timing_path.exists():
        timing_path = args.ranking_csv.parent / "score_scenario_timings.csv"
    timing_by_scenario: dict[int, dict] = {}
    if timing_path.exists():
        for row in read_csv(timing_path):
            scenario_rank = as_int(row, "scenario_rank")
            if scenario_rank in timing_by_scenario:
                raise RuntimeError(
                    f"Score timing file duplicates scenario {scenario_rank}."
                )
            timing_by_scenario[scenario_rank] = row
        if set(timing_by_scenario) != set(score_by_scenario):
            raise RuntimeError(
                "Score timing scenarios do not exactly match the ranking manifest."
            )
        for scenario_rank, row in timing_by_scenario.items():
            if as_int(row, "num_candidates") != candidates_per_scenario:
                raise RuntimeError(
                    f"Score timing candidate count is wrong for scenario {scenario_rank}."
                )
            expected_pair = as_int(score_by_scenario[scenario_rank][0], "pair_index")
            if as_int(row, "pair_index") != expected_pair:
                raise RuntimeError(
                    f"Score timing pair index is wrong for scenario {scenario_rank}."
                )
    ranking_wall_seconds_total = float(ranking_metadata["wall_seconds_total"])
    fallback_wall_per_scenario = ranking_wall_seconds_total / max(
        requested_scenarios, 1
    )

    application_rows = []
    for scenario_rank in complete_ranks:
        expected_scope = {
            str(row["evaluation_scope"])
            for row in expected_by_scenario[scenario_rank]
        }
        if len(expected_scope) != 1:
            raise RuntimeError(f"Scenario {scenario_rank} mixes evaluation scopes.")
        application = final_check_scenario_metrics(
            score_by_scenario[scenario_rank],
            evaluated_by_id=usable_by_id,
            top_k=top_k,
            threshold=primary_threshold,
            evaluation_scope=next(iter(expected_scope)),
        )
        timing = timing_by_scenario.get(scenario_rank)
        if timing is not None:
            if as_int(timing, "pair_index") != application["pair_index"]:
                raise RuntimeError(
                    f"Score timing pair mismatch for scenario {scenario_rank}."
                )
            model_forward_seconds = as_float(timing, "model_forward_sec")
            screening_wall_seconds = as_float(timing, "scenario_wall_sec")
        else:
            model_forward_seconds = float(
                sum(
                    as_float(row, "model_forward_sec")
                    for row in score_by_scenario[scenario_rank]
                )
            )
            screening_wall_seconds = fallback_wall_per_scenario
        application["is_exhaustive_audit"] = scenario_rank in audit_set
        application["model_forward_seconds"] = model_forward_seconds
        application["screening_wall_seconds"] = screening_wall_seconds
        for k in top_k:
            application[f"forward_plus_fixed_sue_seconds_at_{k}"] = (
                model_forward_seconds + application[f"sue_seconds_at_{k}"]
            )
            application[f"wall_plus_fixed_sue_seconds_at_{k}"] = (
                screening_wall_seconds + application[f"sue_seconds_at_{k}"]
            )
            application[f"forward_plus_adaptive_sue_seconds_at_{k}"] = (
                model_forward_seconds + application[f"adaptive_sue_seconds_at_{k}"]
            )
            application[f"wall_plus_adaptive_sue_seconds_at_{k}"] = (
                screening_wall_seconds + application[f"adaptive_sue_seconds_at_{k}"]
            )
        application_rows.append(application)
    application_rows.sort(key=lambda row: row["scenario_rank"])
    write_csv_atomic(
        args.output_dir / "all_scenarios_final_check_metrics.csv",
        application_rows,
        list(application_rows[0]),
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    application_summary_rows = []
    for k in top_k:
        def app_values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in application_rows], dtype=np.float64)

        meaningful_values = app_values(f"meaningful_success_at_{k}")
        raw_success_values = app_values(f"raw_positive_success_at_{k}")
        no_action_values = app_values(f"no_action_at_{k}")
        verified_values = app_values(f"verified_best_relative_benefit_at_{k}")
        implemented_values = app_values(f"implemented_relative_benefit_at_{k}")
        adaptive_benefit_values = app_values(
            f"adaptive_implemented_relative_benefit_at_{k}"
        )
        fixed_seconds = app_values(f"sue_seconds_at_{k}")
        adaptive_seconds = app_values(f"adaptive_sue_seconds_at_{k}")
        capped_calls = app_values(f"adaptive_capped_calls_at_{k}")
        first_success_calls = app_values(f"calls_to_first_success_at_{k}")
        fixed_pipeline = app_values(f"wall_plus_fixed_sue_seconds_at_{k}")
        adaptive_pipeline = app_values(f"wall_plus_adaptive_sue_seconds_at_{k}")

        success_mean, success_low, success_high, scenario_count = bootstrap_mean_ci(
            meaningful_values, rng=rng, num_bootstrap=args.num_bootstrap
        )
        raw_mean, raw_low, raw_high, _ = bootstrap_mean_ci(
            raw_success_values, rng=rng, num_bootstrap=args.num_bootstrap
        )
        no_action_mean, no_action_low, no_action_high, _ = bootstrap_mean_ci(
            no_action_values, rng=rng, num_bootstrap=args.num_bootstrap
        )
        verified_mean, verified_low, verified_high, _ = bootstrap_mean_ci(
            verified_values, rng=rng, num_bootstrap=args.num_bootstrap
        )
        implemented_mean, implemented_low, implemented_high, _ = bootstrap_mean_ci(
            implemented_values, rng=rng, num_bootstrap=args.num_bootstrap
        )
        adaptive_benefit_mean, adaptive_benefit_low, adaptive_benefit_high, _ = (
            bootstrap_mean_ci(
                adaptive_benefit_values,
                rng=rng,
                num_bootstrap=args.num_bootstrap,
            )
        )
        application_summary_rows.append(
            {
                "k": k,
                "num_scenarios": scenario_count,
                "primary_threshold": primary_threshold,
                "meaningful_success_rate": success_mean,
                "meaningful_success_ci95_lower": success_low,
                "meaningful_success_ci95_upper": success_high,
                "raw_positive_success_rate": raw_mean,
                "raw_positive_success_ci95_lower": raw_low,
                "raw_positive_success_ci95_upper": raw_high,
                "no_action_rate": no_action_mean,
                "no_action_ci95_lower": no_action_low,
                "no_action_ci95_upper": no_action_high,
                "mean_verified_best_relative_benefit": verified_mean,
                "verified_best_ci95_lower": verified_low,
                "verified_best_ci95_upper": verified_high,
                "median_verified_best_relative_benefit": percentile(verified_values, 50),
                "p95_verified_best_relative_benefit": percentile(verified_values, 95),
                "mean_implemented_relative_benefit": implemented_mean,
                "implemented_benefit_ci95_lower": implemented_low,
                "implemented_benefit_ci95_upper": implemented_high,
                "median_implemented_relative_benefit": percentile(
                    implemented_values, 50
                ),
                "fixed_budget_sue_calls": k,
                "mean_fixed_sue_seconds": safe_mean(fixed_seconds),
                "median_fixed_sue_seconds": percentile(fixed_seconds, 50),
                "p95_fixed_sue_seconds": percentile(fixed_seconds, 95),
                "mean_fixed_pipeline_seconds": safe_mean(fixed_pipeline),
                "p95_fixed_pipeline_seconds": percentile(fixed_pipeline, 95),
                "conditional_mean_calls_to_first_success": safe_mean(
                    first_success_calls
                ),
                "conditional_median_calls_to_first_success": percentile(
                    first_success_calls, 50
                ),
                "conditional_p95_calls_to_first_success": percentile(
                    first_success_calls, 95
                ),
                "mean_adaptive_capped_calls": safe_mean(capped_calls),
                "median_adaptive_capped_calls": percentile(capped_calls, 50),
                "p95_adaptive_capped_calls": percentile(capped_calls, 95),
                "adaptive_stopped_early_rate": safe_mean(
                    app_values(f"adaptive_stopped_early_at_{k}")
                ),
                "mean_adaptive_sue_seconds": safe_mean(adaptive_seconds),
                "median_adaptive_sue_seconds": percentile(adaptive_seconds, 50),
                "p95_adaptive_sue_seconds": percentile(adaptive_seconds, 95),
                "mean_adaptive_pipeline_seconds": safe_mean(adaptive_pipeline),
                "p95_adaptive_pipeline_seconds": percentile(adaptive_pipeline, 95),
                "mean_adaptive_implemented_relative_benefit": adaptive_benefit_mean,
                "adaptive_benefit_ci95_lower": adaptive_benefit_low,
                "adaptive_benefit_ci95_upper": adaptive_benefit_high,
                "deployment_call_reduction_factor": float(
                    candidates_per_scenario / k
                ),
                "deployment_call_savings_fraction": float(
                    1.0 - k / candidates_per_scenario
                ),
            }
        )
    write_csv_atomic(
        args.output_dir / "all_scenarios_final_check_summary.csv",
        application_summary_rows,
        list(application_summary_rows[0]),
    )

    verified_threshold_rows = []
    for threshold in thresholds:
        threshold_scenarios = [
            final_check_scenario_metrics(
                score_by_scenario[scenario_rank],
                evaluated_by_id=usable_by_id,
                top_k=top_k,
                threshold=threshold,
                evaluation_scope=(
                    "exhaustive_audit"
                    if scenario_rank in audit_set
                    else "model_top_k"
                ),
            )
            for scenario_rank in complete_ranks
        ]
        for k in top_k:
            success = np.asarray(
                [row[f"meaningful_success_at_{k}"] for row in threshold_scenarios],
                dtype=np.float64,
            )
            no_action = np.asarray(
                [row[f"no_action_at_{k}"] for row in threshold_scenarios],
                dtype=np.float64,
            )
            implemented = np.asarray(
                [
                    row[f"implemented_relative_benefit_at_{k}"]
                    for row in threshold_scenarios
                ],
                dtype=np.float64,
            )
            adaptive_calls = np.asarray(
                [row[f"adaptive_capped_calls_at_{k}"] for row in threshold_scenarios],
                dtype=np.float64,
            )
            first_success_calls = np.asarray(
                [row[f"calls_to_first_success_at_{k}"] for row in threshold_scenarios],
                dtype=np.float64,
            )
            adaptive_benefit = np.asarray(
                [
                    row[f"adaptive_implemented_relative_benefit_at_{k}"]
                    for row in threshold_scenarios
                ],
                dtype=np.float64,
            )
            success_mean, success_low, success_high, count = bootstrap_mean_ci(
                success, rng=rng, num_bootstrap=args.num_bootstrap
            )
            implemented_mean, implemented_low, implemented_high, _ = (
                bootstrap_mean_ci(
                    implemented, rng=rng, num_bootstrap=args.num_bootstrap
                )
            )
            verified_threshold_rows.append(
                {
                    "population": "all_complete_hybrid_scenarios",
                    "scope": "verified_model_top_k_only",
                    "threshold": float(threshold),
                    "k": k,
                    "num_scenarios": count,
                    "meaningful_success_rate": success_mean,
                    "success_ci95_lower": success_low,
                    "success_ci95_upper": success_high,
                    "no_action_rate": safe_mean(no_action),
                    "mean_fixed_implemented_relative_benefit": implemented_mean,
                    "implemented_benefit_ci95_lower": implemented_low,
                    "implemented_benefit_ci95_upper": implemented_high,
                    "mean_adaptive_capped_calls": safe_mean(adaptive_calls),
                    "median_adaptive_capped_calls": percentile(adaptive_calls, 50),
                    "p95_adaptive_capped_calls": percentile(adaptive_calls, 95),
                    "conditional_mean_calls_to_first_success": safe_mean(
                        first_success_calls
                    ),
                    "conditional_p95_calls_to_first_success": percentile(
                        first_success_calls, 95
                    ),
                    "mean_adaptive_implemented_relative_benefit": safe_mean(
                        adaptive_benefit
                    ),
                }
            )
    write_csv_atomic(
        args.output_dir / "all_scenarios_verified_threshold_sensitivity.csv",
        verified_threshold_rows,
        list(verified_threshold_rows[0]),
    )

    methods = list(METHOD_SCORES)
    audit_per_scenario = []
    for scenario_rank, rows in sorted(audit_scenarios.items()):
        for method, score_key in METHOD_SCORES.items():
            audit_per_scenario.append(
                scenario_method_metrics(
                    rows,
                    method=method,
                    score_key=score_key,
                    top_k=top_k,
                    threshold=primary_threshold,
                )
            )
    write_csv_atomic(
        args.output_dir / "audit_per_scenario_ranking_metrics.csv",
        audit_per_scenario,
        list(audit_per_scenario[0]),
    )
    metric_names = [
        key
        for key in audit_per_scenario[0]
        if key
        not in {
            "scenario_rank",
            "pair_index",
            "method",
            "num_candidates",
            "num_braessian",
            "has_braessian",
        }
    ]
    audit_aggregate_rows = []
    audit_aggregate_lookup: dict[tuple[str, str], float] = {}
    for method in methods:
        method_rows = [
            row for row in audit_per_scenario if row["method"] == method
        ]
        for metric in metric_names:
            mean, low, high, count = bootstrap_mean_ci(
                np.asarray(
                    [float(row[metric]) for row in method_rows], dtype=np.float64
                ),
                rng=rng,
                num_bootstrap=args.num_bootstrap,
            )
            audit_aggregate_rows.append(
                {
                    "population": "exhaustive_audit_only",
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "num_scenarios": count,
                }
            )
            audit_aggregate_lookup[(method, metric)] = mean
    write_csv_atomic(
        args.output_dir / "audit_aggregate_ranking_metrics.csv",
        audit_aggregate_rows,
        (
            "population",
            "method",
            "metric",
            "mean",
            "ci95_lower",
            "ci95_upper",
            "num_scenarios",
        ),
    )

    higher_exact = {"spearman", "kendall", "average_precision"}
    higher_prefixes = (
        "ndcg_at_",
        "topk_overlap_at_",
        "precision_at_",
        "recall_at_",
        "success_at_",
        "success_conditional_at_",
        "oracle_best_hit_at_",
        "benefit_recovery_at_",
        "raw_positive_benefit_recovery_at_",
        "meaningful_benefit_recovery_at_",
    )
    paired_metric_names = [
        metric
        for metric in metric_names
        if metric in higher_exact or metric.startswith(higher_prefixes)
    ]
    per_method_scenario = {
        method: {
            int(row["scenario_rank"]): row
            for row in audit_per_scenario
            if row["method"] == method
        }
        for method in methods
    }
    paired_rows = []
    for baseline in methods:
        if baseline == "edge_transformer":
            continue
        common_scenarios = sorted(
            set(per_method_scenario["edge_transformer"])
            & set(per_method_scenario[baseline])
        )
        for metric in paired_metric_names:
            differences = []
            for scenario_rank in common_scenarios:
                model_value = float(
                    per_method_scenario["edge_transformer"][scenario_rank][metric]
                )
                baseline_value = float(
                    per_method_scenario[baseline][scenario_rank][metric]
                )
                if np.isfinite(model_value) and np.isfinite(baseline_value):
                    differences.append(model_value - baseline_value)
            difference_array = np.asarray(differences, dtype=np.float64)
            mean, low, high, count = bootstrap_mean_ci(
                difference_array, rng=rng, num_bootstrap=args.num_bootstrap
            )
            paired_rows.append(
                {
                    "population": "exhaustive_audit_only",
                    "model": "edge_transformer",
                    "baseline": baseline,
                    "metric": metric,
                    "mean_paired_difference": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "fraction_paired_scenarios_model_better": (
                        float(np.mean(difference_array > 0.0))
                        if difference_array.size
                        else math.nan
                    ),
                    "num_paired_scenarios": count,
                }
            )
    write_csv_atomic(
        args.output_dir / "audit_paired_method_differences.csv",
        paired_rows,
        (
            "population",
            "model",
            "baseline",
            "metric",
            "mean_paired_difference",
            "ci95_lower",
            "ci95_upper",
            "fraction_paired_scenarios_model_better",
            "num_paired_scenarios",
        ),
    )

    audit_merged = [
        row
        for scenario_rank in sorted(audit_scenarios)
        for row in audit_scenarios[scenario_rank]
    ]
    threshold_results = []
    for threshold in thresholds:
        for method, score_key in METHOD_SCORES.items():
            scenario_values = [
                scenario_method_metrics(
                    rows,
                    method=method,
                    score_key=score_key,
                    top_k=top_k,
                    threshold=threshold,
                )
                for rows in audit_scenarios.values()
            ]
            pooled_truth = np.asarray(
                [row["oracle_relative_benefit"] for row in audit_merged],
                dtype=np.float64,
            )
            pooled_score = np.asarray(
                [row[score_key] for row in audit_merged], dtype=np.float64
            )
            record = {
                "population": "exhaustive_audit_only",
                "threshold": float(threshold),
                "method": method,
                "num_audit_scenarios": len(audit_scenarios),
                "candidate_prevalence": float(np.mean(pooled_truth > threshold)),
                "scenario_prevalence": float(
                    np.mean(
                        [
                            np.any(
                                np.asarray(
                                    [
                                        row["oracle_relative_benefit"]
                                        for row in rows
                                    ]
                                )
                                > threshold
                            )
                            for rows in audit_scenarios.values()
                        ]
                    )
                ),
                "pooled_average_precision": average_precision(
                    pooled_score, pooled_truth > threshold
                ),
                "macro_average_precision": safe_mean(
                    row["average_precision"] for row in scenario_values
                ),
            }
            for k in top_k:
                for metric in (
                    "precision",
                    "recall",
                    "success",
                    "success_conditional",
                ):
                    record[f"{metric}_at_{k}"] = safe_mean(
                        row[f"{metric}_at_{k}"] for row in scenario_values
                    )
            threshold_results.append(record)
    write_csv_atomic(
        args.output_dir / "audit_braess_threshold_sensitivity.csv",
        threshold_results,
        list(threshold_results[0]),
    )

    application_by_rank = {
        int(row["scenario_rank"]): row for row in application_rows
    }
    top_edge_rows = []
    maximum_k = max(top_k)
    for scenario_rank in complete_ranks:
        ordered = stable_model_order(score_by_scenario[scenario_rank])
        oracle_rank: dict[int, int] = {}
        if scenario_rank in audit_scenarios:
            oracle_rank = {
                int(row["row_id"]): rank
                for rank, row in enumerate(
                    sorted(
                        audit_scenarios[scenario_rank],
                        key=lambda row: (
                            -float(row["oracle_relative_benefit"]),
                            int(row["candidate_rank"]),
                            int(row["row_id"]),
                        ),
                    ),
                    start=1,
                )
            }
        fixed_action_id = int(
            application_by_rank[scenario_rank][
                f"selected_row_id_at_{maximum_k}"
            ]
        )
        adaptive_action_id = int(
            application_by_rank[scenario_rank][
                f"adaptive_selected_row_id_at_{maximum_k}"
            ]
        )
        for score_row in ordered[:maximum_k]:
            row_id = as_int(score_row, "row_id")
            evaluated = usable_by_id[row_id]
            top_edge_rows.append(
                {
                    "scenario_rank": scenario_rank,
                    "pair_index": as_int(score_row, "pair_index"),
                    "evaluation_scope": evaluated["evaluation_scope"],
                    "is_exhaustive_audit": scenario_rank in audit_set,
                    "model_rank": as_int(score_row, "model_rank"),
                    "oracle_rank": oracle_rank.get(row_id, ""),
                    "edge_u": as_int(score_row, "edge_u"),
                    "edge_v": as_int(score_row, "edge_v"),
                    "model_relative_benefit": as_float(
                        score_row, "model_relative_benefit"
                    ),
                    "oracle_relative_benefit": evaluated[
                        "oracle_relative_benefit"
                    ],
                    "oracle_absolute_benefit": evaluated["oracle_benefit"],
                    "is_meaningful_braessian": (
                        evaluated["oracle_relative_benefit"] > primary_threshold
                    ),
                    "is_fixed_budget_action_at_max_k": row_id == fixed_action_id,
                    "is_adaptive_first_action_at_max_k": (
                        row_id == adaptive_action_id
                    ),
                }
            )
    write_csv_atomic(
        args.output_dir / "verified_model_top_edges.csv",
        top_edge_rows,
        list(top_edge_rows[0]),
    )

    evaluated_fields = (
        "row_id",
        "scenario_rank",
        "pair_index",
        "candidate_rank",
        "model_rank",
        "edge_u",
        "edge_v",
        "evaluation_scope",
        "baseline_tstt",
        "model_removed_tstt",
        "oracle_removed_tstt",
        "model_relative_benefit",
        "oracle_benefit",
        "oracle_relative_benefit",
        "elapsed_solver_sec",
        "converged",
    )
    write_csv_atomic(
        args.output_dir / "hybrid_evaluated_candidates.csv",
        merged,
        evaluated_fields,
    )

    total_solver_seconds = float(sum(row["elapsed_solver_sec"] for row in merged))
    mean_solver_seconds = total_solver_seconds / max(len(merged), 1)
    audit_full_seconds = np.asarray(
        [
            sum(row["elapsed_solver_sec"] for row in rows)
            for rows in audit_scenarios.values()
        ],
        dtype=np.float64,
    )
    model_forward_seconds = np.asarray(
        [row["model_forward_seconds"] for row in application_rows],
        dtype=np.float64,
    )
    screening_wall_seconds = np.asarray(
        [row["screening_wall_seconds"] for row in application_rows],
        dtype=np.float64,
    )
    possible_full_sue_calls = requested_scenarios * candidates_per_scenario
    expected_hybrid_calls = len(hybrid_rows)
    application_summary_by_k = {
        int(row["k"]): row for row in application_summary_rows
    }
    screening_efficiency = {}
    audit_full_mean_seconds = safe_mean(audit_full_seconds)
    for k in top_k:
        record = application_summary_by_k[k]
        fixed_pipeline = float(record["mean_fixed_pipeline_seconds"])
        adaptive_pipeline = float(record["mean_adaptive_pipeline_seconds"])
        screening_efficiency[str(k)] = {
            "population_for_application_metrics": "all_complete_hybrid_scenarios",
            "deployment_sue_calls_per_scenario_fixed_budget": k,
            "deployment_call_reduction_factor": float(
                candidates_per_scenario / k
            ),
            "deployment_call_savings_fraction": float(
                1.0 - k / candidates_per_scenario
            ),
            "meaningful_success_rate": record["meaningful_success_rate"],
            "no_action_rate": record["no_action_rate"],
            "mean_fixed_sue_seconds": record["mean_fixed_sue_seconds"],
            "mean_adaptive_capped_calls": record[
                "mean_adaptive_capped_calls"
            ],
            "conditional_mean_calls_to_first_success": record[
                "conditional_mean_calls_to_first_success"
            ],
            "mean_adaptive_sue_seconds": record[
                "mean_adaptive_sue_seconds"
            ],
            "audit_estimated_fixed_pipeline_speedup_vs_exhaustive_sue": float(
                audit_full_mean_seconds / max(fixed_pipeline, 1e-12)
            ),
            "audit_estimated_adaptive_pipeline_speedup_vs_exhaustive_sue": float(
                audit_full_mean_seconds / max(adaptive_pipeline, 1e-12)
            ),
            "audit_only_benefit_recovery": audit_aggregate_lookup.get(
                ("edge_transformer", f"benefit_recovery_at_{k}"), math.nan
            ),
            "audit_only_oracle_best_hit_rate": audit_aggregate_lookup.get(
                ("edge_transformer", f"oracle_best_hit_at_{k}"), math.nan
            ),
            "audit_only_conditional_success_rate": audit_aggregate_lookup.get(
                ("edge_transformer", f"success_conditional_at_{k}"), math.nan
            ),
        }

    figures = make_plots(
        args.output_dir,
        merged_rows=audit_merged,
        aggregate_lookup=audit_aggregate_lookup,
        methods=methods,
        top_k=top_k,
    )
    compact_shard_metadata = [
        {
            "shard_id": int(item["shard_id"]),
            "completed": bool(item.get("completed", False)),
            "num_scenarios_in_shard": int(item["num_scenarios_in_shard"]),
            "num_candidates_in_shard": int(item["num_candidates_in_shard"]),
            "num_failed": int(item.get("num_failed", 0)),
            "num_unconverged": int(item.get("num_unconverged", 0)),
        }
        for item in oracle_metadata
    ]
    summary = {
        "task": "braessian-edge-hybrid-downstream-summary-v3",
        "network": "EMA",
        "score_protocol_signature": score_signature,
        "oracle_protocol_signature": next(iter(oracle_signatures)),
        "hybrid_manifest_signature": hybrid_manifest_signature,
        "protocol": {
            "num_scenarios_requested": requested_scenarios,
            "num_complete_hybrid_scenarios": len(complete_ranks),
            "num_incomplete_scenarios_excluded": len(incomplete_ranks),
            "num_audit_scenarios_registered": len(audit_ranks),
            "num_complete_audit_scenarios": len(audit_scenarios),
            "num_deployment_scenarios_registered": (
                requested_scenarios - len(audit_ranks)
            ),
            "audit_seed": int(policy["audit_seed"]),
            "audit_scenario_ranks": audit_ranks,
            "final_check_top_k": final_check_top_k,
            "candidates_per_scenario": candidates_per_scenario,
            "expected_hybrid_sue_calls": expected_hybrid_calls,
            "hypothetical_exhaustive_sue_calls": possible_full_sue_calls,
            "offline_protocol_call_reduction_factor": float(
                possible_full_sue_calls / max(expected_hybrid_calls, 1)
            ),
            "offline_protocol_call_savings_fraction": float(
                1.0 - expected_hybrid_calls / max(possible_full_sue_calls, 1)
            ),
        },
        "metric_scope": {
            "all_scenarios": (
                "Only model top-k final-check success, verified/implemented benefit, "
                "no-action, adaptive early-stop, SUE calls, latency, and threshold "
                "sensitivity over those verified top-k candidates."
            ),
            "exhaustive_audit_only": (
                "Full ranking, heuristic comparison, prevalence, recall, oracle-best "
                "hit, benefit recovery, and regret. No such claims are made for "
                "partial-oracle deployment scenarios."
            ),
        },
        "primary_braess_threshold": primary_threshold,
        "braess_thresholds": thresholds,
        "top_k": top_k,
        "tstt_unit": "vehicle-minutes/hour",
        "statistics_unit": "scenario (cluster/macro), not candidate edge",
        "num_bootstrap": int(args.num_bootstrap),
        "quality_control": {
            "num_failed_oracle_rows": len(failed),
            "num_unconverged_oracle_rows": len(unconverged),
            "oracle_convergence_rate_among_successful_calls": float(
                1.0 - len(unconverged) / max(oracle_ok_count, 1)
            ),
            "require_converged": bool(args.require_converged),
            "allow_incomplete": bool(args.allow_incomplete),
        },
        "runtime": {
            "model_forward_seconds_total_complete_scenarios": float(
                model_forward_seconds.sum()
            ),
            "model_forward_seconds_mean_per_scenario": safe_mean(
                model_forward_seconds
            ),
            "screening_wall_seconds_mean_per_scenario": safe_mean(
                screening_wall_seconds
            ),
            "screening_wall_seconds_p95_per_scenario": percentile(
                screening_wall_seconds, 95
            ),
            "ranking_stage_wall_seconds_total": ranking_wall_seconds_total,
            "hybrid_sue_solver_core_seconds_total": total_solver_seconds,
            "hybrid_sue_mean_seconds_per_call": mean_solver_seconds,
            "audit_exhaustive_sue_seconds_mean_per_scenario": (
                audit_full_mean_seconds
            ),
            "audit_exhaustive_sue_seconds_median_per_scenario": percentile(
                audit_full_seconds, 50
            ),
            "audit_exhaustive_sue_seconds_p95_per_scenario": percentile(
                audit_full_seconds, 95
            ),
        },
        "screening_efficiency": screening_efficiency,
        "files": {
            "all_scenarios_final_check_metrics": str(
                args.output_dir / "all_scenarios_final_check_metrics.csv"
            ),
            "all_scenarios_final_check_summary": str(
                args.output_dir / "all_scenarios_final_check_summary.csv"
            ),
            "all_scenarios_verified_threshold_sensitivity": str(
                args.output_dir
                / "all_scenarios_verified_threshold_sensitivity.csv"
            ),
            "audit_per_scenario_ranking_metrics": str(
                args.output_dir / "audit_per_scenario_ranking_metrics.csv"
            ),
            "audit_aggregate_ranking_metrics": str(
                args.output_dir / "audit_aggregate_ranking_metrics.csv"
            ),
            "audit_paired_method_differences": str(
                args.output_dir / "audit_paired_method_differences.csv"
            ),
            "audit_threshold_sensitivity": str(
                args.output_dir / "audit_braess_threshold_sensitivity.csv"
            ),
            "verified_model_top_edges": str(
                args.output_dir / "verified_model_top_edges.csv"
            ),
            "hybrid_evaluated_candidates": str(
                args.output_dir / "hybrid_evaluated_candidates.csv"
            ),
        },
        "figures": figures,
        "oracle_shards": compact_shard_metadata,
    }
    save_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))



def add_shared_raw_validation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument(
        "--validation-marker",
        type=Path,
        required=True,
        help="Required RAW_PAIRS_MATCH_PYG.ok marker bound to the reconstructed EMA pickle.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="GPU model ranking of feasible removals.")
    add_shared_raw_validation(score)
    score.add_argument("--dataset-dir", required=True, type=Path)
    score.add_argument("--run-dir", required=True, type=Path)
    score.add_argument("--checkpoint", default="")
    score.add_argument("--cfg", default="configs/GatedGCN/network-pairs-topology.yaml")
    score.add_argument("--network", default="EMA", choices=("EMA",))
    score.add_argument("--output-dir", required=True, type=Path)
    score.add_argument("--num-scenarios", type=int, default=500, help="0 uses all test scenarios.")
    score.add_argument("--selection-seed", type=int, default=2026)
    score.add_argument("--seed", type=int, default=0)
    score.add_argument(
        "--batch-size",
        type=int,
        default=236,
        help="Cached candidate templates per forward pass; 236 covers all EMA removals.",
    )
    score.add_argument("--num-threads", type=int, default=16)
    score.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    score.add_argument(
        "--prediction-flow-policy", choices=("clip_zero", "raw"), default="clip_zero"
    )
    score.add_argument("--resume", action="store_true")

    oracle = subparsers.add_parser(
        "oracle", help="Sharded hybrid SUE audit and model-top-k final checks."
    )
    add_shared_raw_validation(oracle)
    oracle.add_argument("--ranking-csv", required=True, type=Path)
    oracle.add_argument("--ranking-metadata", required=True, type=Path)
    oracle.add_argument("--output-dir", required=True, type=Path)
    oracle.add_argument("--shard-id", type=int, required=True)
    oracle.add_argument("--num-shards", type=int, default=10)
    oracle.add_argument("--num-workers", type=int, default=16)
    oracle.add_argument(
        "--audit-scenarios",
        type=int,
        default=100,
        help="Random scenarios with exhaustive SUE checks for unbiased ranking audit.",
    )
    oracle.add_argument("--audit-seed", type=int, default=2027)
    oracle.add_argument(
        "--final-check-k",
        type=int,
        default=10,
        help="Model-ranked candidates checked by SUE outside the audit subset.",
    )
    oracle.add_argument("--max-iter", type=int, default=120)
    oracle.add_argument("--convergence-threshold", type=float, default=1e-5)
    oracle.add_argument("--theta", type=float, default=0.8)
    oracle.add_argument("--value-iter", type=int, default=250)
    oracle.add_argument("--value-tol", type=float, default=1e-8)
    oracle.add_argument("--flow-iter", type=int, default=500)
    oracle.add_argument("--flow-tol", type=float, default=1e-9)
    oracle.add_argument("--retry-flow-iter", type=int, default=2000)
    oracle.add_argument(
        "--sue-loading-protocol",
        choices=(
            "legacy_unrestricted",
            "reasonable_links",
            "stable_unrestricted",
            "stable_reasonable_links",
        ),
        default="legacy_unrestricted",
    )
    oracle.add_argument("--step-rule", choices=("msa_sr", "residual_aware"), default="msa_sr")
    oracle.add_argument("--resume", action="store_true")

    summary = subparsers.add_parser("summarize", help="Rank and application-value summary.")
    summary.add_argument("--ranking-csv", required=True, type=Path)
    summary.add_argument("--ranking-metadata", required=True, type=Path)
    summary.add_argument("--oracle-dir", required=True, type=Path)
    summary.add_argument("--output-dir", required=True, type=Path)
    summary.add_argument("--top-k", default="1,3,5,10")
    summary.add_argument("--primary-braess-threshold", type=float, default=0.001)
    summary.add_argument("--braess-thresholds", default="0,0.0005,0.001,0.005")
    summary.add_argument("--num-bootstrap", type=int, default=2000)
    summary.add_argument("--bootstrap-seed", type=int, default=2026)
    summary.add_argument("--require-converged", action="store_true")
    summary.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "score":
        score_candidates(args)
    elif args.command == "oracle":
        run_oracle(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
