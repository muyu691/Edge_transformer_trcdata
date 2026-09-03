#!/usr/bin/env python3
"""Reconstruct warm-start-compatible raw pairs from PyG data and LHS scenarios."""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct network_pairs_dataset.pkl from an existing PyG dataset. "
            "OD matrices are restored from the deterministic base_scenarios.npz."
        )
    )
    parser.add_argument("--pyg-dir", required=True)
    parser.add_argument("--scenarios-npz", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--expected-network", default="EMA")
    parser.add_argument("--expected-pairs", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inverse_standardized(values: np.ndarray, scaler) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * np.asarray(scaler.scale_, dtype=np.float64) + np.asarray(
        scaler.mean_, dtype=np.float64
    )


def decode_edges(edge_index, node_ids: tuple[int, ...]) -> list[tuple[int, int]]:
    indices = edge_index.detach().cpu().numpy()
    return [
        (node_ids[int(source)], node_ids[int(target)])
        for source, target in indices.T
    ]


def build_graph(
    node_ids: tuple[int, ...],
    edge_list: list[tuple[int, int]],
    edge_attrs: np.ndarray,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    for (source, target), (capacity, speed, length) in zip(edge_list, edge_attrs):
        speed = float(speed)
        length = float(length)
        graph.add_edge(
            int(source),
            int(target),
            capacity=float(capacity),
            speed=speed,
            length=length,
            free_flow_time=(length / max(speed, 1e-12)) * 60.0,
        )
    return graph


def reconstruct_pair(
    data,
    raw_index: int,
    od_matrices: np.ndarray,
    scenario_capacities: np.ndarray,
    scenario_speeds: np.ndarray,
    attr_scaler,
    flow_scaler,
    expected_network: str,
) -> dict:
    network_name = str(getattr(data, "network_name", ""))
    if network_name.lower() != expected_network.lower():
        raise ValueError(
            f"Sample {raw_index} has network_name={network_name!r}, expected {expected_network!r}."
        )

    node_ids = tuple(int(value) for value in data.node_ids.detach().cpu().tolist())
    edge_list_old = decode_edges(data.edge_index_old, node_ids)
    edge_list_new = decode_edges(data.edge_index_new, node_ids)

    pyg_attr_old = inverse_standardized(
        data.edge_attr_old.detach().cpu().numpy(), attr_scaler
    )
    attr_new = inverse_standardized(
        data.edge_attr_new.detach().cpu().numpy(), attr_scaler
    )

    capacities = np.asarray(scenario_capacities[raw_index], dtype=np.float64)
    speeds = np.asarray(scenario_speeds[raw_index], dtype=np.float64)
    if capacities.shape != (len(edge_list_old),) or speeds.shape != (len(edge_list_old),):
        raise ValueError(
            f"Scenario {raw_index} edge arrays do not match PyG old-edge count "
            f"({capacities.shape}, {speeds.shape}, {len(edge_list_old)})."
        )
    # The LHS archive retains float64 old capacity/speed values. PyG supplies
    # edge order and lengths, which are static network attributes.
    attr_old = np.column_stack([capacities, speeds, pyg_attr_old[:, 2]])

    flows_old = inverse_standardized(
        data.flow_old.detach().cpu().numpy(), flow_scaler
    ).reshape(-1)
    flows_new = inverse_standardized(
        data.y.detach().cpu().numpy(), flow_scaler
    ).reshape(-1)

    non_centroid_mask = data.non_centroid_mask.detach().cpu().numpy().astype(bool)
    centroid_nodes = tuple(
        node_id for node_id, is_non_centroid in zip(node_ids, non_centroid_mask)
        if not is_non_centroid
    )

    return {
        "od_matrix": np.asarray(od_matrices[raw_index], dtype=np.float64).copy(),
        "G": build_graph(node_ids, edge_list_old, attr_old),
        "G_prime": build_graph(node_ids, edge_list_new, attr_new),
        "mutation_type": str(getattr(data, "mutation_type", "both")),
        "mutation_info": {
            "type": str(getattr(data, "mutation_type", "both")),
            "reconstructed_from_pyg": True,
            "raw_index": int(raw_index),
        },
        "flows_old": flows_old,
        "flows_new": flows_new,
        "edge_list_old": edge_list_old,
        "edge_list_new": edge_list_new,
        "network_name": network_name,
        "node_ids": node_ids,
        "centroid_nodes": centroid_nodes,
        "node_id_offset": 1,
    }


def save_pickle_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    pyg_dir = Path(args.pyg_dir).resolve()
    scenarios_path = Path(args.scenarios_npz).resolve()
    output_path = Path(args.output_pkl).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")

    with np.load(scenarios_path) as scenarios:
        od_matrices = np.asarray(scenarios["od_matrices"], dtype=np.float64)
        capacities = np.asarray(scenarios["capacities"], dtype=np.float64)
        speeds = np.asarray(scenarios["speeds"], dtype=np.float64)
    if od_matrices.shape[0] != args.expected_pairs:
        raise ValueError(
            f"Scenario count={od_matrices.shape[0]}, expected={args.expected_pairs}."
        )

    with (pyg_dir / "scalers" / "attr_scaler.pkl").open("rb") as handle:
        attr_scaler = pickle.load(handle)
    with (pyg_dir / "scalers" / "flow_scaler.pkl").open("rb") as handle:
        flow_scaler = pickle.load(handle)
    with np.load(pyg_dir / "split_indices.npz") as split_file:
        split_indices = {
            split: np.asarray(split_file[f"{split}_idx"], dtype=np.int64)
            for split in SPLITS
        }

    all_indices = np.concatenate([split_indices[split] for split in SPLITS])
    if len(all_indices) != args.expected_pairs or not np.array_equal(
        np.sort(all_indices), np.arange(args.expected_pairs, dtype=np.int64)
    ):
        raise ValueError("PyG split indices are not a complete raw-index permutation.")

    pairs: list[dict | None] = [None] * args.expected_pairs
    reconstructed = 0
    for split in SPLITS:
        dataset_path = pyg_dir / f"{split}_dataset.pt"
        print(f"Loading {split}: {dataset_path}")
        dataset = torch_load(dataset_path)
        indices = split_indices[split]
        if len(dataset) != len(indices):
            raise ValueError(
                f"{split} dataset length={len(dataset)}, split index length={len(indices)}."
            )
        for split_position, data in enumerate(dataset):
            raw_index = int(indices[split_position])
            if pairs[raw_index] is not None:
                raise ValueError(f"Duplicate raw index in PyG splits: {raw_index}")
            pairs[raw_index] = reconstruct_pair(
                data,
                raw_index,
                od_matrices,
                capacities,
                speeds,
                attr_scaler,
                flow_scaler,
                args.expected_network,
            )
            reconstructed += 1
            if reconstructed % 1000 == 0:
                print(f"Reconstructed {reconstructed}/{args.expected_pairs} pairs")
        del dataset
        gc.collect()

    if any(pair is None for pair in pairs):
        missing = [index for index, pair in enumerate(pairs) if pair is None]
        raise RuntimeError(f"Missing reconstructed raw indices: {missing[:20]}")

    payload = {
        "pairs": pairs,
        "failed_indices": [],
        "reconstruction_metadata": {
            "source": "existing_pyg_plus_base_scenarios",
            "expected_network": args.expected_network,
            "num_pairs": args.expected_pairs,
            "od_source": str(scenarios_path),
            "pyg_source": str(pyg_dir),
            "precision_note": (
                "New-graph attributes and flows were inverse-transformed from float32 PyG tensors."
            ),
        },
    }
    save_pickle_atomic(output_path, payload)
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Saved {args.expected_pairs} reconstructed raw pairs: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
