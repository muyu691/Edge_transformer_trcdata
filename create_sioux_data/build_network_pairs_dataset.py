"""
Build a processed PyG dataset from solved traffic network pairs.

Input:
  network_pairs_dataset.pkl, or a network directory containing batch_* shards

Output:
  train_dataset.pt / val_dataset.pt / test_dataset.pt
  scalers/attr_scaler.pkl / scalers/flow_scaler.pkl
  dataset_meta.json
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from solve_network_pairs import _MODEL_KEYS, validate_sue_certificates
except ModuleNotFoundError:
    from .solve_network_pairs import _MODEL_KEYS, validate_sue_certificates


class ShardedPairs:
    """Re-iterable raw input; only one pickle shard is retained at a time."""

    def __init__(self, input_pkl=None, input_dir=None):
        self.input_dir = input_dir
        if input_dir:
            self.root = Path(input_dir).resolve()
            batches = sorted(path for path in self.root.glob("batch_*") if path.is_dir())
            self.paths = [path / "network_pairs_dataset.pkl" for path in batches]
        else:
            path = Path(input_pkl).resolve()
            self.root = path.parent
            self.paths = [path]
        if not self.paths:
            raise ValueError(f"No batch_* input shards found in {self.root}")
        for path in self.paths:
            if not path.is_file():
                raise ValueError(f"Missing/incomplete input shard: {path}. Finish generation first.")
        self.snapshot = self._snapshot()

    def _snapshot(self):
        return [(path.stat().st_size, path.stat().st_mtime_ns) for path in self.paths]

    def __iter__(self):
        current_paths = ([path / "network_pairs_dataset.pkl" for path in sorted(self.root.glob("batch_*"))
                          if path.is_dir()] if self.input_dir else self.paths)
        if current_paths != self.paths or self._snapshot() != self.snapshot:
            raise ValueError("Input shards changed during conversion. Finish generation first.")
        global_idx = 0
        for path in self.paths:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            pairs = payload["pairs"] if isinstance(payload, dict) else payload
            validate_sue_certificates(pairs)  # Local sample_idx is unique within a shard only.
            for offset, pair in enumerate(pairs):
                yield dict(pair, global_sample_idx=global_idx,
                           source_sample_idx=int(pair["sample_idx"]),
                           source_shard=path.relative_to(self.root).as_posix(),
                           source_pair_offset=offset)
                global_idx += 1
            # Drop the previous payload before loading the next shard.
            if pairs:
                del pair
            del pairs, payload


def scan_pair_sources(pairs):
    """Keep only provenance, enforcing one network/model and no copied scenarios."""
    sources, seen = [], set()
    signature = None
    model = None
    for pair in pairs:
        model = {key: pair["sue_diagnostics_old"].get(key) for key in _MODEL_KEYS}
        current = (
            str(_require_pair_metadata(pair, "network_name")),
            tuple(_require_pair_metadata(pair, "node_ids")),
            tuple(_require_pair_metadata(pair, "centroid_nodes")),
            tuple(map(tuple, pair["edge_list_old"])),
            tuple(model.values()),
        )
        if signature is not None and signature != current:
            raise ValueError("Input mixes networks, base topology, or SUE model parameters.")
        signature = current
        digest = pair["sue_diagnostics_old"]["data_digest"]
        if digest in seen:
            raise ValueError("Duplicate base scenario across shards; refusing split leakage.")
        seen.add(digest)
        sources.append({
            "sample_idx": pair["global_sample_idx"],
            "source_sample_idx": pair["source_sample_idx"],
            "source_shard": pair["source_shard"],
            "source_pair_offset": pair["source_pair_offset"],
            "network_name": pair["network_name"],
            "old_data_digest": digest,
            "new_data_digest": pair["sue_diagnostics_new"]["data_digest"],
        })
    if not sources:
        raise ValueError("No verified pairs to build; inspect the generation failure records.")
    return sources, model


def extract_edge_attrs(G, edge_list: list) -> np.ndarray:
    """Extract [capacity, speed, length] in the canonical edge order."""
    capacities = np.array([G[u][v]["capacity"] for u, v in edge_list], dtype=np.float64)
    speeds = np.array([G[u][v]["speed"] for u, v in edge_list], dtype=np.float64)
    lengths = np.array([G[u][v]["length"] for u, v in edge_list], dtype=np.float64)
    return np.column_stack([capacities, speeds, lengths])


def _require_pair_metadata(pair: dict, key: str):
    value = pair.get(key)
    if value in (None, "", ()):
        raise ValueError(
            f"Missing pair['{key}'] in the raw dataset. "
            "This builder only supports the new network_pairs_dataset.pkl generated "
            "by the current solve_network_pairs.py pipeline."
        )
    return value


def edge_list_to_index(edge_list: list, node_id_to_index: dict[int, int]) -> np.ndarray:
    """Convert raw node ids to contiguous 0-indexed edge_index."""
    if len(edge_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(
        [[node_id_to_index[u], node_id_to_index[v]] for u, v in edge_list],
        dtype=np.int64,
    )
    return arr.T


def fit_scalers(pairs, train_idx: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    """Fit scalers on the training split only."""
    if len(train_idx) == 0:
        raise ValueError("Training split is empty, cannot fit StandardScaler.")

    print("  Incrementally fitting one scaler set on training pairs only...")
    train_set = set(map(int, train_idx))
    attr_scaler, flow_scaler = StandardScaler(), StandardScaler()
    for idx, pair in enumerate(pairs):
        if idx not in train_set:
            continue
        for graph, edges, flows in (("G", "edge_list_old", "flows_old"),
                                    ("G_prime", "edge_list_new", "flows_new")):
            attr_scaler.partial_fit(extract_edge_attrs(pair[graph], pair[edges]))
            flow_scaler.partial_fit(pair[flows].reshape(-1, 1))

    print(f"  attr_scaler mean: {attr_scaler.mean_}")
    print(f"  attr_scaler std:  {attr_scaler.scale_}")
    print(f"  flow_scaler mean: {flow_scaler.mean_[0]:.2f}")
    print(f"  flow_scaler std:  {flow_scaler.scale_[0]:.2f}")

    return attr_scaler, flow_scaler


def save_scalers(attr_scaler: StandardScaler, flow_scaler: StandardScaler, output_dir: str) -> None:
    """Persist fitted scalers."""
    scalers_dir = os.path.join(output_dir, "scalers")
    os.makedirs(scalers_dir, exist_ok=True)

    with open(os.path.join(scalers_dir, "attr_scaler.pkl"), "wb") as handle:
        pickle.dump(attr_scaler, handle)
    with open(os.path.join(scalers_dir, "flow_scaler.pkl"), "wb") as handle:
        pickle.dump(flow_scaler, handle)

    print(f"  Scaler saved to: {scalers_dir}/")


def build_single_data_object(pair: dict, attr_scaler: StandardScaler, flow_scaler: StandardScaler) -> Data:
    """Convert one solved pair into a dynamic-size PyG Data object."""
    node_ids = tuple(int(node_id) for node_id in _require_pair_metadata(pair, "node_ids"))
    centroid_nodes = tuple(int(node_id) for node_id in _require_pair_metadata(pair, "centroid_nodes"))
    network_name = str(_require_pair_metadata(pair, "network_name"))

    num_nodes = len(node_ids)
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    centroid_set = set(centroid_nodes)
    od_dim = int(pair["od_matrix"].shape[0]) if isinstance(pair.get("od_matrix"), np.ndarray) else len(centroid_nodes)

    raw_attr_old = extract_edge_attrs(pair["G"], pair["edge_list_old"])
    raw_attr_new = extract_edge_attrs(pair["G_prime"], pair["edge_list_new"])
    norm_attr_old = attr_scaler.transform(raw_attr_old).astype(np.float32)
    norm_attr_new = attr_scaler.transform(raw_attr_new).astype(np.float32)

    norm_flow_old = flow_scaler.transform(pair["flows_old"].reshape(-1, 1)).astype(np.float32)
    norm_flow_new = flow_scaler.transform(pair["flows_new"].reshape(-1, 1)).astype(np.float32)

    edge_index_old = edge_list_to_index(pair["edge_list_old"], node_id_to_index)
    edge_index_new = edge_list_to_index(pair["edge_list_new"], node_id_to_index)

    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    non_centroid_mask = torch.tensor(
        [node_id not in centroid_set for node_id in node_ids],
        dtype=torch.bool,
    )

    src_old = np.array([node_id_to_index[u] for u, _ in pair["edge_list_old"]], dtype=np.int64)
    dst_old = np.array([node_id_to_index[v] for _, v in pair["edge_list_old"]], dtype=np.int64)
    flows_old_real = pair["flows_old"].reshape(-1).astype(np.float64)
    net_demand_np = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(net_demand_np, dst_old, flows_old_real)
    np.add.at(net_demand_np, src_old, -flows_old_real)
    net_demand = torch.from_numpy(net_demand_np.astype(np.float32))

    old_edge_set = set(tuple(edge) for edge in pair["edge_list_old"])
    new_edge_mask = torch.tensor(
        [tuple(edge) not in old_edge_set for edge in pair["edge_list_new"]],
        dtype=torch.bool,
    )

    data = Data(
        x=x,
        edge_index_old=torch.from_numpy(edge_index_old).long(),
        edge_attr_old=torch.from_numpy(norm_attr_old),
        flow_old=torch.from_numpy(norm_flow_old),
        edge_index_new=torch.from_numpy(edge_index_new).long(),
        edge_attr_new=torch.from_numpy(norm_attr_new),
        y=torch.from_numpy(norm_flow_new),
        non_centroid_mask=non_centroid_mask,
        net_demand=net_demand,
        new_edge_mask=new_edge_mask,
        node_ids=torch.tensor(node_ids, dtype=torch.long),
        num_nodes=num_nodes,
        num_edges_old=len(pair["edge_list_old"]),
        num_edges_new=len(pair["edge_list_new"]),
        centroid_count=len(centroid_nodes),
        centroid_pos=torch.tensor([[node_id_to_index[n] for n in centroid_nodes]], dtype=torch.long),
        free_flow_time_new=torch.tensor(
            [[pair["G_prime"][u][v]["free_flow_time"]] for u, v in pair["edge_list_new"]],
            dtype=torch.float32),
        first_thru_node=int(pair["G_prime"].graph.get("first_thru_node", 1)),
        od_dim=od_dim,
        mutation_type=pair["mutation_type"],
        network_name=network_name,
    )
    if "global_sample_idx" in pair:
        data.sample_idx = int(pair["global_sample_idx"])
        data.source_sample_idx = int(pair["source_sample_idx"])
        data.source_shard = pair["source_shard"]
        data.source_pair_offset = int(pair["source_pair_offset"])
    return data


def build_full_dataset(
    pairs,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    attr_scaler: StandardScaler,
    flow_scaler: StandardScaler,
    output_dir=None,
    od_stats=None,
) -> tuple[list, list, list]:
    """Build Data lists for train/val/test splits."""

    indices = (train_idx, val_idx, test_idx)
    datasets = [[None] * len(idx) for idx in indices]
    destinations = {int(idx): (split, position)
                    for split, arr in enumerate(indices) for position, idx in enumerate(arr)}
    od_maps = None
    positive_sum, positive_count = 0.0, 0
    for idx, pair in enumerate(tqdm(pairs, total=len(destinations), desc="  Building all splits")):
        data = build_single_data_object(pair, attr_scaler, flow_scaler)
        validate_single_data_object(data)
        split, position = destinations[idx]
        datasets[split][position] = data
        if output_dir is not None:
            od = np.asarray(pair["od_matrix"], dtype=np.float64)
            c = int(data.centroid_count)
            if od.shape != (c, c) or not np.isfinite(od).all() or np.any(od < 0):
                raise ValueError("Invalid OD sidecar sample.")
            if od_maps is None:
                od_maps = [np.lib.format.open_memmap(
                    Path(output_dir) / f"{name}_od.npy", mode="w+", dtype=np.float32,
                    shape=(len(arr), c, c))
                    for name, arr in zip(("train", "val", "test"), indices)]
            od_maps[split][position] = od
            if split == 0:
                positive = od[od > 0]
                positive_sum += float(positive.sum())
                positive_count += int(positive.size)
    if od_maps is not None:
        for mapping in od_maps:
            mapping.flush()
        del mapping, od_maps
    if od_stats is not None:
        od_stats["od_scale"] = positive_sum / positive_count if positive_count else 1.0
        od_stats["od_positive_count_train"] = positive_count
    if any(data is None for dataset in datasets for data in dataset):
        raise ValueError("Input samples changed between scan and conversion.")
    return tuple(datasets)


def split_indices(
    num_samples: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/val/test split indices with small-sample protection."""
    if num_samples <= 0:
        raise ValueError("No valid samples found in the input pickle.")
    if not (0 < train_ratio <= 1 and 0 <= val_ratio <= 1
            and train_ratio + val_ratio <= 1):
        raise ValueError("Require 0 < train_ratio <= 1, val_ratio >= 0, and sum <= 1.")

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


def print_dataset_stats(train_dataset: list, val_dataset: list, test_dataset: list) -> None:
    """Print summary statistics for each split."""

    def _stats_one_split(dataset: list, name: str) -> None:
        if len(dataset) == 0:
            print(f"\n  [{name}] 0 samples")
            print("    Split is empty, skip statistics")
            return

        e_old = [int(data.num_edges_old) for data in dataset]
        e_new = [int(data.num_edges_new) for data in dataset]
        count, total, total_sq = 0, 0.0, 0.0
        y_min, y_max = float("inf"), -float("inf")
        for data in dataset:
            values = data.y.detach().double()
            count += values.numel()
            total += float(values.sum())
            total_sq += float(values.square().sum())
            y_min = min(y_min, float(values.min()))
            y_max = max(y_max, float(values.max()))
        mean = total / count
        std = np.sqrt(max(total_sq / count - mean ** 2, 0.0))
        mutation_dist = Counter(data.mutation_type for data in dataset)

        print(f"\n  [{name}] {len(dataset)} samples")
        print(f"    G  edges: fixed {e_old[0]}")
        print(f"    G' edges: min={min(e_new)}, max={max(e_new)}, mean={np.mean(e_new):.1f}")
        print(
            f"    y (normalized flows_new): min={y_min:.3f}, "
            f"max={y_max:.3f}, mean={mean:.3f}, std={std:.3f}"
        )
        print(f"    Mutation type dist: {dict(mutation_dist)}")

    print(f"\n{'=' * 60}")
    print("Dataset Statistics Summary")
    print(f"{'=' * 60}")
    _stats_one_split(train_dataset, "Train")
    _stats_one_split(val_dataset, "Val")
    _stats_one_split(test_dataset, "Test")


def validate_single_data_object(data: Data) -> None:
    """Validate dynamic-size fields and value ranges for one Data object."""
    num_nodes = int(data.num_nodes)
    e_old = int(data.num_edges_old)
    e_new = int(data.num_edges_new)

    assert data.x.shape == (num_nodes, 1), f"x shape error: {data.x.shape}"
    assert data.edge_index_old.shape == (2, e_old), "edge_index_old shape error"
    assert data.edge_index_new.shape == (2, e_new), "edge_index_new shape error"
    assert data.edge_attr_old.shape == (e_old, 3), "edge_attr_old shape error"
    assert data.flow_old.shape == (e_old, 1), "flow_old shape error"
    assert data.edge_attr_new.shape == (e_new, 3), "edge_attr_new shape error"
    assert data.y.shape == (e_new, 1), "y shape error"
    assert data.non_centroid_mask.shape == (num_nodes,), "non_centroid_mask shape error"
    assert data.net_demand.shape == (num_nodes,), f"net_demand shape error: {data.net_demand.shape}"
    assert data.new_edge_mask.shape == (e_new,), "new_edge_mask shape error"

    for name, edge_index in [("edge_index_old", data.edge_index_old), ("edge_index_new", data.edge_index_new)]:
        if edge_index.numel() == 0:
            continue
        assert edge_index.min() >= 0, f"{name} has negative node id"
        assert edge_index.max() < num_nodes, f"{name} node id out of range (max={edge_index.max()})"

    for name, tensor in [
        ("edge_attr_old", data.edge_attr_old),
        ("flow_old", data.flow_old),
        ("edge_attr_new", data.edge_attr_new),
        ("y", data.y),
        ("net_demand", data.net_demand),
    ]:
        assert not torch.isnan(tensor).any(), f"{name} contains NaN"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf"

    assert torch.all(data.x == 1.0), "x is not all-ones placeholder."
    assert data.non_centroid_mask.dtype == torch.bool, "non_centroid_mask must be bool"
    assert data.new_edge_mask.dtype == torch.bool, "new_edge_mask must be bool"
    assert data.net_demand.dtype == torch.float32, f"net_demand dtype should be float32, got {data.net_demand.dtype}"

    if data.non_centroid_mask.any():
        non_centroid_residual = data.net_demand[data.non_centroid_mask].abs().max().item()
        assert non_centroid_residual < 1.0, (
            f"net_demand on non-centroid nodes exceeds tolerance: {non_centroid_residual:.4f} veh/hr"
        )


def save_dataset_metadata(
    output_dir: str,
    example_data: Data,
    train_dataset: list,
    train_size: int,
    val_size: int,
    test_size: int,
    conversion_metadata: dict = None,
) -> None:
    """Persist metadata used by loaders and downstream configs."""
    node_ids = example_data.node_ids.tolist() if hasattr(example_data, "node_ids") else []
    centroid_nodes = [
        node_id
        for node_id, is_non_centroid in zip(node_ids, example_data.non_centroid_mask.tolist())
        if not is_non_centroid
    ]
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty, cannot write dataset metadata.")

    metadata = {
        "network_name": str(getattr(example_data, "network_name", "Unknown")),
        "num_nodes": int(example_data.num_nodes),
        "num_edges_old": int(example_data.num_edges_old),
        "num_edges_new": int(example_data.num_edges_new),
        "od_dim": int(getattr(example_data, "od_dim", 0)),
        "centroid_count": int(getattr(example_data, "centroid_count", 0)),
        "node_ids": node_ids,
        "centroid_nodes": centroid_nodes,
        "splits": {
            "train": int(train_size),
            "val": int(val_size),
            "test": int(test_size),
        },
        "files": {
            "train": "train_dataset.pt",
            "val": "val_dataset.pt",
            "test": "test_dataset.pt",
            "attr_scaler": "scalers/attr_scaler.pkl",
            "flow_scaler": "scalers/flow_scaler.pkl",
        },
    }

    if conversion_metadata:
        metadata["conversion"] = conversion_metadata
        metadata["od_scale"] = conversion_metadata["od_scale"]
        metadata["od_sidecar_version"] = 1
        for split in ("train", "val", "test"):
            metadata["files"][f"{split}_od"] = f"{split}_od.npy"
        metadata["files"]["sample_sources"] = "sample_sources.json"
        metadata["files"]["split_indices"] = "split_indices.npz"
    meta_path = os.path.join(output_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"  Dataset metadata saved: {meta_path}")


def save_split_indices(output_dir: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Persist the exact split indices for downstream analysis."""
    split_path = os.path.join(output_dir, "split_indices.npz")
    np.savez(
        split_path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )
    print(f"  Split indices saved:   {split_path}")


def _print_data_summary(data: Data, split_name: str) -> None:
    """Print a concise sample summary."""
    print(f"\n  [{split_name}] First sample field summary:")
    print(f"    x                  : {tuple(data.x.shape)}")
    print(f"    edge_index_old     : {tuple(data.edge_index_old.shape)}")
    print(f"    edge_attr_old      : {tuple(data.edge_attr_old.shape)}")
    print(f"    flow_old           : {tuple(data.flow_old.shape)}")
    print(f"    edge_index_new     : {tuple(data.edge_index_new.shape)}")
    print(f"    edge_attr_new      : {tuple(data.edge_attr_new.shape)}")
    print(f"    y                  : {tuple(data.y.shape)}")
    print(
        f"    non_centroid_mask  : {tuple(data.non_centroid_mask.shape)} "
        f"(True count: {int(data.non_centroid_mask.sum())})"
    )
    net_demand = data.net_demand
    centroid_mask = ~data.non_centroid_mask
    centroid_values = net_demand[centroid_mask]
    non_centroid_values = net_demand[data.non_centroid_mask]
    print(f"    net_demand         : {tuple(net_demand.shape)}  (from flow divergence, no OD features)")
    if centroid_values.numel() > 0:
        print(
            f"      centroid nodes   : count={centroid_values.numel()}, "
            f"range=[{centroid_values.min():.1f}, {centroid_values.max():.1f}]"
        )
    else:
        print("      centroid nodes   : count=0")
    if non_centroid_values.numel() > 0:
        print(
            f"      non-centroids    : count={non_centroid_values.numel()}, "
            f"max_abs={non_centroid_values.abs().max():.4f}"
        )
    else:
        print("      non-centroids    : count=0")
    print(f"    network_name       : {getattr(data, 'network_name', 'Unknown')}")
    print(f"    num_nodes          : {int(data.num_nodes)}")
    print(f"    od_dim             : {int(getattr(data, 'od_dim', 0))}")
    print(f"    mutation_type      : {data.mutation_type}")


def _save_split(dataset: list, path: str, name: str) -> None:
    """Save one split to a .pt file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(dataset, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  {name:6s}: {len(dataset):5d} samples -> {path} ({size_mb:.1f} MB)")


def run(args) -> None:
    """Execute the full dataset build workflow."""
    output_path = Path(args.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("Output directory is not empty; use a new directory to avoid mixing exports.")

    print(f"\n{'=' * 60}")
    print("Step 1 - Scan verified (G, G') shards (one pickle at a time)")
    print(f"{'=' * 60}")
    pairs = ShardedPairs(args.input_pkl, getattr(args, "input_dir", None))
    sources, model = scan_pair_sources(pairs)
    num_samples = len(sources)
    expected = getattr(args, "expected_samples", None)
    if expected is not None and num_samples != expected:
        raise ValueError(f"Expected {expected} valid pairs, found {num_samples}. Finish generation first.")
    print(f"  Network: {sources[0]['network_name']}; shards: {len(pairs.paths)}")
    print(f"  Number of valid network pairs: {num_samples}")
    print("    NOTE: net_demand will be derived from flows_old divergence (no OD features)")

    print(f"\n{'=' * 60}")
    print("Step 2 - Split Train / Val / Test indices")
    print(f"{'=' * 60}")
    train_idx, val_idx, test_idx = split_indices(
        num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    split_file = getattr(args, "split_indices", None)
    if split_file:
        with np.load(split_file) as saved:
            train_idx, val_idx, test_idx = [saved[f"{name}_idx"].copy() for name in ("train", "val", "test")]
        combined = np.concatenate((train_idx, val_idx, test_idx))
        if combined.dtype.kind not in "iu" or not np.array_equal(np.sort(combined), np.arange(num_samples)):
            raise ValueError("Saved split indices must partition the exact current global sample IDs.")
        source_file = Path(split_file).parent / "sample_sources.json"
        with source_file.open(encoding="utf-8") as handle:
            previous_sources = json.load(handle)
        if len(previous_sources) != len(sources) or any(
            any(old.get(key) != new[key] for key in new)
            for old, new in zip(previous_sources, sources)
        ):
            raise ValueError("Saved split belongs to different source samples.")
    print(f"  Train : {len(train_idx)} samples ({len(train_idx) / num_samples * 100:.1f}%)")
    print(f"  Val   : {len(val_idx)} samples ({len(val_idx) / num_samples * 100:.1f}%)")
    print(f"  Test  : {len(test_idx)} samples ({len(test_idx) / num_samples * 100:.1f}%)")
    os.makedirs(args.output_dir, exist_ok=True)
    save_split_indices(args.output_dir, train_idx, val_idx, test_idx)
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for position, idx in enumerate(indices):
            sources[int(idx)].update(split=name, split_position=position)
    with (output_path / "sample_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(sources, handle, indent=2)

    print(f"\n{'=' * 60}")
    print("Step 3 - Fit StandardScaler (training set only)")
    print(f"{'=' * 60}")
    attr_scaler, flow_scaler = fit_scalers(pairs, train_idx)
    save_scalers(attr_scaler, flow_scaler, args.output_dir)

    print(f"\n{'=' * 60}")
    print("Step 4 - Build PyG Data objects")
    print(f"{'=' * 60}")
    od_stats = {}
    train_dataset, val_dataset, test_dataset = build_full_dataset(
        pairs,
        train_idx,
        val_idx,
        test_idx,
        attr_scaler,
        flow_scaler,
        output_dir=args.output_dir,
        od_stats=od_stats,
    )

    print(f"\n{'=' * 60}")
    print("Step 5 - Validate sample correctness")
    print(f"{'=' * 60}")
    for name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        if len(dataset) == 0:
            print(f"  [{name}] Split is empty, skip sample validation")
            continue
        validate_single_data_object(dataset[0])
        print(f"  [{name}] First sample passed validation")
        _print_data_summary(dataset[0], name)

    print_dataset_stats(train_dataset, val_dataset, test_dataset)

    print(f"\n{'=' * 60}")
    print("Step 6 - Save dataset")
    print(f"{'=' * 60}")
    _save_split(train_dataset, os.path.join(args.output_dir, "train_dataset.pt"), "Train")
    _save_split(val_dataset, os.path.join(args.output_dir, "val_dataset.pt"), "Val")
    _save_split(test_dataset, os.path.join(args.output_dir, "test_dataset.pt"), "Test")
    # Write the loader's completion marker only after all three splits are saved.
    save_dataset_metadata(
        output_dir=args.output_dir,
        example_data=train_dataset[0],
        train_dataset=train_dataset,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        conversion_metadata={
            **od_stats,
            "input_root": str(pairs.root),
            "input_shards": [path.relative_to(pairs.root).as_posix() for path in pairs.paths],
            "seed": args.seed, "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
            "scaler_fit_split": "train", "scaler_fit_sides": ["old", "new"],
            "sue_model": model,
            "sample_idx_scope": "global within this network export; see sample_sources.json",
            "reused_split_indices": str(Path(split_file).resolve()) if split_file else None,
        },
    )

    print(f"\n  All datasets saved to: {args.output_dir}/")
    print(f"  Scaler saved to:       {args.output_dir}/scalers/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build PyG dataset for solved traffic network pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--input_pkl",
        type=str,
        default="processed_data/pairs/network_pairs_dataset.pkl",
        help="Path to the pickle output by solve_network_pairs.py",
    )
    inputs.add_argument(
        "--input_dir", type=str,
        help="One completed network's root containing batch_*/network_pairs_dataset.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data/pyg",
        help="Output directory for .pt files, scaler files, and dataset metadata",
    )
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio (test = 1 - train - val)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--expected_samples", type=int,
                        help="Require this many valid pairs; prevents exporting unfinished runs")
    parser.add_argument("--split_indices", type=str,
                        help="Reuse an existing split_indices.npz (and adjacent sample_sources.json)")
    return parser.parse_args()


def main():
    print("\n" + "=" * 60)
    print("  Build PyG Network Pair Dataset")
    print("=" * 60)
    print(f"  Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    print(f"\n  Run arguments:")
    for key, value in vars(args).items():
        print(f"    {key:20s}: {value}")

    try:
        run(args)
    except KeyboardInterrupt:
        print("\n\n  User interrupted, exiting.")
        sys.exit(1)
    except Exception as exc:
        import traceback

        print(f"\n\n  [Error] Dataset construction failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
