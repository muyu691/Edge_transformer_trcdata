#!/usr/bin/env python3
"""Verify that regenerated raw pairs exactly reproduce an existing PyG dataset."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare regenerated raw network pairs against every sample in an "
            "existing normalized PyG dataset."
        )
    )
    parser.add_argument("--input-pkl", required=True, help="Recovered network_pairs_dataset.pkl")
    parser.add_argument("--pyg-dir", required=True, help="Existing PyG dataset directory")
    parser.add_argument("--expected-network", default="EMA")
    parser.add_argument("--expected-pairs", type=int, default=10000)
    parser.add_argument("--flow-atol", type=float, default=0.25)
    parser.add_argument("--attr-atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default="",
        help="Validation report path (default: beside input pickle)",
    )
    parser.add_argument(
        "--success-marker",
        default="",
        help="Marker created only after all samples pass validation",
    )
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_raw_pairs(path: Path) -> tuple[list, list[int]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if "pairs" not in payload:
            raise ValueError(f"Raw pickle has no 'pairs' field: {path}")
        return payload["pairs"], list(payload.get("failed_indices", []))
    if isinstance(payload, list):
        return payload, []
    raise TypeError(f"Unsupported raw pickle payload: {type(payload)!r}")


def edge_attrs(graph, edge_list: list) -> np.ndarray:
    return np.asarray(
        [
            [
                graph[u][v]["capacity"],
                graph[u][v]["speed"],
                graph[u][v]["length"],
            ]
            for u, v in edge_list
        ],
        dtype=np.float64,
    )


def edge_index(edge_list: list, node_ids: tuple[int, ...]) -> np.ndarray:
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    return np.asarray(
        [[node_to_index[u], node_to_index[v]] for u, v in edge_list],
        dtype=np.int64,
    ).T


def inverse_standardized(values: np.ndarray, scaler) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * np.asarray(scaler.scale_, dtype=np.float64) + np.asarray(
        scaler.mean_, dtype=np.float64
    )


def raw_net_demand(pair: dict, node_ids: tuple[int, ...]) -> np.ndarray:
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    demand = np.zeros(len(node_ids), dtype=np.float64)
    flows = np.asarray(pair["flows_old"], dtype=np.float64).reshape(-1)
    for (u, v), flow in zip(pair["edge_list_old"], flows):
        demand[node_to_index[u]] -= flow
        demand[node_to_index[v]] += flow
    return demand


class ValidationRecorder:
    def __init__(self, max_examples: int) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict] = []
        self.max_examples = max_examples
        self.max_abs_error = {
            "edge_attr_old": 0.0,
            "edge_attr_new": 0.0,
            "flow_old": 0.0,
            "flow_new": 0.0,
            "net_demand": 0.0,
        }

    def mismatch(self, category: str, split: str, split_position: int, raw_index: int, detail: str) -> None:
        self.counts[category] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(
                {
                    "category": category,
                    "split": split,
                    "split_position": int(split_position),
                    "raw_index": int(raw_index),
                    "detail": detail,
                }
            )

    def compare_array(
        self,
        category: str,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        split: str,
        split_position: int,
        raw_index: int,
        rtol: float,
        atol: float,
    ) -> None:
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if actual.shape != expected.shape:
            self.mismatch(
                category,
                split,
                split_position,
                raw_index,
                f"shape {actual.shape} != {expected.shape}",
            )
            return
        if actual.size:
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            if category in self.max_abs_error:
                self.max_abs_error[category] = max(self.max_abs_error[category], error)
        if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
            self.mismatch(
                category,
                split,
                split_position,
                raw_index,
                f"max_abs_error={error:.6g}, rtol={rtol:g}, atol={atol:g}",
            )


def validate_pair(
    pair: dict,
    data,
    *,
    split: str,
    split_position: int,
    raw_index: int,
    attr_scaler,
    flow_scaler,
    expected_network: str,
    rtol: float,
    attr_atol: float,
    flow_atol: float,
    recorder: ValidationRecorder,
) -> None:
    node_ids = tuple(int(value) for value in pair.get("node_ids", sorted(pair["G"].nodes())))
    pyg_node_ids = tuple(int(value) for value in data.node_ids.detach().cpu().tolist())

    raw_network = str(pair.get("network_name", ""))
    pyg_network = str(getattr(data, "network_name", ""))
    if raw_network.lower() != expected_network.lower() or pyg_network.lower() != expected_network.lower():
        recorder.mismatch(
            "network_name", split, split_position, raw_index,
            f"raw={raw_network!r}, pyg={pyg_network!r}, expected={expected_network!r}",
        )
    if node_ids != pyg_node_ids:
        recorder.mismatch(
            "node_ids", split, split_position, raw_index,
            f"raw length={len(node_ids)}, pyg length={len(pyg_node_ids)}",
        )

    old_edges = list(pair["edge_list_old"])
    new_edges = list(pair["edge_list_new"])
    recorder.compare_array(
        "edge_index_old",
        data.edge_index_old.detach().cpu().numpy(),
        edge_index(old_edges, node_ids),
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=0.0, atol=0.0,
    )
    recorder.compare_array(
        "edge_index_new",
        data.edge_index_new.detach().cpu().numpy(),
        edge_index(new_edges, node_ids),
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=0.0, atol=0.0,
    )

    raw_attr_old = edge_attrs(pair["G"], old_edges)
    raw_attr_new = edge_attrs(pair["G_prime"], new_edges)
    recovered_attr_old = inverse_standardized(
        data.edge_attr_old.detach().cpu().numpy(), attr_scaler
    )
    recovered_attr_new = inverse_standardized(
        data.edge_attr_new.detach().cpu().numpy(), attr_scaler
    )
    recorder.compare_array(
        "edge_attr_old", recovered_attr_old, raw_attr_old,
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=rtol, atol=attr_atol,
    )
    recorder.compare_array(
        "edge_attr_new", recovered_attr_new, raw_attr_new,
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=rtol, atol=attr_atol,
    )

    recovered_flow_old = inverse_standardized(
        data.flow_old.detach().cpu().numpy(), flow_scaler
    ).reshape(-1)
    recovered_flow_new = inverse_standardized(
        data.y.detach().cpu().numpy(), flow_scaler
    ).reshape(-1)
    recorder.compare_array(
        "flow_old", recovered_flow_old, np.asarray(pair["flows_old"]).reshape(-1),
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=rtol, atol=flow_atol,
    )
    recorder.compare_array(
        "flow_new", recovered_flow_new, np.asarray(pair["flows_new"]).reshape(-1),
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=rtol, atol=flow_atol,
    )

    old_edge_set = set(old_edges)
    expected_new_mask = np.asarray([edge not in old_edge_set for edge in new_edges], dtype=bool)
    recorder.compare_array(
        "new_edge_mask", data.new_edge_mask.detach().cpu().numpy(), expected_new_mask,
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=0.0, atol=0.0,
    )
    recorder.compare_array(
        "net_demand",
        data.net_demand.detach().cpu().numpy(),
        raw_net_demand(pair, node_ids),
        split=split, split_position=split_position, raw_index=raw_index,
        rtol=rtol, atol=flow_atol,
    )

    if str(pair.get("mutation_type", "")) != str(getattr(data, "mutation_type", "")):
        recorder.mismatch(
            "mutation_type", split, split_position, raw_index,
            f"raw={pair.get('mutation_type')!r}, pyg={getattr(data, 'mutation_type', None)!r}",
        )


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_pkl).resolve()
    pyg_dir = Path(args.pyg_dir).resolve()
    report_path = Path(args.output_json).resolve() if args.output_json else input_path.with_name(
        "raw_pairs_validation.json"
    )
    marker_path = Path(args.success_marker).resolve() if args.success_marker else input_path.with_name(
        "RAW_PAIRS_MATCH_PYG.ok"
    )
    marker_path.unlink(missing_ok=True)

    required = [
        input_path,
        pyg_dir / "split_indices.npz",
        pyg_dir / "scalers" / "attr_scaler.pkl",
        pyg_dir / "scalers" / "flow_scaler.pkl",
    ] + [pyg_dir / f"{split}_dataset.pt" for split in SPLITS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing validation inputs:\n  " + "\n  ".join(missing))

    print(f"Loading raw pairs: {input_path}")
    pairs, failed_indices = load_raw_pairs(input_path)
    print(f"Raw pairs: {len(pairs)}, failed indices: {len(failed_indices)}")

    with (pyg_dir / "scalers" / "attr_scaler.pkl").open("rb") as handle:
        attr_scaler = pickle.load(handle)
    with (pyg_dir / "scalers" / "flow_scaler.pkl").open("rb") as handle:
        flow_scaler = pickle.load(handle)

    split_payload = np.load(pyg_dir / "split_indices.npz")
    split_indices = {
        split: np.asarray(split_payload[f"{split}_idx"], dtype=np.int64)
        for split in SPLITS
    }
    all_indices = np.concatenate([split_indices[split] for split in SPLITS])
    recorder = ValidationRecorder(args.max_examples)

    if len(pairs) != args.expected_pairs:
        recorder.mismatch(
            "pair_count", "global", -1, -1,
            f"raw={len(pairs)}, expected={args.expected_pairs}",
        )
    if failed_indices:
        recorder.mismatch(
            "failed_indices", "global", -1, -1,
            f"count={len(failed_indices)}, first={failed_indices[:10]}",
        )
    expected_permutation = np.arange(args.expected_pairs, dtype=np.int64)
    if len(all_indices) != args.expected_pairs or not np.array_equal(
        np.sort(all_indices), expected_permutation
    ):
        recorder.mismatch(
            "split_indices", "global", -1, -1,
            f"combined split length={len(all_indices)}; not a permutation of 0..{args.expected_pairs - 1}",
        )

    checked = 0
    split_sizes: dict[str, int] = {}
    for split in SPLITS:
        dataset_path = pyg_dir / f"{split}_dataset.pt"
        print(f"Loading {split} split: {dataset_path}")
        dataset = torch_load(dataset_path)
        indices = split_indices[split]
        split_sizes[split] = len(dataset)
        if len(dataset) != len(indices):
            recorder.mismatch(
                "split_length", split, -1, -1,
                f"dataset={len(dataset)}, indices={len(indices)}",
            )

        comparable = min(len(dataset), len(indices))
        for position in range(comparable):
            raw_index = int(indices[position])
            if raw_index < 0 or raw_index >= len(pairs):
                recorder.mismatch(
                    "raw_index_range", split, position, raw_index,
                    f"valid range is 0..{len(pairs) - 1}",
                )
                continue
            validate_pair(
                pairs[raw_index],
                dataset[position],
                split=split,
                split_position=position,
                raw_index=raw_index,
                attr_scaler=attr_scaler,
                flow_scaler=flow_scaler,
                expected_network=args.expected_network,
                rtol=args.rtol,
                attr_atol=args.attr_atol,
                flow_atol=args.flow_atol,
                recorder=recorder,
            )
            checked += 1
            if checked % 1000 == 0:
                print(f"Validated {checked}/{args.expected_pairs} samples")

        del dataset
        gc.collect()

    passed = checked == args.expected_pairs and not recorder.counts
    report = {
        "passed": passed,
        "checked_pairs": checked,
        "expected_pairs": args.expected_pairs,
        "raw_pair_count": len(pairs),
        "failed_raw_indices": len(failed_indices),
        "split_sizes": split_sizes,
        "mismatch_counts": dict(sorted(recorder.counts.items())),
        "mismatch_examples": recorder.examples,
        "max_abs_error": recorder.max_abs_error,
        "tolerances": {
            "rtol": args.rtol,
            "flow_atol": args.flow_atol,
            "attr_atol": args.attr_atol,
        },
        "input_pkl": str(input_path),
        "pyg_dir": str(pyg_dir),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(report_path, report)

    print("=" * 72)
    print(f"Validation passed : {passed}")
    print(f"Checked pairs     : {checked}/{args.expected_pairs}")
    print(f"Mismatch counts   : {dict(recorder.counts)}")
    print(f"Report            : {report_path}")
    if passed:
        marker_path.write_text(
            f"validated={datetime.now(timezone.utc).isoformat()}\n"
            f"input_pkl={input_path}\npyg_dir={pyg_dir}\n",
            encoding="utf-8",
        )
        print(f"Success marker    : {marker_path}")
        return 0

    print("The regenerated raw pairs do not reproduce the existing PyG dataset.", file=sys.stderr)
    print("Do not use this pickle with predictions produced from that PyG dataset.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
