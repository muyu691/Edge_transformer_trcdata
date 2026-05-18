from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from baseline.common import compute_new_edge_mask


@dataclass
class DatasetBundle:
    dataset_dir: Path
    train_data: list
    val_data: list
    test_data: list
    flow_mean: float
    flow_std: float
    metadata: dict


def canonical_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    aliases = {
        "ema": "ema",
        "siouxfalls": "siouxfalls",
        "sioux_falls": "siouxfalls",
        "sioux-falls": "siouxfalls",
        "sioux falls": "siouxfalls",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Choose from: ema, siouxfalls.")
    return aliases[key]


def _require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _load_pt_list(path: Path) -> list:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of PyG Data objects in {path}, got {type(data)}.")
    return data


def _load_flow_scaler_stats(dataset_dir: Path) -> tuple[float, float]:
    scaler_path = _require_path(dataset_dir / "scalers" / "flow_scaler.pkl", "flow scaler")
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise AttributeError(f"flow_scaler.pkl at {scaler_path} does not expose mean_/scale_.")
    return float(scaler.mean_[0]), float(max(scaler.scale_[0], 1e-6))


def _prepare_data_object(data) -> None:
    if getattr(data, "x", None) is None:
        num_nodes = int(data.num_nodes)
        data.x = torch.ones((num_nodes, 1), dtype=torch.float32)

    required = (
        "edge_index_old",
        "edge_attr_old",
        "flow_old",
        "edge_index_new",
        "edge_attr_new",
        "y",
    )
    missing = [name for name in required if getattr(data, name, None) is None]
    if missing:
        raise ValueError(f"Data object is missing required fields: {missing}")

    if getattr(data, "new_edge_mask", None) is None:
        data.new_edge_mask = compute_new_edge_mask(
            edge_index_old=data.edge_index_old,
            edge_index_new=data.edge_index_new,
            total_nodes=int(data.num_nodes),
        ).cpu()

    data.x = data.x.float()
    data.edge_attr_old = data.edge_attr_old.float()
    data.flow_old = data.flow_old.float()
    data.edge_attr_new = data.edge_attr_new.float()
    data.y = data.y.float()
    data.new_edge_mask = data.new_edge_mask.bool()


def _prepare_split(dataset: list) -> list:
    for data in dataset:
        _prepare_data_object(data)
    return dataset


def resolve_dataset_dir(
    dataset_dir: str | None = None,
    dataset_name: str | None = None,
    processed_root: str | None = None,
) -> Path:
    if dataset_dir:
        return Path(dataset_dir).resolve()

    if not dataset_name:
        raise ValueError("Either dataset_dir or dataset_name must be provided.")

    canonical = canonical_dataset_name(dataset_name)
    root = Path(processed_root or "create_sioux_data/processed_data").resolve()
    _require_path(root, "processed root")

    candidate_map = {
        "ema": [
            "ema_pyg_newpolicy_lhs",
            "ema_pyg_dataset",
            "pyg_dataset",
        ],
        "siouxfalls": [
            "siouxfalls_pyg_newpolicy_lhs",
            "siouxfalls_pyg_dataset",
            "siouxfalls_pyg",
        ],
    }
    for name in candidate_map[canonical]:
        candidate = root / name
        if (candidate / "train_dataset.pt").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset_name}' under processed_root={root}. "
        f"Tried: {candidate_map[canonical]}"
    )


def load_dataset_bundle(
    dataset_dir: str | None = None,
    dataset_name: str | None = None,
    processed_root: str | None = None,
) -> DatasetBundle:
    root = resolve_dataset_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        processed_root=processed_root,
    )
    _require_path(root, "dataset directory")

    metadata_path = root / "dataset_meta.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    train_data = _prepare_split(_load_pt_list(_require_path(root / "train_dataset.pt", "train split")))
    val_data = _prepare_split(_load_pt_list(_require_path(root / "val_dataset.pt", "validation split")))
    test_data = _prepare_split(_load_pt_list(_require_path(root / "test_dataset.pt", "test split")))
    flow_mean, flow_std = _load_flow_scaler_stats(root)

    return DatasetBundle(
        dataset_dir=root,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        flow_mean=flow_mean,
        flow_std=flow_std,
        metadata=metadata,
    )


def build_loader(dataset: list, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
