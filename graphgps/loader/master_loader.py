import json
import logging
import os
import os.path as osp
import pickle

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader

from graphgps.loader.dataset.network_pairs_topology import NetworkPairsTopologyDataset
from graphgps.loader.split_generator import prepare_splits, set_dataset_splits
from graphgps.transform.transforms import MaskEdgeFeatureTransform, pre_transform_in_memory


def _canonical_network_name(network_name: str) -> str:
    return (network_name or "").strip().lower()


def _is_network_pairs_processed_dir(path: str) -> bool:
    if not path or not osp.isdir(path):
        return False
    required = ("train_dataset.pt", "val_dataset.pt", "test_dataset.pt", "dataset_meta.json")
    return all(osp.exists(osp.join(path, name)) for name in required)


def _load_metadata(path: str) -> dict:
    metadata_path = osp.join(path, "dataset_meta.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_dirs(dataset_dir: str, processed_root: str):
    seen = set()
    for root in [dataset_dir, processed_root]:
        if not root:
            continue
        root_abs = osp.abspath(root)
        if root_abs not in seen:
            seen.add(root_abs)
            yield root_abs
        if not osp.isdir(root_abs):
            continue
        for entry in os.scandir(root_abs):
            if entry.is_dir():
                candidate = osp.abspath(entry.path)
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate


def _resolve_network_pairs_dir(dataset_dir: str, processed_root: str, network_name: str) -> str:
    expected_name = _canonical_network_name(network_name)
    matches = []
    for candidate in _candidate_dirs(dataset_dir, processed_root):
        if not _is_network_pairs_processed_dir(candidate):
            continue
        metadata = _load_metadata(candidate)
        metadata_name = _canonical_network_name(metadata.get("network_name", ""))
        if expected_name and metadata_name != expected_name:
            continue
        matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            "Could not find a processed network-pairs dataset directory.\n"
            f"  network_name          = {network_name or '<any>'}\n"
            f"  dataset.dir           = {dataset_dir}\n"
            f"  dataset.processed_root= {processed_root}\n"
            "Expected a processed directory containing "
            "train_dataset.pt / val_dataset.pt / test_dataset.pt / dataset_meta.json."
        )

    if len(matches) == 1:
        return matches[0]

    if dataset_dir and _is_network_pairs_processed_dir(dataset_dir):
        return osp.abspath(dataset_dir)

    raise RuntimeError(
        f"Found multiple processed datasets for network_name='{network_name or '*'}': {matches}. "
        "Please point cfg.dataset.dir directly to the desired processed directory."
    )


def _log_loaded_dataset(dataset, dataset_dir: str) -> None:
    logging.info("[*] Loaded dataset '%s' from '%s'", cfg.dataset.name, dataset_dir)
    logging.info("  num graphs: %s", len(dataset))
    logging.info("  num node features: %s", dataset.num_node_features)
    logging.info("  num edge features: %s", dataset.num_edge_features)
    logging.info(
        "  metadata: num_nodes=%s num_edges_old=%s num_edges_new=%s od_dim=%s centroid_count=%s",
        cfg.dataset.num_nodes,
        cfg.dataset.num_edges_old,
        cfg.dataset.num_edges_new,
        cfg.dataset.od_dim,
        cfg.dataset.centroid_count,
    )


def _apply_feature_mask_if_requested(dataset):
    mask_capacity = bool(getattr(cfg.dataset, "mask_capacity", False))
    mask_fft = bool(getattr(cfg.dataset, "mask_fft", False))
    if not (mask_capacity or mask_fft):
        return
    transform = MaskEdgeFeatureTransform(mask_capacity=mask_capacity, mask_fft=mask_fft)
    logging.info("[Ablation] Applying %s", transform)
    pre_transform_in_memory(dataset, transform)


def preformat_network_pairs(dataset_dir: str):
    actual_dir = _resolve_network_pairs_dir(
        dataset_dir=dataset_dir,
        processed_root=getattr(cfg.dataset, "processed_root", dataset_dir),
        network_name=getattr(cfg.dataset, "network_name", ""),
    )
    metadata = _load_metadata(actual_dir)

    cfg.dataset.dir = actual_dir
    cfg.dataset.processed_root = actual_dir
    cfg.dataset.network_name = metadata["network_name"]
    cfg.dataset.num_nodes = int(metadata["num_nodes"])
    cfg.dataset.num_edges_old = int(metadata["num_edges_old"])
    cfg.dataset.num_edges_new = int(metadata["num_edges_new"])
    cfg.dataset.od_dim = int(metadata["od_dim"])
    cfg.dataset.centroid_count = int(metadata["centroid_count"])

    flow_scaler_path = osp.join(actual_dir, metadata["files"]["flow_scaler"])
    with open(flow_scaler_path, "rb") as handle:
        flow_scaler = pickle.load(handle)
    cfg.dataset.flow_mean = float(flow_scaler.mean_[0])
    cfg.dataset.flow_std = float(flow_scaler.scale_[0])

    split_datasets = [
        NetworkPairsTopologyDataset(root=actual_dir, split=split)
        for split in ["train", "val", "test"]
    ]
    dataset = join_dataset_splits(split_datasets)
    _apply_feature_mask_if_requested(dataset)

    if hasattr(dataset, "split_idxs"):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, "split_idxs")
    prepare_splits(dataset)
    _log_loaded_dataset(dataset, actual_dir)
    return dataset


@register_loader("custom_master_loader")
def load_dataset_master(format, name, dataset_dir):
    if format != "PyG-NetworkPairs":
        raise ValueError(
            f"Unsupported dataset format '{format}'. "
            "This codebase now only supports the ST-PINN NetworkPairs dataset."
        )
    del name
    return preformat_network_pairs(dataset_dir)


def join_dataset_splits(datasets):
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = (
        [datasets[0].get(i) for i in range(n1)]
        + [datasets[1].get(i) for i in range(n2)]
        + [datasets[2].get(i) for i in range(n3)]
    )

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    datasets[0].split_idxs = [
        list(range(n1)),
        list(range(n1, n1 + n2)),
        list(range(n1 + n2, n1 + n2 + n3)),
    ]
    return datasets[0]
