from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class MutationPolicy:
    """Targeted joint-mutation policy used by the data generator."""

    high_flow_add_ratio: float = 0.30
    edges_per_node_range: Tuple[int, int] = (1, 3)
    low_flow_delete_ratio: float = 0.20
    delete_edges_per_node: int = 1
    attribute_change_edge_ratio: float = 0.20
    cap_scale_range: Tuple[float, float] = (0.3, 2.0)
    spd_scale_range: Tuple[float, float] = (0.3, 2.0)
    capacity_bounds: Tuple[float, float] = (4000.0, 26000.0)
    speed_bounds: Tuple[float, float] = (45.0, 80.0)


@dataclass(frozen=True)
class NetworkSpec:
    """Resolved network specification used by the data pipeline."""

    network_name: str
    dataset_root: str
    network_file: str
    od_file: str = ""
    parser: str = "tntp"
    node_id_offset: int = 1
    demand_source: str = "lhs"
    centroid_nodes: Optional[Tuple[int, ...]] = None
    mutation_policy: MutationPolicy = MutationPolicy()


_BUILTIN_SPECS = {
    "siouxfalls": {
        "network_name": "SiouxFalls",
        "dataset_root": "../sioux_data",
        "network_file": "SiouxFalls_net.tntp",
        "od_file": "",
        "centroid_nodes": tuple(range(1, 12)),
    },
    "ema": {
        "network_name": "EMA",
        "dataset_root": "../ema_data",
        "network_file": "EMA_net.tntp",
        "od_file": "EMA_trips.tntp",
        "centroid_nodes": None,
    },
    "anaheim": {
        "network_name": "Anaheim",
        "dataset_root": "../anaheim_data",
        "network_file": "Anaheim_net.tntp",
        "od_file": "Anaheim_trips.tntp",
        "centroid_nodes": None,
    },
}


def _normalize_name(network_name: str) -> str:
    return (network_name or "SiouxFalls").strip().lower()


def parse_centroid_nodes(raw_value: Optional[Iterable[int] | str]) -> Optional[Tuple[int, ...]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if not cleaned:
            return None
        return tuple(int(item.strip()) for item in cleaned.split(",") if item.strip())
    return tuple(int(item) for item in raw_value)


def _resolve_path(dataset_root: str, file_path: str) -> str:
    if not file_path:
        return ""
    if os.path.isabs(file_path):
        return file_path
    if dataset_root:
        return os.path.join(dataset_root, file_path)
    return file_path


def _resolve_dataset_root(dataset_root: str, relative_to_module: bool = False) -> str:
    if not dataset_root:
        return ""
    if os.path.isabs(dataset_root):
        return dataset_root
    base_dir = _MODULE_DIR if relative_to_module else os.getcwd()
    return os.path.abspath(os.path.join(base_dir, dataset_root))


def _build_custom_spec(
    network_name: str,
    dataset_root: str,
    network_file: str,
    od_file: str,
    parser: str,
    node_id_offset: int,
    demand_source: str,
    centroid_nodes: Optional[Tuple[int, ...]],
) -> NetworkSpec:
    if not network_file:
        raise ValueError(
            f"Unknown network_name='{network_name}'. "
            "Provide --network_file (and optionally --dataset_root / --od_file) "
            "to use a custom network preset."
        )

    resolved_root = _resolve_dataset_root(dataset_root, relative_to_module=False)
    resolved_network_file = _resolve_path(resolved_root, network_file)
    resolved_od_file = _resolve_path(resolved_root, od_file)
    display_name = (network_name or os.path.splitext(os.path.basename(network_file))[0]).strip()
    if not display_name:
        display_name = "CustomNetwork"

    return NetworkSpec(
        network_name=display_name,
        dataset_root=resolved_root,
        network_file=resolved_network_file,
        od_file=resolved_od_file,
        parser=parser,
        node_id_offset=node_id_offset,
        demand_source=demand_source,
        centroid_nodes=centroid_nodes,
        mutation_policy=MutationPolicy(),
    )


def resolve_network_spec(
    network_name: str = "SiouxFalls",
    dataset_root: str = "",
    network_file: str = "",
    od_file: str = "",
    parser: str = "tntp",
    node_id_offset: int = 1,
    demand_source: str = "lhs",
    centroid_nodes: Optional[Iterable[int] | str] = None,
) -> NetworkSpec:
    key = _normalize_name(network_name)
    explicit_centroids = parse_centroid_nodes(centroid_nodes)

    if key not in _BUILTIN_SPECS:
        return _build_custom_spec(
            network_name=network_name,
            dataset_root=dataset_root,
            network_file=network_file,
            od_file=od_file,
            parser=parser,
            node_id_offset=node_id_offset,
            demand_source=demand_source,
            centroid_nodes=explicit_centroids,
        )

    base = _BUILTIN_SPECS[key]
    resolved_root = _resolve_dataset_root(
        dataset_root or base["dataset_root"],
        relative_to_module=not bool(dataset_root),
    )
    resolved_network_file = _resolve_path(resolved_root, network_file or base["network_file"])
    resolved_od_file = _resolve_path(resolved_root, od_file or base["od_file"])

    if explicit_centroids is None:
        explicit_centroids = base["centroid_nodes"]

    return NetworkSpec(
        network_name=base["network_name"],
        dataset_root=resolved_root,
        network_file=resolved_network_file,
        od_file=resolved_od_file,
        parser=parser,
        node_id_offset=node_id_offset,
        demand_source=demand_source,
        centroid_nodes=explicit_centroids,
        mutation_policy=MutationPolicy(),
    )
