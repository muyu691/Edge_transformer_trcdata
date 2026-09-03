"""Generate realism-preserving traffic scenarios and network pairs.

Each sample starts from the attributes and OD matrix of a parsed TNTP
baseline.  Demand, capacity, and speed receive small multiplicative operating
condition perturbations.  The resulting graph ``G`` is then changed by exactly
one flow-independent reconfiguration to produce ``G_prime``.

The node set and OD matrix are identical in ``G`` and ``G_prime``.  OD is
stored for SUE solving only and is not converted into a model feature here.
"""

from copy import deepcopy
import os
import pickle
from typing import Optional, Sequence

import networkx as nx
import numpy as np
from tqdm import tqdm

try:
    from network_registry import MutationPolicy
except ModuleNotFoundError:
    from .network_registry import MutationPolicy


class ReconfigurationError(RuntimeError):
    """Raised when a requested reconfiguration cannot be applied safely."""


def _validate_positive_edge_attributes(graph: nx.DiGraph) -> None:
    """Reject baselines that cannot support physical perturbations."""
    if graph.number_of_nodes() == 0:
        raise ValueError("baseline_graph must contain at least one node.")
    if graph.number_of_edges() == 0:
        raise ValueError("baseline_graph must contain at least one edge.")

    for u, v, data in graph.edges(data=True):
        for name in ("capacity", "speed", "length"):
            if name not in data:
                raise ValueError(f"Edge {(u, v)} is missing required attribute '{name}'.")
            value = float(data[name])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Edge {(u, v)} has invalid {name}={data[name]!r}; "
                    f"all physical edge attributes must be finite and positive."
                )


def generate_baseline_perturbed_scenarios(
    base_od_matrix: np.ndarray,
    baseline_graph: nx.DiGraph,
    num_samples: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate scenarios by perturbing a TNTP OD/network baseline.

    For scenario ``s`` the OD multiplier is
    ``clip(g_s * origin_factor_i * destination_factor_j, 0.5, 1.5)``,
    with ``g_s`` in ``[0.90, 1.10]`` and the origin/destination factors in
    ``[0.85, 1.15]``.  Baseline-zero OD entries remain zero and the diagonal is
    forced to zero.  Edge capacities and speeds are independently scaled by
    ``[0.90, 1.10]`` and ``[0.95, 1.05]`` respectively.  Length is never
    sampled; it remains stored on ``baseline_graph`` and is reused later by
    :func:`build_scenario_graph`.

    The three returned arrays intentionally retain the existing downstream
    shapes: ``[N, C, C]``, ``[N, E]``, and ``[N, E]``.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if not isinstance(baseline_graph, nx.DiGraph):
        raise TypeError("baseline_graph must be a networkx.DiGraph.")

    base_od = np.asarray(base_od_matrix, dtype=np.float64)
    if base_od.ndim != 2 or base_od.shape[0] != base_od.shape[1]:
        raise ValueError("base_od_matrix must be a square [C, C] matrix.")
    if base_od.shape[0] == 0:
        raise ValueError("base_od_matrix must contain at least one centroid.")
    if not np.all(np.isfinite(base_od)) or np.any(base_od < 0.0):
        raise ValueError("base_od_matrix must contain finite, non-negative demand.")

    _validate_positive_edge_attributes(baseline_graph)
    edges = list(baseline_graph.edges())
    baseline_capacities = np.asarray(
        [baseline_graph[u][v]["capacity"] for u, v in edges],
        dtype=np.float64,
    )
    baseline_speeds = np.asarray(
        [baseline_graph[u][v]["speed"] for u, v in edges],
        dtype=np.float64,
    )

    num_centroids = int(base_od.shape[0])
    num_edges = len(edges)
    od_matrices = np.empty(
        (num_samples, num_centroids, num_centroids),
        dtype=np.float64,
    )
    capacities = np.empty((num_samples, num_edges), dtype=np.float64)
    speeds = np.empty((num_samples, num_edges), dtype=np.float64)

    rng = np.random.default_rng(seed)
    zero_mask = base_od == 0.0
    for sample_idx in range(num_samples):
        global_factor = float(rng.uniform(0.90, 1.10))
        origin_factors = rng.uniform(0.85, 1.15, size=num_centroids)
        destination_factors = rng.uniform(0.85, 1.15, size=num_centroids)
        od_multiplier = np.clip(
            global_factor * origin_factors[:, None] * destination_factors[None, :],
            0.50,
            1.50,
        )

        scenario_od = base_od * od_multiplier
        scenario_od[zero_mask] = 0.0
        np.fill_diagonal(scenario_od, 0.0)
        od_matrices[sample_idx] = scenario_od

        capacities[sample_idx] = baseline_capacities * rng.uniform(
            0.90,
            1.10,
            size=num_edges,
        )
        speeds[sample_idx] = baseline_speeds * rng.uniform(
            0.95,
            1.05,
            size=num_edges,
        )

    return od_matrices, capacities, speeds


def _ratio_count(size: int, ratio: float) -> int:
    """Translate a requested ratio into a non-zero, bounded edge count."""
    if size <= 0:
        return 0
    return min(size, max(1, int(round(float(ratio) * size))))


def _mutation_choices(policy: MutationPolicy) -> tuple[tuple[str, ...], np.ndarray]:
    mutation_types = ("closure", "capacity_change", "new_link")
    probabilities = np.asarray(
        [
            policy.closure_probability,
            policy.capacity_change_probability,
            policy.new_link_probability,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("Mutation probabilities must be finite and non-negative.")
    if not np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Mutation probabilities must sum to 1.0.")
    return mutation_types, probabilities


def _reachable_positive_od_pairs(
    graph: nx.DiGraph,
    od_matrix: np.ndarray,
    centroid_nodes: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Return positive-demand centroid pairs that are reachable in ``graph``."""
    od = np.asarray(od_matrix, dtype=np.float64)
    centroids = tuple(int(node_id) for node_id in centroid_nodes)
    if od.shape != (len(centroids), len(centroids)):
        raise ReconfigurationError(
            "OD shape and centroid_nodes disagree while checking closure connectivity."
        )

    reachable_pairs = []
    for origin_idx, destination_idx in np.argwhere(od > 0.0):
        if origin_idx == destination_idx:
            continue
        origin = centroids[int(origin_idx)]
        destination = centroids[int(destination_idx)]
        if nx.has_path(graph, origin, destination):
            reachable_pairs.append((origin, destination))
    return tuple(reachable_pairs)


def mutate_closure(
    graph: nx.DiGraph,
    rng: np.random.Generator,
    closure_ratios: Sequence[float],
    od_matrix: np.ndarray,
    centroid_nodes: Sequence[int],
) -> tuple[nx.DiGraph, dict]:
    """Randomly close links while preserving the baseline reachability contract."""
    ratios = tuple(float(value) for value in closure_ratios)
    if not ratios or any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in ratios):
        raise ValueError("closure_ratios must contain values in (0, 1].")

    candidates = [(u, v) for u, v in graph.edges() if u != v]
    if not candidates:
        raise ReconfigurationError("Closure failed: the graph has no non-self-loop edge.")

    requested_ratio = float(rng.choice(np.asarray(ratios, dtype=np.float64)))
    requested_count = _ratio_count(graph.number_of_edges(), requested_ratio)
    preserve_strong_connectivity = nx.is_strongly_connected(graph)
    reachable_od_pairs = (
        tuple()
        if preserve_strong_connectivity
        else _reachable_positive_od_pairs(graph, od_matrix, centroid_nodes)
    )

    mutated = deepcopy(graph)
    deleted_edges = []
    for candidate_idx in rng.permutation(len(candidates)):
        if len(deleted_edges) >= requested_count:
            break
        u, v = candidates[int(candidate_idx)]
        if not mutated.has_edge(u, v):
            continue

        trial = mutated.copy()
        trial.remove_edge(u, v)
        if preserve_strong_connectivity:
            remains_valid = nx.is_strongly_connected(trial)
        else:
            remains_valid = all(
                nx.has_path(trial, origin, destination)
                for origin, destination in reachable_od_pairs
            )

        if remains_valid:
            mutated.remove_edge(u, v)
            deleted_edges.append((int(u), int(v)))

    if not deleted_edges:
        raise ReconfigurationError(
            "Closure failed: no edge could be removed without breaking required connectivity."
        )

    return mutated, {
        "deleted_edges": deleted_edges,
        "requested_ratio": requested_ratio,
        "requested_count": requested_count,
        "actual_deleted_count": len(deleted_edges),
    }


def mutate_capacity_change(
    graph: nx.DiGraph,
    rng: np.random.Generator,
    edge_ratio: float,
    reduction_probability: float,
    reduction_scale_range: Sequence[float],
    expansion_scale_range: Sequence[float],
) -> tuple[nx.DiGraph, dict]:
    """Change capacity on a random edge subset without changing other attributes."""
    edge_ratio = float(edge_ratio)
    reduction_probability = float(reduction_probability)
    reduction_range = tuple(float(value) for value in reduction_scale_range)
    expansion_range = tuple(float(value) for value in expansion_scale_range)
    if not np.isfinite(edge_ratio) or edge_ratio <= 0.0 or edge_ratio > 1.0:
        raise ValueError("capacity_change_edge_ratio must be in (0, 1].")
    if not np.isfinite(reduction_probability) or not 0.0 <= reduction_probability <= 1.0:
        raise ValueError("capacity_reduction_probability must be in [0, 1].")
    if (
        len(reduction_range) != 2
        or len(expansion_range) != 2
        or not 0.0 < reduction_range[0] <= reduction_range[1]
        or not 0.0 < expansion_range[0] <= expansion_range[1]
    ):
        raise ValueError("Capacity scale ranges must be ordered and strictly positive.")

    edges = list(graph.edges())
    requested_count = _ratio_count(len(edges), edge_ratio)
    if requested_count == 0:
        raise ReconfigurationError("Capacity change failed: the graph has no edge.")

    selected_indices = rng.choice(len(edges), size=requested_count, replace=False)
    mutated = deepcopy(graph)
    changed_edges = []
    capacity_scale = {}
    capacity_before = {}
    capacity_after = {}

    for edge_idx in selected_indices:
        u, v = edges[int(edge_idx)]
        old_capacity = float(mutated[u][v]["capacity"])
        if rng.random() < reduction_probability:
            scale = float(rng.uniform(reduction_range[0], reduction_range[1]))
        else:
            scale = float(rng.uniform(expansion_range[0], expansion_range[1]))
        new_capacity = old_capacity * scale
        if not np.isfinite(new_capacity) or new_capacity <= 0.0:
            raise ReconfigurationError(
                f"Capacity change produced invalid capacity on edge {(u, v)}."
            )

        mutated[u][v]["capacity"] = new_capacity
        edge = (int(u), int(v))
        changed_edges.append(edge)
        capacity_scale[edge] = scale
        capacity_before[edge] = old_capacity
        capacity_after[edge] = new_capacity

    return mutated, {
        "changed_edges": changed_edges,
        "requested_ratio": edge_ratio,
        "actual_changed_count": len(changed_edges),
        "capacity_scale": capacity_scale,
        "capacity_before": capacity_before,
        "capacity_after": capacity_after,
    }


def _topology_local_candidates(
    graph: nx.DiGraph,
    hop_range: Sequence[int],
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Enumerate missing directed links whose current hop distance is local."""
    hops = tuple(int(value) for value in hop_range)
    if len(hops) != 2 or hops[0] < 1 or hops[0] > hops[1]:
        raise ValueError("new_link_hop_range must be an ordered pair of positive integers.")
    min_hops, max_hops = hops

    candidates = []
    for source in graph.nodes():
        paths = nx.single_source_shortest_path(graph, source, cutoff=max_hops)
        for target, path in paths.items():
            hop_distance = len(path) - 1
            if (
                source != target
                and min_hops <= hop_distance <= max_hops
                and not graph.has_edge(source, target)
            ):
                candidates.append(
                    (int(source), int(target), tuple(int(node_id) for node_id in path))
                )
    return candidates


def mutate_new_link(
    graph: nx.DiGraph,
    rng: np.random.Generator,
    edge_ratio: float,
    max_new_links: int,
    hop_range: Sequence[int],
    length_scale_range: Sequence[float],
) -> tuple[nx.DiGraph, dict]:
    """Add topology-local shortcut links with corridor-derived attributes."""
    edge_ratio = float(edge_ratio)
    max_new_links = int(max_new_links)
    length_range = tuple(float(value) for value in length_scale_range)
    if not np.isfinite(edge_ratio) or edge_ratio <= 0.0:
        raise ValueError("new_link_edge_ratio must be positive.")
    if max_new_links <= 0:
        raise ValueError("max_new_links must be positive.")
    if (
        len(length_range) != 2
        or not 0.0 < length_range[0] <= length_range[1]
    ):
        raise ValueError("new_link_length_scale_range must be ordered and positive.")

    requested_count = min(
        max_new_links,
        max(1, int(round(edge_ratio * graph.number_of_edges()))),
    )
    candidates = _topology_local_candidates(graph, hop_range)
    if not candidates:
        raise ReconfigurationError(
            "New-link mutation failed: no missing link has an admissible hop distance."
        )

    actual_target = min(requested_count, len(candidates))
    selected_indices = rng.choice(len(candidates), size=actual_target, replace=False)
    mutated = deepcopy(graph)
    added_edges = []
    added_edge_attributes = {}

    for candidate_idx in selected_indices:
        u, v, path = candidates[int(candidate_idx)]
        corridor_edges = list(zip(path[:-1], path[1:]))
        corridor_length = float(
            sum(float(graph[a][b]["length"]) for a, b in corridor_edges)
        )
        corridor_capacities = np.asarray(
            [graph[a][b]["capacity"] for a, b in corridor_edges],
            dtype=np.float64,
        )
        corridor_speeds = np.asarray(
            [graph[a][b]["speed"] for a, b in corridor_edges],
            dtype=np.float64,
        )

        shortcut_scale = float(rng.uniform(length_range[0], length_range[1]))
        new_length = corridor_length * shortcut_scale
        new_capacity = float(np.median(corridor_capacities))
        new_speed = float(np.median(corridor_speeds))
        new_free_flow_time = new_length / new_speed * 60.0
        generated = np.asarray(
            [new_capacity, new_speed, new_length, new_free_flow_time],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(generated)) or np.any(generated <= 0.0):
            raise ReconfigurationError(
                f"New-link mutation produced invalid attributes for edge {(u, v)}."
            )

        mutated.add_edge(
            u,
            v,
            capacity=new_capacity,
            speed=new_speed,
            length=new_length,
            free_flow_time=new_free_flow_time,
        )
        edge = (int(u), int(v))
        added_edges.append(edge)
        added_edge_attributes[edge] = {
            "capacity": new_capacity,
            "speed": new_speed,
            "length": new_length,
            "free_flow_time": new_free_flow_time,
            "corridor_path": path,
            "shortcut_scale": shortcut_scale,
        }

    if not added_edges:
        raise ReconfigurationError("New-link mutation failed to add any edge.")

    return mutated, {
        "added_edges": added_edges,
        "requested_count": requested_count,
        "actual_added_count": len(added_edges),
        "added_edge_attributes": added_edge_attributes,
    }


def build_scenario_graph(
    baseline_graph: nx.DiGraph,
    capacities_i: np.ndarray,
    speeds_i: np.ndarray,
) -> nx.DiGraph:
    """Build one operating-condition graph in canonical baseline edge order."""
    edges = list(baseline_graph.edges())
    capacities_i = np.asarray(capacities_i, dtype=np.float64).reshape(-1)
    speeds_i = np.asarray(speeds_i, dtype=np.float64).reshape(-1)
    if capacities_i.shape != (len(edges),) or speeds_i.shape != (len(edges),):
        raise ValueError(
            "Scenario capacity/speed arrays must match list(baseline_graph.edges())."
        )
    if (
        not np.all(np.isfinite(capacities_i))
        or not np.all(np.isfinite(speeds_i))
        or np.any(capacities_i <= 0.0)
        or np.any(speeds_i <= 0.0)
    ):
        raise ValueError("Scenario capacities and speeds must be finite and positive.")

    scenario_graph = deepcopy(baseline_graph)
    for edge_idx, (u, v) in enumerate(edges):
        length = float(baseline_graph[u][v]["length"])
        speed = float(speeds_i[edge_idx])
        capacity = float(capacities_i[edge_idx])
        free_flow_time = length / speed * 60.0
        if not np.isfinite(free_flow_time) or free_flow_time <= 0.0:
            raise ValueError(f"Edge {(u, v)} has invalid scenario free-flow time.")

        scenario_graph[u][v]["capacity"] = capacity
        scenario_graph[u][v]["speed"] = speed
        scenario_graph[u][v]["free_flow_time"] = free_flow_time

    return scenario_graph


def _apply_reconfiguration(
    graph: nx.DiGraph,
    mutation_type: str,
    rng: np.random.Generator,
    policy: MutationPolicy,
    od_matrix: np.ndarray,
    centroid_nodes: Sequence[int],
) -> tuple[nx.DiGraph, dict]:
    """Dispatch exactly one primary reconfiguration type."""
    if mutation_type == "closure":
        mutated, mutation_info = mutate_closure(
            graph,
            rng,
            closure_ratios=policy.closure_ratios,
            od_matrix=od_matrix,
            centroid_nodes=centroid_nodes,
        )
    elif mutation_type == "capacity_change":
        mutated, mutation_info = mutate_capacity_change(
            graph,
            rng,
            edge_ratio=policy.capacity_change_edge_ratio,
            reduction_probability=policy.capacity_reduction_probability,
            reduction_scale_range=policy.capacity_reduction_scale_range,
            expansion_scale_range=policy.capacity_expansion_scale_range,
        )
    elif mutation_type == "new_link":
        mutated, mutation_info = mutate_new_link(
            graph,
            rng,
            edge_ratio=policy.new_link_edge_ratio,
            max_new_links=policy.max_new_links,
            hop_range=policy.new_link_hop_range,
            length_scale_range=policy.new_link_length_scale_range,
        )
    else:
        raise ValueError(f"Unsupported mutation_type={mutation_type!r}.")

    mutation_info = {"type": mutation_type, **mutation_info}
    return mutated, mutation_info


def generate_network_pairs(
    G_topo: nx.DiGraph,
    od_matrices: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    seed: int = 42,
    network_name: str = "Unknown",
    node_ids: Optional[tuple[int, ...]] = None,
    centroid_nodes: Optional[tuple[int, ...]] = None,
    node_id_offset: int = 1,
    mutation_policy: Optional[MutationPolicy] = None,
) -> tuple[list, list[dict]]:
    """Generate ``(G, G_prime)`` pairs and explicit mutation-failure records.

    Reconfiguration selection is independent of old flows.  Each successful
    pair stores its original ``sample_idx`` so later solver filtering cannot
    misalign it with the corresponding base scenario or old SUE solution.
    """
    if not isinstance(G_topo, nx.DiGraph):
        raise TypeError("G_topo must be a networkx.DiGraph.")
    _validate_positive_edge_attributes(G_topo)

    od_matrices = np.asarray(od_matrices, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    if od_matrices.ndim != 3 or od_matrices.shape[1] != od_matrices.shape[2]:
        raise ValueError("od_matrices must have shape [N, C, C].")

    num_samples = int(od_matrices.shape[0])
    num_edges = G_topo.number_of_edges()
    if capacities.shape != (num_samples, num_edges):
        raise ValueError(
            f"capacities must have shape {(num_samples, num_edges)}, got {capacities.shape}."
        )
    if speeds.shape != (num_samples, num_edges):
        raise ValueError(
            f"speeds must have shape {(num_samples, num_edges)}, got {speeds.shape}."
        )
    if num_samples == 0:
        return [], []

    resolved_node_ids = (
        tuple(int(node_id) for node_id in node_ids)
        if node_ids is not None
        else tuple(int(node_id) for node_id in sorted(G_topo.nodes()))
    )
    if len(set(resolved_node_ids)) != len(resolved_node_ids) or set(resolved_node_ids) != set(G_topo.nodes()):
        raise ValueError("node_ids must list every graph node exactly once.")
    if centroid_nodes is None:
        raise ValueError("centroid_nodes is required for OD-to-node alignment.")
    resolved_centroids = tuple(int(node_id) for node_id in centroid_nodes)
    od_dim = int(od_matrices.shape[1])
    if len(resolved_centroids) != od_dim:
        raise ValueError(
            f"centroid_nodes has length {len(resolved_centroids)}, but OD dimension is {od_dim}."
        )
    if len(set(resolved_centroids)) != len(resolved_centroids):
        raise ValueError("centroid_nodes must not contain duplicates.")
    if not set(resolved_centroids).issubset(G_topo.nodes()):
        raise ValueError("Every centroid node must exist in G_topo.")

    policy = mutation_policy or MutationPolicy()
    mutation_names, mutation_probabilities = _mutation_choices(policy)
    rng = np.random.default_rng(seed)
    mutation_types = rng.choice(
        np.asarray(mutation_names, dtype=object),
        size=num_samples,
        p=mutation_probabilities,
    )

    print(f"\n{'=' * 60}")
    print(f"Generating {num_samples} (G, G') network pairs")
    print(f"{'=' * 60}")
    print("  Primary mutation probabilities:")
    for mutation_name, probability in zip(mutation_names, mutation_probabilities):
        print(f"    {mutation_name:16s}: {probability:.0%}")

    scenario_pairs = []
    failure_records = []
    expected_nodes = set(G_topo.nodes())
    for sample_idx in tqdm(range(num_samples), desc="  Generating network pairs"):
        mutation_type = str(mutation_types[sample_idx])
        scenario_graph = build_scenario_graph(
            G_topo,
            capacities[sample_idx],
            speeds[sample_idx],
        )

        try:
            mutated_graph, mutation_info = _apply_reconfiguration(
                graph=scenario_graph,
                mutation_type=mutation_type,
                rng=rng,
                policy=policy,
                od_matrix=od_matrices[sample_idx],
                centroid_nodes=resolved_centroids,
            )
        except ReconfigurationError as exc:
            failure_records.append(
                {
                    "index": sample_idx,
                    "reason": str(exc),
                    "mutation_type": mutation_type,
                }
            )
            continue

        if set(mutated_graph.nodes()) != expected_nodes:
            raise AssertionError(
                f"Scenario {sample_idx}: G and G_prime must have identical node sets."
            )

        scenario_pairs.append(
            {
                "sample_idx": sample_idx,
                "od_matrix": od_matrices[sample_idx].copy(),
                "G": scenario_graph,
                "G_prime": mutated_graph,
                "mutation_type": mutation_type,
                "mutation_info": mutation_info,
                "network_name": network_name,
                "node_ids": resolved_node_ids,
                "centroid_nodes": resolved_centroids,
                "node_id_offset": node_id_offset,
            }
        )

    if scenario_pairs:
        edge_counts = [pair["G_prime"].number_of_edges() for pair in scenario_pairs]
        print(
            "  Generation complete: "
            f"{len(scenario_pairs)} valid, {len(failure_records)} failed; "
            f"G' edges min={min(edge_counts)}, max={max(edge_counts)}, "
            f"mean={np.mean(edge_counts):.1f}."
        )
    else:
        print(
            "  Generation complete: no valid network pairs; "
            f"{len(failure_records)} mutations failed."
        )

    return scenario_pairs, failure_records


def save_scenarios(
    od_matrices: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    save_path: str = "processed_data/raw/base_scenarios.npz",
) -> None:
    """Save compact baseline-perturbed scenario arrays."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    np.savez_compressed(
        save_path,
        od_matrices=od_matrices,
        capacities=capacities,
        speeds=speeds,
    )
    print(f"  Base scenarios saved: {save_path}")


def load_scenarios(
    load_path: str = "processed_data/raw/base_scenarios.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load baseline-perturbed scenario arrays."""
    data = np.load(load_path)
    print(f"  Base scenarios loaded: {load_path}")
    return data["od_matrices"], data["capacities"], data["speeds"]


def save_scenario_pairs(
    scenario_pairs: list,
    save_path: str = "processed_data/raw/scenario_pairs.pkl",
) -> None:
    """Serialize NetworkX network-pair dictionaries with pickle."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "wb") as handle:
        pickle.dump(scenario_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Network pair data saved: {save_path} ({len(scenario_pairs)} pairs)")


def load_scenario_pairs(
    load_path: str = "processed_data/raw/scenario_pairs.pkl",
) -> list:
    """Load serialized NetworkX network-pair dictionaries."""
    with open(load_path, "rb") as handle:
        scenario_pairs = pickle.load(handle)
    print(f"  Network pair data loaded: {load_path} ({len(scenario_pairs)} pairs)")
    return scenario_pairs
