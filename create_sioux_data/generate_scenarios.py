"""
Generate network reconfiguration scenarios: (G, G') network pair data

This module accomplishes the following tasks:
1. Uses Latin Hypercube Sampling (LHS) to generate base scenario OD matrices and G edge attributes
2. Implements the targeted joint-mutation strategy used to build each G'
3. Constructs (G, G') network pairs where every sample undergoes topology change + attribute change

Strict Constraints:
- G and G' share exactly the same OD matrix (OD is only used for SUE computation, never as node features)
- G and G' have identical node sets; only edges and edge attributes may vary
- Every generated sample applies:
  * add edges from the highest-flow 30% nodes
  * delete one outgoing edge from the lowest-flow 20% nodes
  * mutate attributes on a random 20% subset of edges
"""

from copy import deepcopy
import math
from typing import Optional

import networkx as nx
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

try:
    from network_registry import MutationPolicy
except ModuleNotFoundError:
    from .network_registry import MutationPolicy


# ============================================================
# Part 1: Base Scenario Generation (LHS Sampling)
# ============================================================

def generate_lhs_base_scenarios(
    num_samples: int = 2000,
    num_centroids: int = 11,
    num_edges: int = 76,
    seed: int = 42
) -> tuple:
    """
    Use Latin Hypercube Sampling (LHS) to generate base scenario OD matrices and edge attributes for G.

    Parameter ranges (from paper specifications):
    - OD demand: 0 - 1500 vehicles/OD pair
    - Capacity: 4000 - 26000
    - Speed: 45 - 80 km/h

    Args:
        num_samples:   Number of scenarios to generate (default 2000)
        num_centroids: Number of centroid nodes (11 for Sioux Falls)
        num_edges:     Number of edges (76 for Sioux Falls)
        seed:          Random seed

    Returns:
        od_matrices: np.ndarray [num_samples, num_centroids, num_centroids]
        capacities:  np.ndarray [num_samples, num_edges]   -- G's capacities
        speeds:      np.ndarray [num_samples, num_edges]   -- G's speeds
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} base scenarios with LHS sampling")
    print(f"{'='*60}")

    num_od_pairs = num_centroids * num_centroids          # 11*11 = 121
    num_dims = num_od_pairs + num_edges * 2               # 121 + 76 + 76 = 273

    sampler = qmc.LatinHypercube(d=num_dims, seed=seed)
    samples = sampler.random(n=num_samples)               # [num_samples, 273]

    # Split by dimension
    od_raw  = samples[:, :num_od_pairs]                   # [N, 121]
    cap_raw = samples[:, num_od_pairs : num_od_pairs + num_edges]   # [N, 76]
    spd_raw = samples[:, num_od_pairs + num_edges:]       # [N, 76]

    # Map to actual range
    od_matrices = (od_raw * 1500.0).reshape(num_samples, num_centroids, num_centroids)
    capacities  = cap_raw * (26000 - 4000) + 4000
    speeds      = spd_raw * (80 - 45) + 45

    print(f"  OD matrix shape:  {od_matrices.shape}")
    print(f"  Capacity matrix shape: {capacities.shape}")
    print(f"  Speed matrix shape: {speeds.shape}")
    print(f"  OD range: [{od_matrices.min():.1f}, {od_matrices.max():.1f}]")
    print(f"  Capacity range: [{capacities.min():.1f}, {capacities.max():.1f}]")
    print(f"  Speed range: [{speeds.min():.1f}, {speeds.max():.1f}]")
    print(f"  LHS sampling complete!")

    return od_matrices, capacities, speeds


def _generate_lhs_edge_parameters(
    num_samples: int,
    num_edges: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate edge capacities and speeds without regenerating OD matrices."""
    sampler = qmc.LatinHypercube(d=num_edges * 2, seed=seed)
    samples = sampler.random(n=num_samples)

    cap_raw = samples[:, :num_edges]
    spd_raw = samples[:, num_edges:]

    capacities = cap_raw * (26000 - 4000) + 4000
    speeds = spd_raw * (80 - 45) + 45
    return capacities, speeds


def generate_base_scenarios_from_od_matrix(
    base_od_matrix: np.ndarray,
    num_samples: int = 2000,
    num_edges: int = 76,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeat a parsed OD matrix while still sampling edge attributes with LHS."""
    if base_od_matrix.ndim != 2 or base_od_matrix.shape[0] != base_od_matrix.shape[1]:
        raise ValueError("base_od_matrix must be a square [C, C] matrix.")

    od_matrices = np.repeat(base_od_matrix[None, :, :], num_samples, axis=0)
    capacities, speeds = _generate_lhs_edge_parameters(
        num_samples=num_samples,
        num_edges=num_edges,
        seed=seed,
    )
    return od_matrices, capacities, speeds


def _scaled_count(size: int, ratio: float, minimum: int = 1) -> int:
    """Convert a ratio to a valid integer count using ceil."""
    if size <= 0:
        return 0
    return min(size, max(minimum, int(math.ceil(size * ratio))))


def _compute_node_flow_sum(
    G: nx.DiGraph,
    flows_old: np.ndarray,
) -> dict[int, float]:
    """Aggregate incident old-graph flow onto each node."""
    edges = list(G.edges())
    if len(edges) != int(len(flows_old)):
        raise ValueError(
            f"flows_old length ({len(flows_old)}) does not match graph edge count ({len(edges)})."
        )

    node_flow_sum = {int(node_id): 0.0 for node_id in G.nodes()}
    for idx, (u, v) in enumerate(edges):
        flow_value = float(flows_old[idx])
        node_flow_sum[int(u)] += flow_value
        node_flow_sum[int(v)] += flow_value
    return node_flow_sum


def _select_ranked_nodes(
    node_flow_sum: dict[int, float],
    count: int,
    descending: bool,
) -> list[int]:
    """Select the highest- or lowest-flow nodes according to count."""
    if count <= 0:
        return []
    sorted_nodes = sorted(
        node_flow_sum,
        key=lambda node_id: node_flow_sum[node_id],
        reverse=descending,
    )
    return [int(node_id) for node_id in sorted_nodes[:count]]


def _resolve_mutation_ranges(
    G: nx.DiGraph,
    mutation_policy: Optional[MutationPolicy],
) -> dict:
    """Resolve absolute mutation counts from graph size and policy ratios."""
    policy = mutation_policy or MutationPolicy()
    num_nodes = max(G.number_of_nodes(), 1)
    num_edges = max(G.number_of_edges(), 1)

    return {
        'num_add_nodes': _scaled_count(num_nodes, policy.high_flow_add_ratio, minimum=1),
        'edges_per_node_range': policy.edges_per_node_range,
        'num_delete_nodes': _scaled_count(num_nodes, policy.low_flow_delete_ratio, minimum=1),
        'delete_edges_per_node': max(1, int(policy.delete_edges_per_node)),
        'num_attr_change_edges': _scaled_count(
            num_edges,
            policy.attribute_change_edge_ratio,
            minimum=1,
        ),
        'cap_scale_range': policy.cap_scale_range,
        'spd_scale_range': policy.spd_scale_range,
        'capacity_bounds': policy.capacity_bounds,
        'speed_bounds': policy.speed_bounds,
    }


# ============================================================
# Part 2: Three Types of Topology Mutation Algorithms
# ============================================================

def mutate_add_edges(
    G: nx.DiGraph,
    candidate_nodes: list[int],
    rng: np.random.Generator,
    edges_per_node_range: tuple = (1, 3),
) -> tuple:
    """
    Mutation Operation 1: Add Edges
    Add new out-edges to the highest-flow nodes of G.

    Algorithm logic:
    1. Candidate nodes are pre-selected outside this function from the top 30% highest-flow nodes.
    2. For each candidate node, randomly add 1-3 out-edges:
       - Target node: uniform random sample from all nodes in G, excluding self and already connected targets
       - New edge attributes: uniformly sample within [min, max] of each attribute from existing edges; ensures reasonable values
       - Automatically calculate free_flow_time = (length / speed) * 60

    Args:
        G:                  Current scenario NetworkX DiGraph (with full edge attributes)
        candidate_nodes:    Highest-flow source nodes selected from the old graph
        rng:                numpy random generator
        edges_per_node_range: Range for number of edges to add per node (inclusive)

    Returns:
        G_new:       DiGraph with new edges added
        added_edges: list of (u, v) newly added directed edges
    """
    edges    = list(G.edges())
    node_ids = list(G.nodes())   # Sioux Falls: nodes 1-24

    # --- Step 1: For each high-flow node, add 1-3 new out-edges ---
    G_new = deepcopy(G)
    added_edges = []

    # Compute attribute ranges from existing edges; new edges uniformly sample in-range
    all_caps = [G[u][v]['capacity']      for u, v in edges]
    all_spds = [G[u][v]['speed']         for u, v in edges]
    all_lens = [G[u][v]['length']        for u, v in edges]
    cap_min, cap_max = min(all_caps), max(all_caps)
    spd_min, spd_max = min(all_spds), max(all_spds)
    len_min, len_max = min(all_lens), max(all_lens)

    for node in candidate_nodes:
        # Exclude self and already existing out-edge targets (no repeats)
        existing_targets = set(G_new.successors(node))
        candidates = [n for n in node_ids if n != node and n not in existing_targets]

        if not candidates:
            # Node already connected to all other nodes; skip
            continue

        num_to_add = int(rng.integers(edges_per_node_range[0], edges_per_node_range[1] + 1))
        num_to_add = min(num_to_add, len(candidates))

        # Randomly pick target nodes from candidates
        targets = rng.choice(candidates, size=num_to_add, replace=False)

        for target in targets:
            # Edge attributes: sample uniformly within [min, max] from existing
            new_cap = float(rng.uniform(cap_min, cap_max))
            new_spd = float(rng.uniform(spd_min, spd_max))
            new_len = float(rng.uniform(len_min, len_max))
            # free_flow_time (minutes) = length (km) / speed (km/h) * 60
            new_fft = (new_len / new_spd) * 60.0

            G_new.add_edge(
                int(node), int(target),
                capacity=new_cap,
                speed=new_spd,
                length=new_len,
                free_flow_time=new_fft
            )
            added_edges.append((int(node), int(target)))

    return G_new, added_edges


def mutate_delete_edges(
    G: nx.DiGraph,
    candidate_nodes: list[int],
    rng: np.random.Generator,
    delete_edges_per_node: int = 1,
) -> tuple:
    """
    Mutation Operation 2: Delete Edges
    Delete outgoing edges from the lowest-flow nodes while maintaining strong connectivity.

    Algorithm logic:
    1. Randomly determine target number of deletions num_delete ∈ [5, 10]
    2. Randomly permute the edge order, and try deleting each edge in turn:
       - Temporarily remove the edge
       - Check if the graph is still strongly connected
         * Strong connectivity: for every ordered pair of nodes, there is a directed path from one to the other
         * This is necessary for SUE to be well-posed
       - If remains connected: confirm deletion and add to deleted list
       - If not connected: restore the edge (with all its original attributes)
    3. Stop as soon as num_delete deletions confirmed

    Note: If after going through all edges, the target is not reached, just keep what you deleted so far (no hard error)

    Args:
        G:                Current scenario NetworkX DiGraph
        rng:              numpy random generator
        num_delete_range: Range for number of edges to delete (inclusive)

    Returns:
        G_new:         DiGraph after deletions
        deleted_edges: list of (u, v) deleted directed edges
    """
    G_new = deepcopy(G)
    deleted_edges = []

    for node in candidate_nodes:
        deletions_for_node = 0
        outgoing_edges = list(G_new.out_edges(node))
        if not outgoing_edges:
            continue

        shuffled_indices = rng.permutation(len(outgoing_edges))
        for idx in shuffled_indices:
            if deletions_for_node >= delete_edges_per_node:
                break

            u, v = outgoing_edges[int(idx)]
            if not G_new.has_edge(u, v):
                continue

            edge_data = dict(G_new[u][v])
            G_new.remove_edge(u, v)
            if nx.is_strongly_connected(G_new):
                deleted_edges.append((u, v))
                deletions_for_node += 1
            else:
                G_new.add_edge(u, v, **edge_data)

    return G_new, deleted_edges


def mutate_attributes(
    G: nx.DiGraph,
    rng: np.random.Generator,
    num_attr_change_edges: int,
    cap_scale_range: tuple = (0.3, 2.0),
    spd_scale_range: tuple = (0.3, 2.0),
    capacity_bounds: tuple = (4000.0, 26000.0),
    speed_bounds: tuple = (45.0, 80.0),
) -> tuple:
    """
    Mutation Operation 3: Change Attributes
    Randomly mutate only a subset of edges, then clip capacity/speed back into feasible bounds.

    Algorithm logic:
    1. Iterate over every directed edge (u, v) in G
    2. Independently sample capacity scale λ_cap ~ Uniform(0.3, 2.0)
       Independently sample speed scale λ_spd ~ Uniform(0.3, 2.0)
    3. Update attributes:
       new_capacity      = old_capacity * λ_cap
       new_speed         = old_speed    * λ_spd
       new_free_flow_time = (length / new_speed) * 60   (recalculated with updated speed)
    4. Return the modified graph and per-edge scaling logs

    Note: length is unchanged (physical distance is not affected by reconfiguration);
    free_flow_time must be recalculated based on updated speed.

    Args:
        G:               Current scenario NetworkX DiGraph
        rng:             numpy random generator
        cap_scale_range: Range for capacity scale factor (inclusive)
        spd_scale_range: Range for speed scale factor (inclusive)

    Returns:
        G_new:       DiGraph with attributes mutated
        attr_changes: dict {(u, v): {'cap_scale': float, 'spd_scale': float}}
    """
    G_new = deepcopy(G)
    attr_changes = {}
    edges = list(G_new.edges())
    if not edges:
        return G_new, attr_changes

    num_attr_change_edges = min(max(1, int(num_attr_change_edges)), len(edges))
    selected_indices = rng.choice(len(edges), size=num_attr_change_edges, replace=False)

    for edge_idx in selected_indices:
        u, v = edges[int(edge_idx)]
        cap_scale = float(rng.uniform(cap_scale_range[0], cap_scale_range[1]))
        spd_scale = float(rng.uniform(spd_scale_range[0], spd_scale_range[1]))

        old_cap = G_new[u][v]['capacity']
        old_spd = G_new[u][v]['speed']
        old_len = G_new[u][v]['length']   # Physical distance remains unchanged

        new_cap = float(np.clip(old_cap * cap_scale, capacity_bounds[0], capacity_bounds[1]))
        new_spd = float(np.clip(old_spd * spd_scale, speed_bounds[0], speed_bounds[1]))
        # free_flow_time (minutes) = length (km) / speed (km/h) * 60
        new_fft = (old_len / new_spd) * 60.0

        G_new[u][v]['capacity']       = new_cap
        G_new[u][v]['speed']          = new_spd
        G_new[u][v]['free_flow_time'] = new_fft

        attr_changes[(u, v)] = {
            'cap_scale': cap_scale,
            'spd_scale': spd_scale,
            'capacity_after_clip': new_cap,
            'speed_after_clip': new_spd,
        }

    return G_new, attr_changes


# ============================================================
# Part 3: Build Concrete Scenario G and Generate G'
# ============================================================

def build_scenario_graph(G_topo: nx.DiGraph, capacities_i: np.ndarray, speeds_i: np.ndarray) -> nx.DiGraph:
    """
    Assign Sioux Falls base topology (G_topo) with the edge attributes for the i-th scenario,
    and return the specific G_i for that scenario (can be passed directly to SUE solver).

    Edge order matches exactly list(G_topo.edges()),
    thus capacities_i[j] corresponds to list(G_topo.edges())[j].

    Args:
        G_topo:       Sioux Falls base directed graph (with length attribute)
        capacities_i: np.ndarray [num_edges_G], edge capacities for scenario i
        speeds_i:     np.ndarray [num_edges_G], edge speeds for scenario i

    Returns:
        G_i: NetworkX DiGraph with assigned attributes
    """
    G_i = deepcopy(G_topo)
    edges = list(G_topo.edges())

    for j, (u, v) in enumerate(edges):
        length = G_topo[u][v]['length']         # Physical distance remains unchanged
        spd    = float(speeds_i[j])
        cap    = float(capacities_i[j])
        fft    = (length / spd) * 60.0          # Minutes

        G_i[u][v]['capacity']       = cap
        G_i[u][v]['speed']          = spd
        G_i[u][v]['free_flow_time'] = fft

    return G_i


def _apply_topology_mutation(
    G: nx.DiGraph,
    flows_old: np.ndarray,
    rng: np.random.Generator,
    mutation_info: dict,
    mutation_policy: Optional[MutationPolicy] = None,
) -> nx.DiGraph:
    """
    Internal helper: apply the targeted topology mutation to G.

    Args:
        G:           The current graph
        flows_old:   Historical flows on G (used for identifying top-flow nodes during edge addition)
        rng:         numpy random generator
        mutation_info: dict for recording mutation details (modified in-place)

    Returns:
        G_mutated: Graph after topology mutation
    """
    ranges = _resolve_mutation_ranges(G, mutation_policy)
    node_flow_sum = _compute_node_flow_sum(G, flows_old)
    high_flow_nodes = _select_ranked_nodes(
        node_flow_sum=node_flow_sum,
        count=ranges['num_add_nodes'],
        descending=True,
    )
    low_flow_nodes = _select_ranked_nodes(
        node_flow_sum=node_flow_sum,
        count=ranges['num_delete_nodes'],
        descending=False,
    )

    G_mut, added_edges = mutate_add_edges(
        G,
        candidate_nodes=high_flow_nodes,
        rng=rng,
        edges_per_node_range=ranges['edges_per_node_range'],
    )
    G_mut, deleted_edges = mutate_delete_edges(
        G_mut,
        candidate_nodes=low_flow_nodes,
        rng=rng,
        delete_edges_per_node=ranges['delete_edges_per_node'],
    )

    mutation_info['topo_op'] = 'add_then_delete'
    mutation_info['high_flow_nodes'] = high_flow_nodes
    mutation_info['low_flow_nodes'] = low_flow_nodes
    mutation_info['added_edges'] = added_edges
    mutation_info['deleted_edges'] = deleted_edges
    return G_mut


# ============================================================
# Part 4: Main Function for Network Pair Generation
# ============================================================

def generate_network_pairs(
    G_topo: nx.DiGraph,
    od_matrices: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    flows_old: np.ndarray,
    seed: int = 42,
    network_name: str = 'Unknown',
    node_ids: Optional[tuple[int, ...]] = None,
    centroid_nodes: Optional[tuple[int, ...]] = None,
    node_id_offset: int = 1,
    mutation_policy: Optional[MutationPolicy] = None,
) -> list:
    """
    Main function: for each base scenario, generate the corresponding mutated network G', yielding a full list of (G, G') pairs.

    Current strategy:
    - every sample applies topology change + attribute change
    - topology change = add edges on high-flow nodes, then delete edges on low-flow nodes
    - attribute change = randomly perturb 20% of edges and clip back to feasible bounds

    Returned structure for each pair (scenario_pair dict):
    ```
    {
        'od_matrix'     : np.ndarray [11, 11],  # shared OD matrix (for SUE only, never model input!)
        'G'             : nx.DiGraph,           # scenario G (fixed topology + LHS attributes)
        'G_prime'       : nx.DiGraph,           # mutated network G' (may have different edge set and attributes)
        'mutation_type' : str,                  # always 'both' under the targeted strategy
        'mutation_info' : dict,                 # details: added/deleted edges, attribute scaling info
    }
    ```

    Node set consistency guarantee:
    - G and G' always have identical node sets (24 nodes in Sioux Falls)
    - Nodes are never added or deleted; only edge set and edge attributes are changed

    Args:
        G_topo:      Sioux Falls base directed graph (read from .tntp file, contains length attribute)
        od_matrices: np.ndarray [N, 11, 11], OD matrices generated by LHS
        capacities:  np.ndarray [N, 76],     G's capacities generated by LHS
        speeds:      np.ndarray [N, 76],     G's speeds generated by LHS
        flows_old:   np.ndarray [N, 76],     historical flows solved on G (used to locate top-flow nodes for edge add mutations)
        seed:        Random seed

    Returns:
        scenario_pairs: list of dict, length N
    """
    num_samples = od_matrices.shape[0]
    rng = np.random.default_rng(seed)

    mutation_types = np.full(num_samples, 'both', dtype=object)

    print(f"\n{'='*60}")
    print(f"Generating {num_samples} (G, G') network pairs")
    print(f"{'='*60}")
    print("  Mutation type distribution:")
    print(f"    Topology+attribute (both): {num_samples} ({100.0:.1f}%)")

    scenario_pairs = []
    num_nodes = G_topo.number_of_nodes()
    node_ids = tuple(node_ids) if node_ids is not None else tuple(sorted(G_topo.nodes()))
    centroid_nodes = tuple(centroid_nodes) if centroid_nodes is not None else tuple()

    for i in tqdm(range(num_samples), desc="  Generating network pairs"):
        mutation_type = mutation_types[i]
        mutation_info = {'type': mutation_type}

        # --- Step 1: Build G_i for scenario i (fixed topology + LHS attributes) ---
        G_i = build_scenario_graph(G_topo, capacities[i], speeds[i])

        # --- Step 2: Starting from G_i, build G' ---
        G_prime = deepcopy(G_i)

        # Topology change first, then mutate attributes on the changed graph.
        G_prime = _apply_topology_mutation(
            G_prime,
            flows_old[i],
            rng,
            mutation_info,
            mutation_policy=mutation_policy,
        )
        ranges = _resolve_mutation_ranges(G_prime, mutation_policy)
        G_prime, attr_changes = mutate_attributes(
            G_prime,
            rng,
            num_attr_change_edges=ranges['num_attr_change_edges'],
            cap_scale_range=ranges['cap_scale_range'],
            spd_scale_range=ranges['spd_scale_range'],
            capacity_bounds=ranges['capacity_bounds'],
            speed_bounds=ranges['speed_bounds'],
        )
        mutation_info['attr_changes'] = attr_changes
        mutation_info['attr_changes_count'] = len(attr_changes)

        # --- Step 3: Node set consistency check ---
        # G and G' must have identical node sets; only edge set may change
        assert set(G_i.nodes()) == set(G_prime.nodes()), (
            f"Scenario {i}: Node set mismatch between G and G'! G: {set(G_i.nodes())}, G': {set(G_prime.nodes())}"
        )
        assert G_prime.number_of_nodes() == num_nodes, (
            f"Scenario {i}: G' node count {G_prime.number_of_nodes()} does not match expected {num_nodes}!"
        )

        scenario_pairs.append({
            'od_matrix'    : od_matrices[i].copy(),   # [11, 11], OD matrix (for SUE only)
            'G'            : G_i,                     # Scenario G (NetworkX DiGraph)
            'G_prime'      : G_prime,                 # Mutated network G' (NetworkX DiGraph)
            'mutation_type': mutation_type,
            'mutation_info': mutation_info,
            'network_name' : network_name,
            'node_ids'     : node_ids,
            'centroid_nodes': centroid_nodes,
            'node_id_offset': node_id_offset,
        })

    # --- Collate stats ---
    edge_counts_G      = [len(list(p['G'].edges()))       for p in scenario_pairs]
    edge_counts_Gprime = [len(list(p['G_prime'].edges())) for p in scenario_pairs]

    print(f"\n  Generation complete!")
    print(f"  G  edge count: fixed at {edge_counts_G[0]} (same in all scenarios)")
    print(f"  G' edge count: min={min(edge_counts_Gprime)}, "
          f"max={max(edge_counts_Gprime)}, "
          f"mean={np.mean(edge_counts_Gprime):.1f}")

    return scenario_pairs


# ============================================================
# Part 5: Utilities for Saving and Loading
# ============================================================

def save_scenarios(od_matrices, capacities, speeds, save_path='processed_data/raw/base_scenarios.npz'):
    """
    Save LHS base scenario data (numpy format, compact and compressed).

    Args:
        od_matrices: [N, 11, 11]
        capacities:  [N, 76]
        speeds:      [N, 76]
        save_path:   Target file path (.npz)
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, od_matrices=od_matrices, capacities=capacities, speeds=speeds)
    print(f"  Base scenarios saved: {save_path}")


def load_scenarios(load_path='processed_data/raw/base_scenarios.npz'):
    """
    Load LHS base scenario data.

    Returns:
        od_matrices, capacities, speeds
    """
    data = np.load(load_path)
    print(f"  Base scenarios loaded: {load_path}")
    return data['od_matrices'], data['capacities'], data['speeds']


def save_scenario_pairs(scenario_pairs: list, save_path='processed_data/raw/scenario_pairs.pkl'):
    """
    Save (G, G') pair list (includes NetworkX graph objects, serialized with pickle).

    Note: Each scenario_pair contains two NetworkX DiGraph objects, memory intensive.
    For large datasets (> 5000 samples), consider saving in batches.

    Args:
        scenario_pairs: list of dict, length = N
        save_path:      Target file path (.pkl)
    """
    import os
    import pickle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scenario_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Network pair data saved: {save_path} ({len(scenario_pairs)} pairs)")


def load_scenario_pairs(load_path='processed_data/raw/scenario_pairs.pkl') -> list:
    """
    Load (G, G') pair list.

    Returns:
        scenario_pairs: list of dict
    """
    import pickle
    with open(load_path, 'rb') as f:
        scenario_pairs = pickle.load(f)
    print(f"  Network pair data loaded: {load_path} ({len(scenario_pairs)} pairs)")
    return scenario_pairs
