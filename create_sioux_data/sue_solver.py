import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm


EPS = 1e-12
V_MAX = 1e6
EXP_CLIP_MIN = -700.0
EXP_CLIP_MAX = 60.0


class MarkovLoadingConvergenceWarning(UserWarning):
    """Raised when the inner Markov loading step misses the configured tolerance."""


def bpr_travel_time(flow, capacity, free_flow_time, alpha=0.15, beta=4.0):
    """
    BPR 
    t = t0 * (1 + alpha * (flow / capacity)^beta)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap_safe = np.maximum(np.asarray(capacity, dtype=np.float64), EPS)
        ratio = np.clip(np.asarray(flow, dtype=np.float64) / cap_safe, 0.0, 1e6)
        return np.asarray(free_flow_time, dtype=np.float64) * (1.0 + alpha * (ratio ** beta))


def _build_sparse_edge_incidence(num_nodes, tails, heads):
    num_edges = tails.shape[0]
    edge_ids = np.arange(num_edges, dtype=np.int64)
    ones = np.ones(num_edges, dtype=np.float64)
    out_mat = sp.csr_matrix((ones, (tails, edge_ids)), shape=(num_nodes, num_edges))
    in_mat = sp.csr_matrix((ones, (heads, edge_ids)), shape=(num_nodes, num_edges))
    return out_mat, in_mat


def _centroid_destination_nodes(od_matrix, num_nodes):
    num_centroids = int(od_matrix.shape[0])
    if num_centroids > num_nodes:
        raise ValueError(
            f"OD centroid count ({num_centroids}) exceeds graph nodes ({num_nodes})."
        )
    return np.arange(num_centroids, dtype=np.int64)


def _compute_reasonable_link_mask(travel_times, tails, heads, num_nodes, dest_nodes):
    """
    Keep only links that move closer to each destination.

    Dense cyclic networks with small link costs can make unrestricted
    recursive-logit loading circulate indefinitely instead of being absorbed at
    the destination.  A destination-specific downhill mask preserves stochastic
    route choice over reasonable links while making the Markov chain proper.
    """
    num_edges = tails.shape[0]
    tt = np.asarray(travel_times, dtype=np.float64).reshape(num_edges)

    reverse_graph = sp.csr_matrix((tt, (heads, tails)), shape=(num_nodes, num_nodes))
    dist = dijkstra(
        csgraph=reverse_graph,
        directed=True,
        indices=dest_nodes,
        return_predecessors=False,
    )
    dist = np.atleast_2d(dist).T

    tail_dist = dist[tails, :]
    head_dist = dist[heads, :]
    reasonable = np.isfinite(tail_dist) & np.isfinite(head_dist) & (
        head_dist < tail_dist - 1e-12
    )
    reasonable[tails[:, None] == dest_nodes[None, :]] = False

    outgoing = [np.flatnonzero(tails == node) for node in range(num_nodes)]
    for dest_col, dest_node in enumerate(dest_nodes):
        for node in range(num_nodes):
            if node == int(dest_node):
                continue
            edge_ids = outgoing[node]
            if edge_ids.size == 0 or np.any(reasonable[edge_ids, dest_col]):
                continue
            finite_edge_ids = edge_ids[np.isfinite(dist[heads[edge_ids], dest_col])]
            if finite_edge_ids.size == 0:
                continue
            scores = tt[finite_edge_ids] + dist[heads[finite_edge_ids], dest_col]
            best_edge = finite_edge_ids[int(np.argmin(scores))]
            reasonable[best_edge, dest_col] = True

    return reasonable


def _solve_recursive_logit_values(
    travel_times,
    tails,
    heads,
    out_mat,
    dest_nodes,
    theta,
    reasonable_mask=None,
    max_iter=200,
    tol=1e-8,
):
    num_nodes = int(out_mat.shape[0])
    num_edges = tails.shape[0]
    num_dests = dest_nodes.shape[0]
    tt = np.asarray(travel_times, dtype=np.float64).reshape(num_edges)

    V = np.zeros((num_nodes, num_dests), dtype=np.float64)
    V[dest_nodes, np.arange(num_dests)] = 0.0

    for _ in range(max_iter):
        # utility_e,d = -theta * (c_e + V_head,d)
        utility = -theta * (tt[:, None] + V[heads, :])
        utility = np.clip(utility, EXP_CLIP_MIN, EXP_CLIP_MAX)
        z = np.exp(utility)  # [E, D]
        if reasonable_mask is not None:
            z = z * reasonable_mask

        # S_u,d = sum_{e from u} z_e,d
        S = out_mat @ z  # [N, D]
        V_new = -np.log(np.maximum(S, EPS)) / theta

        V_new[dest_nodes, np.arange(num_dests)] = 0.0

        unreachable = S <= EPS
        if np.any(unreachable):
            V_new[unreachable] = V_MAX
            V_new[dest_nodes, np.arange(num_dests)] = 0.0

        rel = np.linalg.norm(V_new - V) / (np.linalg.norm(V) + 1.0)
        V = V_new
        if rel < tol:
            break

    return V


def _markov_logit_network_loading(
    travel_times,
    od_matrix,
    tails,
    heads,
    out_mat,
    in_mat,
    theta=0.8,
    value_iter=250,
    value_tol=1e-8,
    flow_iter=500,
    flow_tol=1e-9,
):
    num_nodes = out_mat.shape[0]
    num_edges = tails.shape[0]

    od = np.asarray(od_matrix, dtype=np.float64)
    dest_nodes = _centroid_destination_nodes(od, num_nodes)
    num_dests = dest_nodes.shape[0]
    tt = np.asarray(travel_times, dtype=np.float64).reshape(num_edges)
    reasonable_mask = _compute_reasonable_link_mask(
        travel_times=tt,
        tails=tails,
        heads=heads,
        num_nodes=num_nodes,
        dest_nodes=dest_nodes,
    )

    V = _solve_recursive_logit_values(
        travel_times=tt,
        tails=tails,
        heads=heads,
        out_mat=out_mat,
        dest_nodes=dest_nodes,
        theta=theta,
        reasonable_mask=reasonable_mask,
        max_iter=value_iter,
        tol=value_tol,
    )  # [N, D]

    utility = -theta * (tt[:, None] + V[heads, :])
    utility = np.clip(utility, EXP_CLIP_MIN, EXP_CLIP_MAX)
    z = np.exp(utility)  # [E, D]
    z = z * reasonable_mask
    S = out_mat @ z      # [N, D]
    p = np.zeros_like(z)
    denom = S[tails, :]
    valid = denom > EPS
    p[valid] = z[valid] / denom[valid]

    is_out_of_dest = tails[:, None] == dest_nodes[None, :]
    p[is_out_of_dest] = 0.0
    p = np.clip(p, 0.0, 1.0)

    q = np.zeros((num_nodes, num_dests), dtype=np.float64)
    q[:num_dests, :] = od
    q[dest_nodes, np.arange(num_dests)] = 0.0

    x = q.copy()  
    loading_converged = False
    for _ in range(flow_iter):
        edge_flow_by_dest = x[tails, :] * p        # [E, D]
        x_new = q + (in_mat @ edge_flow_by_dest)   # [N, D]

        x_new = np.maximum(x_new, 0.0)

        rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1.0)
        x = x_new
        if rel < flow_tol:
            loading_converged = True
            break

    edge_flow_by_dest = x[tails, :] * p
    flows = np.sum(edge_flow_by_dest, axis=1)  # [E]
    flows = np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(flows, 0.0), loading_converged


def _relative_gap(flows, aux_flows, travel_times):
    x = np.asarray(flows, dtype=np.float64)
    y = np.asarray(aux_flows, dtype=np.float64)
    t = np.asarray(travel_times, dtype=np.float64)

    flow_gap = np.linalg.norm(y - x) / (np.linalg.norm(x) + EPS)
    cx = float(np.dot(t, x))
    cy = float(np.dot(t, y))
    cost_gap = abs(cx - cy) / max(abs(cx), EPS)
    return flow_gap, cost_gap


def _msa_sr_step(iteration, flow_gap, prev_flow_gap, beta=1.2, gamma=0.72):
    k = max(int(iteration), 1)
    alpha = beta / (k ** gamma)

    if np.isfinite(prev_flow_gap) and prev_flow_gap > 0.0:
        ratio = flow_gap / prev_flow_gap
        if ratio > 1.02:
            alpha *= 0.60
        elif ratio > 0.98:
            alpha *= 0.85
        elif ratio < 0.70:
            alpha *= 1.08

    return float(np.clip(alpha, 0.02, 0.80))


def markov_logit_sue_solver(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    max_iter=120,
    convergence_threshold=1e-5,
    verbose=True,
    theta=0.8,
    bpr_alpha=0.15,
    bpr_beta=4.0,
    value_iter=250,
    value_tol=1e-8,
    flow_iter=500,
    flow_tol=1e-9,
):
    edges = list(G.edges())
    num_edges = len(edges)
    if num_edges == 0:
        return np.zeros(0, dtype=np.float64)

    # Graph edge arrays 
    tails = np.asarray([u - 1 for u, _ in edges], dtype=np.int64)
    heads = np.asarray([v - 1 for _, v in edges], dtype=np.int64)
    num_nodes = int(max(np.max(tails), np.max(heads)) + 1)

    out_mat, in_mat = _build_sparse_edge_incidence(num_nodes, tails, heads)

    cap = np.asarray(capacities, dtype=np.float64).reshape(num_edges)
    t0 = np.asarray(free_flow_times, dtype=np.float64).reshape(num_edges)
    cap = np.maximum(cap, EPS)
    t0 = np.maximum(t0, EPS)

    loading_warning_count = 0
    flows, loading_converged = _markov_logit_network_loading(
        travel_times=t0,
        od_matrix=od_matrix,
        tails=tails,
        heads=heads,
        out_mat=out_mat,
        in_mat=in_mat,
        theta=theta,
        value_iter=value_iter,
        value_tol=value_tol,
        flow_iter=flow_iter,
        flow_tol=flow_tol,
    )
    if not loading_converged:
        loading_warning_count += 1

    prev_flow_gap = np.inf
    for it in range(1, max_iter + 1):
        travel_times = bpr_travel_time(flows, cap, t0, alpha=bpr_alpha, beta=bpr_beta)

        aux_flows, loading_converged = _markov_logit_network_loading(
            travel_times=travel_times,
            od_matrix=od_matrix,
            tails=tails,
            heads=heads,
            out_mat=out_mat,
            in_mat=in_mat,
            theta=theta,
            value_iter=value_iter,
            value_tol=value_tol,
            flow_iter=flow_iter,
            flow_tol=flow_tol,
        )
        if not loading_converged:
            loading_warning_count += 1

        flow_gap, cost_gap = _relative_gap(flows, aux_flows, travel_times)
        step = _msa_sr_step(it, flow_gap, prev_flow_gap, beta=1.2, gamma=0.72)

        new_flows = (1.0 - step) * flows + step * aux_flows
        new_flows = np.maximum(np.nan_to_num(new_flows, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        update_gap = np.linalg.norm(new_flows - flows) / (np.linalg.norm(flows) + EPS)
        flows = new_flows
        prev_flow_gap = flow_gap

        if verbose and (it == 1 or it % 5 == 0):
            print(
                f"    Iter {it:03d} | step={step:.4f} | "
                f"flow_gap={flow_gap:.6e} | cost_gap={cost_gap:.6e} | update_gap={update_gap:.6e}"
            )

        if max(flow_gap, cost_gap, update_gap) < convergence_threshold:
            if verbose:
                print(f"  SUE converged at iter {it}, gap={max(flow_gap, cost_gap, update_gap):.3e}")
            break

    if loading_warning_count > 0:
        warnings.warn(
            (
                f"Markov loading remained non-convergent in {loading_warning_count} inner solve(s) "
                f"with flow_iter={flow_iter}. Increase flow_iter or relax flow_tol for larger networks."
            ),
            MarkovLoadingConvergenceWarning,
            stacklevel=2,
        )

    return flows


def advanced_sue_solver(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    max_iter=120,
    convergence_threshold=1e-5,
    verbose=True,
    theta=0.8,
    value_iter=250,
    value_tol=1e-8,
    flow_iter=500,
    flow_tol=1e-9,
):
    return markov_logit_sue_solver(
        G=G,
        od_matrix=od_matrix,
        capacities=capacities,
        free_flow_times=free_flow_times,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        verbose=verbose,
        theta=theta,
        value_iter=value_iter,
        value_tol=value_tol,
        flow_iter=flow_iter,
        flow_tol=flow_tol,
    )


def frank_wolfe_sue(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    max_iter=120,
    convergence_threshold=1e-5,
    verbose=True,
    value_iter=250,
    value_tol=1e-8,
    flow_iter=500,
    flow_tol=1e-9,
):
    return markov_logit_sue_solver(
        G=G,
        od_matrix=od_matrix,
        capacities=capacities,
        free_flow_times=free_flow_times,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        verbose=verbose,
        value_iter=value_iter,
        value_tol=value_tol,
        flow_iter=flow_iter,
        flow_tol=flow_tol,
    )


def solve_sue_batch(
    G,
    od_matrices,
    capacities,
    speeds,
    method='frank_wolfe',
    verbose=True,
    value_iter=250,
    value_tol=1e-8,
    flow_iter=500,
    flow_tol=1e-9,
):
    try:
        from .utils import compute_free_flow_times
    except ImportError:
        from utils import compute_free_flow_times

    num_samples = int(od_matrices.shape[0])
    num_edges = len(list(G.edges()))

    print(f"\n{'=' * 60}")
    print(f"Solving SUE for {num_samples} scenarios using '{method}' method")
    print(f"{'=' * 60}")

    print("  Computing free-flow times...")
    free_flow_times = compute_free_flow_times(G, speeds)

    all_flows = np.zeros((num_samples, num_edges), dtype=np.float64)

    method = str(method).lower()
    if method in ('frank_wolfe', 'advanced', 'markov_logit', 'msa_sr'):
        solver_func = frank_wolfe_sue
    else:
        raise ValueError(f"Unknown method: {method}")

    print("  Running traffic assignment...")
    for i in tqdm(range(num_samples), desc="  Progress", disable=not verbose):
        all_flows[i] = solver_func(
            G=G,
            od_matrix=od_matrices[i],
            capacities=capacities[i],
            free_flow_times=free_flow_times[i],
            max_iter=120,
            convergence_threshold=1e-5,
            verbose=False,
            value_iter=value_iter,
            value_tol=value_tol,
            flow_iter=flow_iter,
            flow_tol=flow_tol,
        )

    print("\n SUE solving completed!")
    print("  Flow statistics:")
    print(f"    Min: {all_flows.min():.2f}")
    print(f"    Max: {all_flows.max():.2f}")
    print(f"    Mean: {all_flows.mean():.2f}")
    print(f"    Std: {all_flows.std():.2f}")
    return all_flows


def save_flows(flows, save_path='processed_data/raw/flows.npz'):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, flows=flows)
    print(f"\n Flows saved to: {save_path}")


def load_flows(load_path='processed_data/raw/flows.npz'):
    data = np.load(load_path)
    print(f"\n Flows loaded from: {load_path}")
    return data['flows']
