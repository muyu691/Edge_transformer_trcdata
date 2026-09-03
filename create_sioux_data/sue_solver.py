import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


EPS = 1e-12
V_MAX = 1e6
EXP_CLIP_MIN = -700.0
EXP_CLIP_MAX = 60.0


class SUEConvergenceError(RuntimeError):
    """Raised when a Markov-logit SUE solve fails a required convergence check."""

    def __init__(self, message, *, stage, diagnostics=None):
        super().__init__(message)
        self.stage = str(stage)
        self.diagnostics = dict(diagnostics or {})


def bpr_travel_time(flow, capacity, free_flow_time, alpha=0.15, beta=4.0):
    """
    BPR 
    t = t0 * (1 + alpha * (flow / capacity)^beta)
    """
    flow = np.asarray(flow, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    free_flow_time = np.asarray(free_flow_time, dtype=np.float64)
    if not np.all(np.isfinite(flow)) or np.any(flow < 0.0):
        raise ValueError("flow must contain finite, non-negative values.")
    if not np.all(np.isfinite(capacity)) or np.any(capacity <= 0.0):
        raise ValueError("capacity must contain finite, strictly positive values.")
    if not np.all(np.isfinite(free_flow_time)) or np.any(free_flow_time <= 0.0):
        raise ValueError("free_flow_time must contain finite, strictly positive values.")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("BPR alpha must be finite and non-negative.")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("BPR beta must be finite and strictly positive.")

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            travel_time = free_flow_time * (1.0 + alpha * ((flow / capacity) ** beta))
        except FloatingPointError as exc:
            raise FloatingPointError("BPR travel-time evaluation overflowed or became invalid.") from exc
    if not np.all(np.isfinite(travel_time)) or np.any(travel_time <= 0.0):
        raise FloatingPointError("BPR travel-time evaluation produced invalid values.")
    return travel_time


def _build_sparse_edge_incidence(num_nodes, tails, heads):
    num_edges = tails.shape[0]
    edge_ids = np.arange(num_edges, dtype=np.int64)
    ones = np.ones(num_edges, dtype=np.float64)
    out_mat = sp.csr_matrix((ones, (tails, edge_ids)), shape=(num_nodes, num_edges))
    in_mat = sp.csr_matrix((ones, (heads, edge_ids)), shape=(num_nodes, num_edges))
    return out_mat, in_mat


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

    converged = False
    for _ in range(max_iter):
        # utility_e,d = -theta * (c_e + V_head,d)
        utility = -theta * (tt[:, None] + V[heads, :])
        utility = np.clip(utility, EXP_CLIP_MIN, EXP_CLIP_MAX)
        z = np.exp(utility)  # [E, D]
        if reasonable_mask is not None:
            z = z * reasonable_mask

        # S_u,d = sum_{e from u} z_e,d
        S = out_mat @ z  # [N, D]
        # EXP_CLIP_MIN keeps every representable, permitted transition
        # strictly positive.  Using EPS here would incorrectly classify small
        # but valid probabilities as unreachable and silently discard demand.
        V_new = -np.log(np.maximum(S, np.finfo(np.float64).tiny)) / theta

        V_new[dest_nodes, np.arange(num_dests)] = 0.0

        unreachable = S <= 0.0
        if np.any(unreachable):
            V_new[unreachable] = V_MAX
            V_new[dest_nodes, np.arange(num_dests)] = 0.0

        rel = np.linalg.norm(V_new - V) / (np.linalg.norm(V) + 1.0)
        if not np.isfinite(rel) or not np.all(np.isfinite(V_new)):
            raise FloatingPointError("Recursive-logit value iteration produced NaN or Inf.")
        V = V_new
        if rel <= tol:
            converged = True
            break

    return V, converged


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
    loading_protocol="reasonable_links",
    centroid_indices=None,
):
    num_nodes = out_mat.shape[0]
    num_edges = tails.shape[0]

    od = np.asarray(od_matrix, dtype=np.float64)
    if centroid_indices is None:
        raise ValueError("centroid_indices must be supplied explicitly.")
    dest_nodes = np.asarray(centroid_indices, dtype=np.int64).reshape(-1)
    num_dests = dest_nodes.shape[0]
    if od.shape != (num_dests, num_dests):
        raise ValueError(
            f"od_matrix must have shape ({num_dests}, {num_dests}), got {od.shape}."
        )
    tt = np.asarray(travel_times, dtype=np.float64).reshape(num_edges)
    if loading_protocol in {"reasonable_links", "stable_reasonable_links"}:
        reasonable_mask = _compute_reasonable_link_mask(
            travel_times=tt,
            tails=tails,
            heads=heads,
            num_nodes=num_nodes,
            dest_nodes=dest_nodes,
        )
    elif loading_protocol in {"legacy_unrestricted", "stable_unrestricted"}:
        reasonable_mask = None
    else:
        raise ValueError(f"Unsupported SUE loading protocol: {loading_protocol!r}")

    V, value_converged = _solve_recursive_logit_values(
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
    if reasonable_mask is not None:
        z = z * reasonable_mask
    S = out_mat @ z      # [N, D]
    if loading_protocol == "legacy_unrestricted":
        # Exact loading rule used by datasets created before the reasonable-link
        # mask was introduced. Keep this branch for dataset reproducibility.
        p = z / np.maximum(S[tails, :], EPS)
    elif loading_protocol == "reasonable_links":
        p = np.zeros_like(z)
        denom = S[tails, :]
        valid = denom > 0.0
        p[valid] = z[valid] / denom[valid]
    else:
        # A masked row can have a valid but extremely small denominator.  The
        # old EPS cutoff silently discarded its OD demand.  Normalize every
        # representable positive row for a flow-conserving initialization.
        p = np.zeros_like(z)
        denom = S[tails, :]
        valid = denom > 0.0
        p[valid] = z[valid] / denom[valid]

    is_out_of_dest = tails[:, None] == dest_nodes[None, :]
    p[is_out_of_dest] = 0.0
    p = np.clip(p, 0.0, 1.0)

    # Every non-destination state must distribute all probability over its
    # outgoing links.  Otherwise the loading can appear converged while OD
    # demand has simply vanished through an underflowed or unreachable row.
    row_probability = out_mat @ p
    expected_probability = np.ones_like(row_probability)
    expected_probability[dest_nodes, np.arange(num_dests)] = 0.0
    transition_converged = bool(
        np.all(np.isfinite(row_probability))
        and np.max(np.abs(row_probability - expected_probability)) <= 1e-8
    )

    q = np.zeros((num_nodes, num_dests), dtype=np.float64)
    q[dest_nodes, :] = od
    q[dest_nodes, np.arange(num_dests)] = 0.0

    x = q.copy()  
    loading_converged = False
    for _ in range(flow_iter):
        edge_flow_by_dest = x[tails, :] * p        # [E, D]
        x_new = q + (in_mat @ edge_flow_by_dest)   # [N, D]

        if not np.all(np.isfinite(x_new)):
            raise FloatingPointError("Markov flow loading produced NaN or Inf.")
        if np.any(x_new < -EPS):
            raise FloatingPointError("Markov flow loading produced negative state flow.")
        x_new = np.maximum(x_new, 0.0)

        rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1.0)
        x = x_new
        if not np.isfinite(rel):
            raise FloatingPointError("Markov flow-loading residual became NaN or Inf.")
        if rel <= flow_tol:
            loading_converged = True
            break

    edge_flow_by_dest = x[tails, :] * p
    flows = np.sum(edge_flow_by_dest, axis=1)  # [E]
    if not np.all(np.isfinite(flows)):
        raise FloatingPointError("Markov network loading produced NaN or Inf edge flow.")
    if np.any(flows < -EPS):
        raise FloatingPointError("Markov network loading produced negative edge flow.")
    loading_converged = bool(value_converged and transition_converged and loading_converged)
    return np.maximum(flows, 0.0), loading_converged


def _relative_gap(flows, aux_flows, travel_times):
    x = np.asarray(flows, dtype=np.float64)
    y = np.asarray(aux_flows, dtype=np.float64)
    t = np.asarray(travel_times, dtype=np.float64)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)):
        raise FloatingPointError("Cannot compute a convergence gap from NaN or Inf values.")
    flow_gap = np.linalg.norm(y - x) / (np.linalg.norm(x) + EPS)
    cx = float(np.dot(t, x))
    cy = float(np.dot(t, y))
    cost_gap = abs(cx - cy) / max(abs(cx), EPS)
    if not np.isfinite(flow_gap) or not np.isfinite(cost_gap):
        raise FloatingPointError("SUE convergence gap became NaN or Inf.")
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


def _residual_aware_step(
    iteration,
    flow_gap,
    cost_gap,
    prev_flow_gap,
    exponent=0.5,
    min_step=0.02,
    max_step=0.80,
):
    """Cap MSA-SR by the current fixed-point residual.

    The common MSA-SR schedule starts at 0.8 regardless of initialization
    quality.  This cap keeps that behavior for large-residual states while
    preserving more of a state that already has a small equilibrium residual.
    No extra network loading is needed to select the step.
    """
    if exponent <= 0.0:
        raise ValueError("residual_step_exponent must be positive.")
    if not (0.0 < min_step <= max_step <= 1.0):
        raise ValueError(
            "Residual-aware step bounds must satisfy 0 < min_step <= max_step <= 1."
        )

    base_step = _msa_sr_step(iteration, flow_gap, prev_flow_gap)
    residual = max(float(flow_gap), float(cost_gap), 0.0)
    residual_cap = float(np.clip(residual ** exponent, min_step, max_step))
    return float(min(base_step, residual_cap))


def _prepare_solver_inputs(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    node_ids,
    centroid_nodes,
):
    """Validate inputs and map arbitrary graph node ids to contiguous indices."""
    if not G.is_directed():
        raise ValueError("Markov-logit SUE requires a directed graph.")
    if G.is_multigraph():
        raise ValueError("Markov-logit SUE currently requires a DiGraph, not a MultiDiGraph.")

    edges = list(G.edges())
    if not edges:
        raise ValueError("Cannot solve SUE on a graph with no directed edges.")
    if not nx.is_strongly_connected(G):
        raise ValueError("Cannot solve SUE on a graph that is not strongly connected.")

    if node_ids is None:
        raise ValueError("node_ids must be supplied explicitly.")
    if centroid_nodes is None:
        raise ValueError("centroid_nodes must be supplied explicitly.")
    node_ids = tuple(node_ids)
    centroid_nodes = tuple(centroid_nodes)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node_ids contains duplicates.")
    if set(node_ids) != set(G.nodes()):
        missing = set(G.nodes()) - set(node_ids)
        extra = set(node_ids) - set(G.nodes())
        raise ValueError(
            f"node_ids must match graph nodes exactly (missing={sorted(missing)}, extra={sorted(extra)})."
        )
    if len(centroid_nodes) != len(set(centroid_nodes)):
        raise ValueError("centroid_nodes contains duplicates.")
    unknown_centroids = set(centroid_nodes) - set(node_ids)
    if unknown_centroids:
        raise ValueError(f"centroid_nodes contains graph-unknown ids: {sorted(unknown_centroids)}.")

    od = np.asarray(od_matrix, dtype=np.float64)
    num_centroids = len(centroid_nodes)
    if od.shape != (num_centroids, num_centroids):
        raise ValueError(
            f"od_matrix must have shape ({num_centroids}, {num_centroids}), got {od.shape}."
        )
    if not np.all(np.isfinite(od)) or np.any(od < 0.0):
        raise ValueError("od_matrix must contain finite, non-negative demand.")
    od = od.copy()
    np.fill_diagonal(od, 0.0)

    num_edges = len(edges)
    cap = np.asarray(capacities, dtype=np.float64).reshape(-1)
    t0 = np.asarray(free_flow_times, dtype=np.float64).reshape(-1)
    if cap.shape != (num_edges,):
        raise ValueError(f"capacities must have shape ({num_edges},), got {cap.shape}.")
    if t0.shape != (num_edges,):
        raise ValueError(f"free_flow_times must have shape ({num_edges},), got {t0.shape}.")
    if not np.all(np.isfinite(cap)) or np.any(cap <= 0.0):
        raise ValueError("capacities must contain finite, strictly positive values.")
    if not np.all(np.isfinite(t0)) or np.any(t0 <= 0.0):
        raise ValueError("free_flow_times must contain finite, strictly positive values.")

    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    tails = np.asarray([node_id_to_index[u] for u, _ in edges], dtype=np.int64)
    heads = np.asarray([node_id_to_index[v] for _, v in edges], dtype=np.int64)
    centroid_indices = np.asarray(
        [node_id_to_index[node_id] for node_id in centroid_nodes],
        dtype=np.int64,
    )
    return edges, od, cap, t0, tails, heads, centroid_indices


def markov_logit_sue_solver(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    *,
    node_ids,
    centroid_nodes,
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
    initial_flows=None,
    return_diagnostics=False,
    loading_protocol="reasonable_links",
    initial_flow_mode="direct",
    initial_loading_protocol=None,
    step_rule="msa_sr",
    residual_step_exponent=0.5,
    residual_step_min=0.02,
    residual_step_max=0.80,
    stop_before_update=False,
    step_warmup_iters=0,
    step_warmup_max=1.0,
):
    if max_iter <= 0 or value_iter <= 0 or flow_iter <= 0:
        raise ValueError("max_iter, value_iter, and flow_iter must be positive.")
    if not np.isfinite(convergence_threshold) or convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be finite and positive.")
    if not np.isfinite(value_tol) or value_tol <= 0.0:
        raise ValueError("value_tol must be finite and positive.")
    if not np.isfinite(flow_tol) or flow_tol <= 0.0:
        raise ValueError("flow_tol must be finite and positive.")
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and positive.")
    network_loading_calls = 0
    negative_initial_flow_count = 0
    if initial_flow_mode not in {"direct", "cost_loaded"}:
        raise ValueError(f"Unsupported initial_flow_mode: {initial_flow_mode!r}")
    if step_rule not in {"msa_sr", "residual_aware"}:
        raise ValueError(f"Unsupported SUE step_rule: {step_rule!r}")
    if step_warmup_iters < 0:
        raise ValueError("step_warmup_iters must be non-negative.")
    if not (0.0 < step_warmup_max <= 1.0):
        raise ValueError("step_warmup_max must satisfy 0 < value <= 1.")
    resolved_initial_loading_protocol = initial_loading_protocol or loading_protocol

    edges, od, cap, t0, tails, heads, centroid_indices = _prepare_solver_inputs(
        G=G,
        od_matrix=od_matrix,
        capacities=capacities,
        free_flow_times=free_flow_times,
        node_ids=node_ids,
        centroid_nodes=centroid_nodes,
    )
    num_edges = len(edges)
    num_nodes = G.number_of_nodes()
    out_mat, in_mat = _build_sparse_edge_incidence(num_nodes, tails, heads)

    if initial_flows is None:
        initialization = "free_flow_loading"
        flows, loading_converged = _markov_logit_network_loading(
            travel_times=t0,
            od_matrix=od,
            tails=tails,
            heads=heads,
            out_mat=out_mat,
            in_mat=in_mat,
            theta=theta,
            value_iter=value_iter,
            value_tol=value_tol,
            flow_iter=flow_iter,
            flow_tol=flow_tol,
            loading_protocol=resolved_initial_loading_protocol,
            centroid_indices=centroid_indices,
        )
        network_loading_calls += 1
        if not loading_converged:
            raise SUEConvergenceError(
                "Initial Markov-logit network loading did not converge.",
                stage="initial_loading",
                diagnostics={
                    "network_loading_calls": network_loading_calls,
                    "value_iter": int(value_iter),
                    "flow_iter": int(flow_iter),
                    "loading_protocol": resolved_initial_loading_protocol,
                },
            )
    else:
        guidance_flows = np.asarray(initial_flows, dtype=np.float64).reshape(-1)
        if guidance_flows.shape != (num_edges,):
            raise ValueError(
                f"initial_flows must have shape ({num_edges},), got {guidance_flows.shape}."
            )
        if not np.all(np.isfinite(guidance_flows)):
            raise ValueError("initial_flows contains NaN or Inf.")
        negative_initial_flow_count = int(np.count_nonzero(guidance_flows < 0.0))
        guidance_flows = np.maximum(guidance_flows, 0.0)

        if initial_flow_mode == "direct":
            initialization = "provided_direct"
            flows = guidance_flows.copy()
        else:
            initialization = "provided_cost_loading"
            guidance_times = bpr_travel_time(
                guidance_flows,
                cap,
                t0,
                alpha=bpr_alpha,
                beta=bpr_beta,
            )
            flows, loading_converged = _markov_logit_network_loading(
                travel_times=guidance_times,
                od_matrix=od,
                tails=tails,
                heads=heads,
                out_mat=out_mat,
                in_mat=in_mat,
                theta=theta,
                value_iter=value_iter,
                value_tol=value_tol,
                flow_iter=flow_iter,
                flow_tol=flow_tol,
                loading_protocol=resolved_initial_loading_protocol,
                centroid_indices=centroid_indices,
            )
            network_loading_calls += 1
            if not loading_converged:
                raise SUEConvergenceError(
                    "Warm-start Markov-logit network loading did not converge.",
                    stage="initial_loading",
                    diagnostics={
                        "network_loading_calls": network_loading_calls,
                        "value_iter": int(value_iter),
                        "flow_iter": int(flow_iter),
                        "loading_protocol": resolved_initial_loading_protocol,
                    },
                )

    initial_state_flows = flows.copy()

    prev_flow_gap = np.inf
    outer_candidate_converged = False
    iterations = 0
    initial_flow_gap = np.nan
    initial_cost_gap = np.nan
    final_update_gap = np.inf
    step_sizes = []
    for it in range(1, max_iter + 1):
        iterations = it
        travel_times = bpr_travel_time(flows, cap, t0, alpha=bpr_alpha, beta=bpr_beta)

        aux_flows, loading_converged = _markov_logit_network_loading(
            travel_times=travel_times,
            od_matrix=od,
            tails=tails,
            heads=heads,
            out_mat=out_mat,
            in_mat=in_mat,
            theta=theta,
            value_iter=value_iter,
            value_tol=value_tol,
            flow_iter=flow_iter,
            flow_tol=flow_tol,
            loading_protocol=loading_protocol,
            centroid_indices=centroid_indices,
        )
        network_loading_calls += 1
        if not loading_converged:
            raise SUEConvergenceError(
                f"Markov-logit network loading did not converge at outer iteration {it}.",
                stage="outer_loading",
                diagnostics={
                    "outer_iteration": int(it),
                    "network_loading_calls": network_loading_calls,
                    "value_iter": int(value_iter),
                    "flow_iter": int(flow_iter),
                    "loading_protocol": loading_protocol,
                },
            )

        flow_gap, cost_gap = _relative_gap(flows, aux_flows, travel_times)
        if it == 1:
            initial_flow_gap = float(flow_gap)
            initial_cost_gap = float(cost_gap)
        if stop_before_update and max(flow_gap, cost_gap) <= convergence_threshold:
            # The current state is already a fixed point to the requested
            # tolerance.  Do not overwrite a converged warm start merely to
            # perform a nominal outer update.
            iterations = it - 1
            final_update_gap = 0.0
            outer_candidate_converged = True
            if verbose:
                print(
                    f"  SUE converged before update {it}, "
                    f"gap={max(flow_gap, cost_gap):.3e}"
                )
            break
        if step_rule == "msa_sr":
            step = _msa_sr_step(it, flow_gap, prev_flow_gap, beta=1.2, gamma=0.72)
        else:
            step = _residual_aware_step(
                it,
                flow_gap,
                cost_gap,
                prev_flow_gap,
                exponent=residual_step_exponent,
                min_step=residual_step_min,
                max_step=residual_step_max,
            )
        if it <= step_warmup_iters:
            step = min(step, step_warmup_max)
        step_sizes.append(step)

        new_flows = (1.0 - step) * flows + step * aux_flows
        if not np.all(np.isfinite(new_flows)):
            raise FloatingPointError("Outer SUE update produced NaN or Inf.")
        if np.any(new_flows < -EPS):
            raise FloatingPointError("Outer SUE update produced negative flow.")
        new_flows = np.maximum(new_flows, 0.0)

        update_gap = np.linalg.norm(new_flows - flows) / (np.linalg.norm(flows) + EPS)
        if not np.isfinite(update_gap):
            raise FloatingPointError("Outer SUE update residual became NaN or Inf.")
        flows = new_flows
        prev_flow_gap = flow_gap
        final_update_gap = float(update_gap)

        if verbose and (it == 1 or it % 5 == 0):
            print(
                f"    Iter {it:03d} | step={step:.4f} | "
                f"flow_gap={flow_gap:.6e} | cost_gap={cost_gap:.6e} | update_gap={update_gap:.6e}"
            )

        if max(flow_gap, cost_gap, update_gap) <= convergence_threshold:
            outer_candidate_converged = True
            if verbose:
                print(f"  SUE converged at iter {it}, gap={max(flow_gap, cost_gap, update_gap):.3e}")
            break

    # Verify the returned state itself, rather than accepting residuals that
    # were measured before the final outer update.
    final_travel_times = bpr_travel_time(
        flows,
        cap,
        t0,
        alpha=bpr_alpha,
        beta=bpr_beta,
    )
    final_aux_flows, final_loading_converged = _markov_logit_network_loading(
        travel_times=final_travel_times,
        od_matrix=od,
        tails=tails,
        heads=heads,
        out_mat=out_mat,
        in_mat=in_mat,
        theta=theta,
        value_iter=value_iter,
        value_tol=value_tol,
        flow_iter=flow_iter,
        flow_tol=flow_tol,
        loading_protocol=loading_protocol,
        centroid_indices=centroid_indices,
    )
    network_loading_calls += 1
    if not final_loading_converged:
        raise SUEConvergenceError(
            "Final fixed-point Markov-logit loading did not converge.",
            stage="final_loading",
            diagnostics={
                "iterations": int(iterations),
                "network_loading_calls": network_loading_calls,
                "value_iter": int(value_iter),
                "flow_iter": int(flow_iter),
                "loading_protocol": loading_protocol,
            },
        )

    final_flow_gap, final_cost_gap = _relative_gap(
        flows,
        final_aux_flows,
        final_travel_times,
    )
    final_metric = float(max(final_flow_gap, final_cost_gap, final_update_gap))
    converged = bool(np.isfinite(final_metric) and final_metric <= convergence_threshold)
    diagnostics = {
        "converged": bool(converged),
        "outer_candidate_converged": bool(outer_candidate_converged),
        "final_fixed_point_verified": bool(converged),
        "iterations": int(iterations),
        "network_loading_calls": int(network_loading_calls),
        "initialization": initialization,
        "initial_flow_gap": float(initial_flow_gap),
        "initial_cost_gap": float(initial_cost_gap),
        "final_flow_gap": float(final_flow_gap),
        "final_cost_gap": float(final_cost_gap),
        "final_update_gap": float(final_update_gap),
        "final_convergence_metric": final_metric,
        "loading_warning_count": 0,
        "negative_initial_flow_count": int(negative_initial_flow_count),
        "loading_protocol": loading_protocol,
        "initial_flow_mode": initialization,
        "initial_loading_protocol": resolved_initial_loading_protocol,
        "step_rule": step_rule,
        "initial_step_size": float(step_sizes[0]) if step_sizes else 0.0,
        "mean_step_size": float(np.mean(step_sizes)) if step_sizes else 0.0,
        "residual_step_exponent": float(residual_step_exponent),
        "residual_step_min": float(residual_step_min),
        "residual_step_max": float(residual_step_max),
        "stop_before_update": bool(stop_before_update),
        "initial_prior_weight": float(1.0 - step_sizes[0]) if step_sizes else 1.0,
        "step_warmup_iters": int(step_warmup_iters),
        "step_warmup_max": float(step_warmup_max),
        "initial_state_flows": initial_state_flows,
    }
    if not converged:
        raise SUEConvergenceError(
            (
                "Markov-logit SUE did not satisfy the final fixed-point tolerance: "
                f"flow_gap={final_flow_gap:.6e}, cost_gap={final_cost_gap:.6e}, "
                f"update_gap={final_update_gap:.6e}, threshold={convergence_threshold:.6e}."
            ),
            stage="final_fixed_point",
            diagnostics=diagnostics,
        )
    return (flows, diagnostics) if return_diagnostics else flows


def save_flows(flows, save_path='processed_data/raw/flows.npz'):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, flows=flows)
    print(f"\n Flows saved to: {save_path}")


def load_flows(load_path='processed_data/raw/flows.npz'):
    data = np.load(load_path)
    print(f"\n Flows loaded from: {load_path}")
    return data['flows']
