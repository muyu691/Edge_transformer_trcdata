import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


EPS = 1e-12
SOLVER_VERSION = "markov_logit_logspace_v3"


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


def _allowed_link_mask(G, node_ids, dest_nodes):
    """A non-through zone may emit its own demand, but may only be entered
    when it is the destination. FIRST THRU NODE=1 leaves TNTP roads unrestricted.
    """
    first_thru = int(G.graph.get("first_thru_node", 1))
    if first_thru <= 1:
        return np.ones((G.number_of_edges(), len(dest_nodes)), dtype=bool)
    node_index = {node: i for i, node in enumerate(node_ids)}
    heads = np.asarray([node_index[v] for _, v in G.edges()])
    restricted = np.asarray([node < first_thru for node in node_ids])
    return ~restricted[heads, None] | (heads[:, None] == dest_nodes[None, :])


def _compute_reasonable_link_mask(travel_times, tails, heads, num_nodes, dest_nodes,
                                  allowed_mask=None, od_matrix=None, downhill=True):
    """
    Keep only links that move closer to each destination.

    Dense cyclic networks with small link costs can make unrestricted
    recursive-logit loading circulate indefinitely instead of being absorbed at
    the destination.  A destination-specific downhill mask preserves stochastic
    route choice over reasonable links while making the Markov chain proper.
    """
    num_edges = tails.shape[0]
    tt = np.asarray(travel_times, dtype=np.float64).reshape(num_edges)

    if allowed_mask is None or np.all(allowed_mask):
        reverse_graph = sp.csr_matrix((tt, (heads, tails)), shape=(num_nodes, num_nodes))
        dist = np.atleast_2d(dijkstra(reverse_graph, directed=True, indices=dest_nodes)).T
    else:
        # Distances must obey the SAME zone restriction as the probabilities;
        # masking forbidden links only after shortest paths can strand a zone.
        dist = np.empty((num_nodes, len(dest_nodes)))
        for column, destination in enumerate(dest_nodes):
            keep = allowed_mask[:, column]
            reverse_graph = sp.csr_matrix(
                (tt[keep], (heads[keep], tails[keep])), shape=(num_nodes, num_nodes),
            )
            dist[:, column] = dijkstra(reverse_graph, directed=True, indices=destination)

    tail_dist = dist[tails, :]
    head_dist = dist[heads, :]
    reasonable = np.isfinite(tail_dist) & np.isfinite(head_dist)
    if downhill:
        reasonable &= head_dist < tail_dist
    reasonable[tails[:, None] == dest_nodes[None, :]] = False
    if allowed_mask is not None:
        reasonable &= allowed_mask

    # Never insert a non-downhill fallback edge: it can create a cycle and
    # silently change the declared choice set. Positive costs imply a downhill
    # shortest-path successor at every reachable non-destination node.
    counts = np.zeros((num_nodes, len(dest_nodes)), dtype=np.int64)
    np.add.at(counts, tails, reasonable)
    counts[dest_nodes, np.arange(len(dest_nodes))] = 1
    # A connector leading only into a different zone can legitimately be
    # unreachable. Only positive-demand OD origins MUST reach this destination.
    if od_matrix is not None:
        required = np.asarray(od_matrix) > 0.0
        np.fill_diagonal(required, False)
        if np.any(required & (counts[dest_nodes, :] == 0)):
            raise ValueError("A positive-demand OD has no legal route without zone transit.")

    return reasonable


def _logit_rows(travel_times, values, tails, heads, out_mat, dest_nodes,
                theta, reasonable_mask):
    """Segmented log-sum-exp and softmax, shifted BEFORE exponentiation.

    Rows are (tail node, destination). Absolute utilities are never clipped;
    only negligible relative probabilities may naturally underflow to zero.
    Destination rows are absorbing and deliberately have no outgoing mass.
    """
    utility = -theta * (travel_times[:, None] + values[heads, :])
    permitted = tails[:, None] != dest_nodes[None, :]
    if reasonable_mask is not None:
        permitted = permitted & reasonable_mask
    if not np.all(np.isfinite(utility[permitted])):
        raise FloatingPointError("Non-finite recursive-logit utility on a permitted edge.")
    utility = np.where(permitted, utility, -np.inf)
    row_max = np.full(values.shape, -np.inf)
    np.maximum.at(row_max, tails, utility)
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    weights = np.exp(utility - safe_max[tails, :])
    row_sum = out_mat @ weights
    reachable = row_sum > 0.0
    required = (out_mat @ permitted) > 0
    required[dest_nodes, np.arange(len(dest_nodes))] = False
    if np.any(required & ~reachable):
        raise SUEConvergenceError(
            "A non-destination state has no representable path to its destination.",
            stage="transition_normalization",
        )
    log_sum = safe_max + np.log(np.where(reachable, row_sum, 1.0))
    updated = -log_sum / theta
    updated[dest_nodes, np.arange(len(dest_nodes))] = 0.0
    probabilities = weights / np.where(reachable, row_sum, 1.0)[tails, :]
    return updated, probabilities


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
        V_new, _ = _logit_rows(tt, V, tails, heads, out_mat, dest_nodes,
                               theta, reasonable_mask)
        # Absolute error in log-utility units; dividing by a huge congested
        # value could hide probability-relevant errors.
        rel = theta * np.max(np.abs(V_new - V))
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
    fixed_reasonable_mask=None,
    return_diagnostics=False,
    allowed_mask=None,
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
        if fixed_reasonable_mask is None:
            reasonable_mask = _compute_reasonable_link_mask(
                travel_times=tt,
                tails=tails,
                heads=heads,
                num_nodes=num_nodes,
                dest_nodes=dest_nodes,
                allowed_mask=allowed_mask,
                od_matrix=od,
            )
        else:
            reasonable_mask = np.asarray(fixed_reasonable_mask, dtype=bool)
            expected_shape = (num_edges, num_dests)
            if reasonable_mask.shape != expected_shape:
                raise ValueError(
                    "fixed_reasonable_mask must have shape "
                    f"{expected_shape}, got {reasonable_mask.shape}."
                )
    elif loading_protocol == "stable_unrestricted":
        reasonable_mask = None
    else:
        raise ValueError(f"Unsupported SUE loading protocol: {loading_protocol!r}")

    if allowed_mask is not None:
        reasonable_mask = (allowed_mask if reasonable_mask is None
                           else reasonable_mask & allowed_mask)

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

    checked_values, p = _logit_rows(tt, V, tails, heads, out_mat, dest_nodes,
                                   theta, reasonable_mask)
    bellman_residual = float(theta * np.max(np.abs(checked_values - V)))
    if not value_converged or bellman_residual > value_tol:
        raise SUEConvergenceError(
            "Recursive-logit values did not satisfy the Bellman tolerance.",
            stage="recursive_values",
            diagnostics={"bellman_residual": bellman_residual,
                         "value_iter": int(value_iter), "value_tol": float(value_tol)},
        )

    # Every non-destination state must distribute all probability over its
    # outgoing links.  Otherwise the loading can appear converged while OD
    # demand has simply vanished through an underflowed or unreachable row.
    row_probability = out_mat @ p
    expected_probability = (np.ones_like(row_probability) if reasonable_mask is None
                            else ((out_mat @ reasonable_mask) > 0).astype(float))
    expected_probability[dest_nodes, np.arange(num_dests)] = 0.0
    transition_error = float(np.max(np.abs(row_probability - expected_probability)))
    transition_converged = bool(
        np.all(np.isfinite(row_probability))
        and transition_error <= 1e-12
    )

    q = np.zeros((num_nodes, num_dests), dtype=np.float64)
    q[dest_nodes, :] = od
    q[dest_nodes, np.arange(num_dests)] = 0.0
    if np.any((q > 0.0) & (expected_probability == 0.0)):
        raise SUEConvergenceError("Positive demand has no legal destination-reaching route.",
                                  stage="od_reachability")

    x = q.copy()  
    loading_converged = False
    for loading_iteration in range(1, flow_iter + 1):
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
    incoming = in_mat @ edge_flow_by_dest
    outgoing = out_mat @ edge_flow_by_dest
    demand = od.sum(axis=0) - np.diag(od)
    arrivals = incoming[dest_nodes, np.arange(num_dests)]
    arrival_error = float(np.max(np.abs(arrivals - demand) / np.maximum(demand, 1.0)))
    balance = outgoing - incoming - q
    balance[dest_nodes, np.arange(num_dests)] += demand
    balance_error = float(np.max(np.abs(balance) / np.maximum(demand[None, :], 1.0)))
    state_residual = float(np.linalg.norm(q + incoming - x) / (np.linalg.norm(x) + 1.0))
    diagnostics = {
        "bellman_residual": bellman_residual,
        "transition_error": transition_error,
        "flow_loading_iterations": loading_iteration,
        "flow_loading_residual": state_residual,
        "destination_arrival_error": arrival_error,
        "destination_balance_error": balance_error,
    }
    if not (transition_converged and loading_converged
            and max(arrival_error, balance_error, state_residual) <= flow_tol):
        raise SUEConvergenceError(
            "Markov loading failed probability, flow-balance, or destination-arrival checks.",
            stage="flow_loading", diagnostics=diagnostics,
        )
    flows = np.sum(edge_flow_by_dest, axis=1)  # [E]
    if not np.all(np.isfinite(flows)):
        raise FloatingPointError("Markov network loading produced NaN or Inf edge flow.")
    if np.any(flows < -EPS):
        raise FloatingPointError("Markov network loading produced negative edge flow.")
    loading_converged = bool(value_converged and transition_converged and loading_converged)
    result = (np.maximum(flows, 0.0), loading_converged)
    return (*result, diagnostics) if return_diagnostics else result


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

    # A positive hard floor prevents the MSA sequence from tending to zero and
    # can sustain a fixed-point oscillation.  Keep only the upper safeguard;
    # alpha is strictly positive for every finite iteration.
    return float(min(alpha, 0.80))


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
    if num_centroids == 0:
        raise ValueError("centroid_nodes must not be empty.")
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
    stop_before_update=True,
    step_warmup_iters=0,
    step_warmup_max=1.0,
    verify_only=False,
    secant_safeguard=True,
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
    supported_protocols = {"reasonable_links", "stable_reasonable_links", "stable_unrestricted"}
    for protocol in (loading_protocol, initial_loading_protocol or loading_protocol):
        if protocol not in supported_protocols:
            raise ValueError(
                f"Unsupported SUE loading protocol {protocol!r}. The unsafe legacy "
                "EPS/clipping loader is retired; use an explicit supported protocol."
            )
    if verify_only and (initial_flows is None or initial_flow_mode != "direct"):
        raise ValueError("verify_only requires an unchanged, direct initial_flows array.")
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
    allowed_mask = _allowed_link_mask(G, node_ids, centroid_indices)
    if not np.all(allowed_mask):
        # Prune destination-unreachable connector states for both protocols.
        allowed_mask = _compute_reasonable_link_mask(
            t0, tails, heads, num_nodes, centroid_indices,
            allowed_mask=allowed_mask, od_matrix=od, downhill=False,
        )

    # Freeze the destination-specific reasonable-link choice set for the whole
    # outer solve.  Recomputing this discrete mask from congested costs at every
    # iteration makes the fixed-point map discontinuous: links can repeatedly
    # enter and leave the choice set near equilibrium, so a strict tolerance may
    # be impossible to satisfy.  Free-flow costs provide a scenario-specific,
    # flow-independent set. This is explicitly a fixed-free-flow choice-set
    # model, not the old dynamically reselected mask or unrestricted SUE.
    fixed_reasonable_mask = None
    reasonable_protocols = {"reasonable_links", "stable_reasonable_links"}
    if (
        loading_protocol in reasonable_protocols
        or resolved_initial_loading_protocol in reasonable_protocols
    ):
        fixed_reasonable_mask = _compute_reasonable_link_mask(
            travel_times=t0,
            tails=tails,
            heads=heads,
            num_nodes=num_nodes,
            dest_nodes=centroid_indices,
            allowed_mask=allowed_mask,
            od_matrix=od,
        )

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
            fixed_reasonable_mask=fixed_reasonable_mask,
            allowed_mask=allowed_mask,
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
        if verify_only and negative_initial_flow_count:
            raise ValueError("Cannot verify a label containing negative flows.")
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
                fixed_reasonable_mask=fixed_reasonable_mask,
                allowed_mask=allowed_mask,
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
    final_update_gap = 0.0  # No outer update has been performed yet.
    step_sizes = []
    previous_state = None
    previous_residual = None
    safeguarded_steps = 0
    for it in range(1, 1 if verify_only else max_iter + 1):
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
            fixed_reasonable_mask=fixed_reasonable_mask,
            allowed_mask=allowed_mask,
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
        if (stop_before_update or it > 1) and max(flow_gap, cost_gap) <= convergence_threshold:
            # The current state is already a fixed point to the requested
            # tolerance.  Do not overwrite a converged warm start merely to
            # perform a nominal outer update. When stop_before_update=False,
            # require one update, then check its result at the next loading.
            iterations = it - 1
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
        # BPR can make the residual extremely sensitive after a closure. A
        # secant estimate caps the step when the current MSA schedule is too
        # aggressive, using existing loadings only. This is a numerical
        # safeguard, not a convergence proof; final verification remains mandatory.
        residual = aux_flows - flows
        if secant_safeguard and previous_state is not None:
            state_change = np.linalg.norm(flows - previous_state)
            residual_change = np.linalg.norm(residual - previous_residual)
            if residual_change > 0.0 and state_change > 0.0:
                cap_step = 0.8 * state_change / residual_change
                if cap_step < step:
                    step = float(cap_step)
                    safeguarded_steps += 1
        previous_state = flows.copy()
        previous_residual = residual.copy()
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

        # Pre-update gaps cannot certify new_flows. The next iteration (or the
        # final verification at the budget limit) must load the updated state.

    # Verify the returned state itself, rather than accepting residuals that
    # were measured before the final outer update.
    final_travel_times = bpr_travel_time(
        flows,
        cap,
        t0,
        alpha=bpr_alpha,
        beta=bpr_beta,
    )
    final_aux_flows, final_loading_converged, loading_diagnostics = _markov_logit_network_loading(
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
        fixed_reasonable_mask=fixed_reasonable_mask,
        allowed_mask=allowed_mask,
        return_diagnostics=True,
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
    net_demand = np.zeros(num_nodes)
    net_demand[centroid_indices] = od.sum(axis=1) - od.sum(axis=0)
    balance = out_mat @ flows - in_mat @ flows - net_demand
    conservation_error = float(np.max(np.abs(balance)) / max(float(od.sum()), 1.0))
    # Fixed-point residual is the acceptance criterion. Last-update magnitude
    # is recorded honestly as a diagnostic, not overwritten on early exit.
    final_metric = float(max(final_flow_gap, final_cost_gap))
    converged = bool(np.isfinite(final_metric) and final_metric <= convergence_threshold
                     and conservation_error <= flow_tol)
    diagnostics = {
        "solver_version": SOLVER_VERSION,
        "first_thru_node": int(G.graph.get("first_thru_node", 1)),
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
        "final_conservation_error": conservation_error,
        "final_conservation_max_abs": float(np.max(np.abs(balance))),
        "convergence_threshold": float(convergence_threshold),
        "theta": float(theta),
        "bpr_alpha": float(bpr_alpha),
        "bpr_beta": float(bpr_beta),
        "value_tol": float(value_tol),
        "flow_tol": float(flow_tol),
        "value_iter": int(value_iter),
        "flow_iter": int(flow_iter),
        "max_iter": int(max_iter),
        "verification_only": bool(verify_only),
        "loading_diagnostics": loading_diagnostics,
        "loading_warning_count": 0,
        "negative_initial_flow_count": int(negative_initial_flow_count),
        "loading_protocol": loading_protocol,
        "reasonable_link_basis": (
            "free_flow_time" if loading_protocol in reasonable_protocols else None
        ),
        "initial_flow_mode": initialization,
        "initial_loading_protocol": resolved_initial_loading_protocol,
        "step_rule": step_rule,
        "secant_safeguard": bool(secant_safeguard),
        "safeguarded_steps": int(safeguarded_steps),
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
                f"update_gap={final_update_gap:.6e}, threshold={convergence_threshold:.6e}, "
                f"conservation_error={conservation_error:.6e}."
            ),
            stage="final_fixed_point",
            diagnostics=diagnostics,
        )
    return (flows, diagnostics) if return_diagnostics else flows


def verify_sue_solution(G, od_matrix, capacities, free_flow_times, flows, **params):
    """Revalidate an existing label under current equations; never update it."""
    _, diagnostics = markov_logit_sue_solver(
        G, od_matrix, capacities, free_flow_times, initial_flows=flows,
        verify_only=True, return_diagnostics=True, verbose=False, **params,
    )
    diagnostics.pop("initial_state_flows", None)
    return diagnostics


def save_flows(flows, save_path='processed_data/raw/flows.npz'):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, flows=flows)
    print(f"\n Flows saved to: {save_path}")


def load_flows(load_path='processed_data/raw/flows.npz'):
    data = np.load(load_path)
    print(f"\n Flows loaded from: {load_path}")
    return data['flows']
