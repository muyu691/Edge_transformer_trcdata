from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible baseline runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    aliases = {
        "mlp": "mlp_baseline",
        "mlp_baseline": "mlp_baseline",
        "single_topology_gatedgcn": "single_topology_gatedgcn",
        "gatedgcn": "single_topology_gatedgcn",
        "nodecentricgnn": "node_centric_gnn",
        "node_centric_gnn": "node_centric_gnn",
        "node-centric-gnn": "node_centric_gnn",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Choose from: mlp_baseline, single_topology_gatedgcn, NodeCentricGNN."
        )
    return aliases[key]


def match_edge_indices(
    edge_index_old: Tensor,
    edge_index_new: Tensor,
    total_nodes: int,
) -> Tensor:
    """Match each new-graph edge to the old-graph row index, or -1 if absent."""
    e_new = edge_index_new.size(1)
    match_idx = torch.full(
        (e_new,),
        fill_value=-1,
        dtype=torch.long,
        device=edge_index_new.device,
    )
    if edge_index_old.numel() == 0 or e_new == 0:
        return match_idx

    base = int(total_nodes)
    old_keys = edge_index_old[0].long() * base + edge_index_old[1].long()
    new_keys = edge_index_new[0].long() * base + edge_index_new[1].long()

    sorted_old_keys, perm = torch.sort(old_keys)
    positions = torch.searchsorted(sorted_old_keys, new_keys)
    valid = positions < sorted_old_keys.numel()
    if not valid.any():
        return match_idx

    valid_positions = positions[valid]
    matched = sorted_old_keys[valid_positions] == new_keys[valid]
    if matched.any():
        valid_rows = valid.nonzero(as_tuple=False).view(-1)
        match_idx[valid_rows[matched]] = perm[valid_positions[matched]]
    return match_idx


def compute_new_edge_mask(
    edge_index_old: Tensor,
    edge_index_new: Tensor,
    total_nodes: int,
) -> Tensor:
    """Return a boolean mask marking edges present only in the new graph."""
    return match_edge_indices(
        edge_index_old=edge_index_old,
        edge_index_new=edge_index_new,
        total_nodes=total_nodes,
    ) < 0


def denormalize_flow(flow_scaled: Tensor, flow_mean: float, flow_std: float) -> Tensor:
    """Undo the training-set flow normalization."""
    return flow_scaled * float(flow_std) + float(flow_mean)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
