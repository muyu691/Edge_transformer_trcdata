from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from baseline.common import match_edge_indices


OURS_HIDDEN_DIM = 128


class EdgeAlignmentModule(nn.Module):
    """Build the aligned 8-dim edge feature used by all baselines."""

    def forward(
        self,
        edge_index_old: torch.Tensor,
        edge_attr_old: torch.Tensor,
        flow_old: torch.Tensor,
        edge_index_new: torch.Tensor,
        edge_attr_new: torch.Tensor,
        total_nodes: int,
    ) -> torch.Tensor:
        device = edge_attr_new.device
        dtype = edge_attr_new.dtype
        e_new = edge_index_new.size(1)

        match_idx = match_edge_indices(
            edge_index_old=edge_index_old,
            edge_index_new=edge_index_new,
            total_nodes=total_nodes,
        )
        retained_mask = match_idx >= 0

        old_feats = torch.cat([edge_attr_old, flow_old], dim=-1)
        aligned_old = torch.zeros((e_new, 4), dtype=dtype, device=device)
        if retained_mask.any():
            aligned_old[retained_mask] = old_feats[match_idx[retained_mask]]

        is_new_edge = (~retained_mask).to(dtype=dtype).view(-1, 1)
        return torch.cat([aligned_old, edge_attr_new, is_new_edge], dim=-1)


class _GNNBatch:
    __slots__ = ("x", "edge_attr", "edge_index")

    def __init__(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> None:
        self.x = x
        self.edge_attr = edge_attr
        self.edge_index = edge_index


@dataclass
class ModelConfig:
    hidden_dim: int = OURS_HIDDEN_DIM
    dropout: float = 0.1
    residual: bool = True
    num_layers_old: int = 3
    num_layers_new: int = 3
    mlp_depth: int = 3
