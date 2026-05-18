from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv

from baseline.model_common import EdgeAlignmentModule, ModelConfig


class NodeCentricOldEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.edge_to_weight = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.layers = nn.ModuleList(
            [GraphConv(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers_old)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers_old)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        edge_index_old: torch.Tensor,
        edge_attr_old: torch.Tensor,
        flow_old: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        device = edge_attr_old.device
        dtype = edge_attr_old.dtype
        x = torch.ones((num_nodes, self.hidden_dim), device=device, dtype=dtype)
        edge_weight = self.edge_to_weight(torch.cat([edge_attr_old, flow_old], dim=-1)).squeeze(-1)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index_old, edge_weight=edge_weight)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + residual
        return x


class NodeCentricNewReasoner(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.node_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.edge_to_weight = nn.Sequential(
            nn.Linear(8, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.layers = nn.ModuleList(
            [GraphConv(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers_new)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers_new)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.edge_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 8, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        edge_index_new: torch.Tensor,
        aligned: torch.Tensor,
        h_nodes_old: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        device = h_nodes_old.device
        dtype = h_nodes_old.dtype
        x_init = torch.ones((num_nodes, self.hidden_dim), device=device, dtype=dtype)
        x = self.node_fusion(torch.cat([x_init, h_nodes_old], dim=-1))
        edge_weight = self.edge_to_weight(aligned).squeeze(-1)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index_new, edge_weight=edge_weight)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + residual

        src, dst = edge_index_new[0], edge_index_new[1]
        edge_repr = torch.cat([x[src], x[dst], aligned], dim=-1)
        return self.edge_decoder(edge_repr)


class NodeCentricGNN(nn.Module):
    """GraphConv baseline with node-centric aggregation and edge decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = NodeCentricOldEncoder(config)
        self.aligner = EdgeAlignmentModule()
        self.reasoner = NodeCentricNewReasoner(config)

    def forward(self, batch) -> torch.Tensor:
        h_nodes_old = self.encoder(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            num_nodes=batch.num_nodes,
        )
        aligned = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=batch.num_nodes,
        )
        return self.reasoner(
            edge_index_new=batch.edge_index_new,
            aligned=aligned,
            h_nodes_old=h_nodes_old,
            num_nodes=batch.num_nodes,
        )
