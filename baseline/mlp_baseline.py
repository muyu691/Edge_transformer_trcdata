from __future__ import annotations

import torch
import torch.nn as nn

from baseline.model_common import EdgeAlignmentModule, ModelConfig


class MLPBaseline(nn.Module):
    """Edge-wise baseline without graph message passing."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.aligner = EdgeAlignmentModule()

        layers: list[nn.Module] = []
        in_dim = 8
        depth = max(int(config.mlp_depth), 2)
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        aligned = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=batch.num_nodes,
        )
        return self.mlp(aligned)
