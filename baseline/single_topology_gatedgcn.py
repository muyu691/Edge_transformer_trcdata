from __future__ import annotations

import torch
import torch.nn as nn

from graphgps.layer.gatedgcn_layer import GatedGCNLayer

from baseline.model_common import EdgeAlignmentModule, ModelConfig, _GNNBatch


class SingleTopologyGatedGCN(nn.Module):
    """GatedGCN baseline that only propagates on the new topology."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.aligner = EdgeAlignmentModule()
        self.node_enc = nn.Linear(1, config.hidden_dim)
        self.edge_enc = nn.Linear(8, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                GatedGCNLayer(
                    in_dim=config.hidden_dim,
                    out_dim=config.hidden_dim,
                    dropout=config.dropout,
                    residual=config.residual,
                )
                for _ in range(config.num_layers_new)
            ]
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self._replace_batchnorm_with_layernorm()

    def _replace_batchnorm_with_layernorm(self) -> None:
        def _replace(module: nn.Module) -> None:
            for name, child in module.named_children():
                if "BatchNorm" in child.__class__.__name__ and hasattr(child, "num_features"):
                    setattr(module, name, nn.LayerNorm(child.num_features))
                else:
                    _replace(child)

        _replace(self)

    def forward(self, batch) -> torch.Tensor:
        aligned = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=batch.num_nodes,
        )
        x = self.node_enc(batch.x)
        e = self.edge_enc(aligned)

        for layer in self.layers:
            mini_batch = _GNNBatch(x=x, edge_attr=e, edge_index=batch.edge_index_new)
            mini_batch = layer(mini_batch)
            x = mini_batch.x
            e = mini_batch.edge_attr
        return self.edge_decoder(e)
