from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def compute_constraint_violation_stats(
    pred_real: torch.Tensor,
    edge_index_new: torch.Tensor,
    net_demand: torch.Tensor,
    node_batch: torch.Tensor,
    ptr: torch.Tensor,
) -> list[dict[str, float]]:
    pred_real = pred_real.view(-1).double()
    edge_index_new = edge_index_new.long()
    net_demand = net_demand.view(-1).double()
    node_batch = node_batch.view(-1).long()
    ptr = ptr.view(-1).long()

    num_nodes = int(net_demand.numel())
    residual = (-net_demand).clone()
    src = edge_index_new[0].view(-1).long()
    dst = edge_index_new[1].view(-1).long()
    residual.index_add_(0, dst, pred_real)
    residual.index_add_(0, src, -pred_real)

    stats: list[dict[str, float]] = []
    num_graphs = max(int(ptr.numel()) - 1, 0)
    eps = 1e-12
    for graph_idx in range(num_graphs):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        if end <= start:
            continue
        graph_residual = residual[start:end]
        graph_demand = net_demand[start:end]
        abs_residual = graph_residual.abs()
        con_mae = float(abs_residual.mean().item())
        con_rmse = float(torch.sqrt(graph_residual.pow(2).mean()).item())
        con_max = float(abs_residual.max().item())
        denom = float(graph_demand.abs().sum().item())
        relcon = 0.0 if denom <= eps else float(abs_residual.sum().item() / denom)
        stats.append(
            {
                "graph_index": graph_idx,
                "con_mae": con_mae,
                "con_rmse": con_rmse,
                "con_max": con_max,
                "relcon": relcon,
            }
        )
    return stats


@dataclass
class ConstraintViolationAccumulator:
    con_mae_values: list[float] = field(default_factory=list)
    con_rmse_values: list[float] = field(default_factory=list)
    con_max_values: list[float] = field(default_factory=list)
    relcon_values: list[float] = field(default_factory=list)

    def update(
        self,
        pred_real: torch.Tensor,
        edge_index_new: torch.Tensor,
        net_demand: torch.Tensor,
        node_batch: torch.Tensor,
        ptr: torch.Tensor,
    ) -> None:
        batch_stats = compute_constraint_violation_stats(
            pred_real=pred_real,
            edge_index_new=edge_index_new,
            net_demand=net_demand,
            node_batch=node_batch,
            ptr=ptr,
        )
        for item in batch_stats:
            self.con_mae_values.append(float(item["con_mae"]))
            self.con_rmse_values.append(float(item["con_rmse"]))
            self.con_max_values.append(float(item["con_max"]))
            self.relcon_values.append(float(item["relcon"]))

    def as_dict(self) -> dict[str, float]:
        if not self.relcon_values:
            return {
                "num_graphs": 0,
                "con_mae": 0.0,
                "con_rmse": 0.0,
                "con_max": 0.0,
                "relcon": 0.0,
                "relcon_p95": 0.0,
            }
        relcon_tensor = torch.tensor(self.relcon_values, dtype=torch.float64)
        relcon_p95 = float(torch.quantile(relcon_tensor, 0.95).item())
        return {
            "num_graphs": int(len(self.relcon_values)),
            "con_mae": float(sum(self.con_mae_values) / len(self.con_mae_values)),
            "con_rmse": float(sum(self.con_rmse_values) / len(self.con_rmse_values)),
            "con_max": float(sum(self.con_max_values) / len(self.con_max_values)),
            "relcon": float(sum(self.relcon_values) / len(self.relcon_values)),
            "relcon_p95": relcon_p95,
        }
