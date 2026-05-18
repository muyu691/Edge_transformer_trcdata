from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class RegressionAccumulator:
    count: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_true: float = 0.0
    sum_true_sq: float = 0.0
    sum_abs_true: float = 0.0

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        pred = pred.view(-1).double()
        true = true.view(-1).double()
        if pred.numel() == 0:
            return
        error = pred - true
        self.count += int(pred.numel())
        self.sum_abs_error += float(error.abs().sum().item())
        self.sum_squared_error += float(error.pow(2).sum().item())
        self.sum_true += float(true.sum().item())
        self.sum_true_sq += float(true.pow(2).sum().item())
        self.sum_abs_true += float(true.abs().sum().item())

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "wmape": 0.0}
        mae = self.sum_abs_error / self.count
        rmse = math.sqrt(self.sum_squared_error / self.count)
        denom = self.sum_true_sq - (self.sum_true ** 2) / self.count
        r2 = 0.0 if denom <= 1e-12 else 1.0 - (self.sum_squared_error / denom)
        wmape = 0.0 if self.sum_abs_true <= 1e-12 else self.sum_abs_error / self.sum_abs_true
        return {
            "count": int(self.count),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "wmape": float(wmape),
        }


@dataclass
class MetricBundle:
    all_edges: RegressionAccumulator = field(default_factory=RegressionAccumulator)
    old_edges: RegressionAccumulator = field(default_factory=RegressionAccumulator)
    new_edges: RegressionAccumulator = field(default_factory=RegressionAccumulator)

    def update(self, pred: torch.Tensor, true: torch.Tensor, new_edge_mask: torch.Tensor) -> None:
        pred = pred.view(-1)
        true = true.view(-1)
        mask_new = new_edge_mask.view(-1).bool()
        mask_old = ~mask_new

        self.all_edges.update(pred, true)
        self.old_edges.update(pred[mask_old], true[mask_old])
        self.new_edges.update(pred[mask_new], true[mask_new])

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            "all_edges": self.all_edges.as_dict(),
            "old_edges": self.old_edges.as_dict(),
            "new_edges": self.new_edges.as_dict(),
        }


@dataclass
class SplitMetrics:
    normalized: MetricBundle = field(default_factory=MetricBundle)
    real: MetricBundle = field(default_factory=MetricBundle)

    def as_dict(self) -> dict[str, dict]:
        return {
            "normalized": self.normalized.as_dict(),
            "real": self.real.as_dict(),
        }
