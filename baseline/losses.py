from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    loss_name: str = "l1"
    lambda_old: float = 1.0
    lambda_new_start: float = 1.0
    lambda_new_final: float = 1.0
    lambda_new_warmup_epochs: int = 0
    lambda_new_schedule: str = "linear"


def compute_lambda_new(epoch: int, config: LossConfig) -> float:
    schedule = str(config.lambda_new_schedule).lower()
    if schedule in ("constant", "none") or int(config.lambda_new_warmup_epochs) <= 0:
        return float(config.lambda_new_final)
    if schedule != "linear":
        raise ValueError(f"Unsupported lambda_new schedule: {config.lambda_new_schedule}")

    progress = min(max((int(epoch) - 1) / float(config.lambda_new_warmup_epochs), 0.0), 1.0)
    return float(config.lambda_new_start) + progress * (
        float(config.lambda_new_final) - float(config.lambda_new_start)
    )


def _base_loss(pred: torch.Tensor, true: torch.Tensor, loss_name: str) -> torch.Tensor:
    name = str(loss_name).lower()
    if name == "l1":
        return F.l1_loss(pred, true)
    if name == "smoothl1":
        return F.smooth_l1_loss(pred, true)
    if name == "mse":
        return F.mse_loss(pred, true)
    raise ValueError(f"Unsupported loss name: {loss_name}")


def _subset_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    mask = mask.view(-1).bool()
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred.view(-1)
    if true.ndim == 2 and true.shape[1] == 1:
        true = true.view(-1)
    if mask.numel() != pred.shape[0]:
        raise ValueError(
            f"Subset mask has {mask.numel()} entries, but predictions have {pred.shape[0]} rows."
        )
    if not torch.any(mask):
        return pred.new_zeros(())
    return _base_loss(pred[mask], true[mask], loss_name)


def compute_weighted_supervised_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    new_edge_mask: torch.Tensor | None,
    config: LossConfig,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    lambda_new = compute_lambda_new(epoch=epoch, config=config)
    lambda_old = float(config.lambda_old)

    if new_edge_mask is None:
        loss_all = _base_loss(pred, true, config.loss_name)
        stats = {
            "loss_total": float(loss_all.detach().cpu().item()),
            "loss_old": float(loss_all.detach().cpu().item()),
            "loss_new": 0.0,
            "lambda_old": lambda_old,
            "lambda_new": 0.0,
        }
        return loss_all, stats

    mask_new = new_edge_mask.view(-1).bool()
    mask_old = ~mask_new
    loss_old = _subset_loss(pred, true, mask_old, config.loss_name)
    loss_new = _subset_loss(pred, true, mask_new, config.loss_name)
    total = lambda_old * loss_old + lambda_new * loss_new

    stats = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_old": float(loss_old.detach().cpu().item()),
        "loss_new": float(loss_new.detach().cpu().item()),
        "lambda_old": lambda_old,
        "lambda_new": lambda_new,
    }
    return total, stats
