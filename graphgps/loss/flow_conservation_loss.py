"""Old diffusion_backbone_only supervision plus optional conservation regularization."""

import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg


def _align_pred_true_shapes(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match singleton dimensions before applying subset masks."""
    if pred.ndim == 1 and true.ndim == 2 and true.shape[1] == 1:
        pred = pred.view(-1, 1)
    elif true.ndim == 1 and pred.ndim == 2 and pred.shape[1] == 1:
        true = true.view(-1, 1)
    return pred, true


def _compute_data_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Dispatch the base supervised term according to cfg.model.loss_fun."""
    loss_fun = str(cfg.model.loss_fun).lower()
    if loss_fun == "l1":
        return F.l1_loss(pred, true)
    if loss_fun == "smoothl1":
        return F.smooth_l1_loss(pred, true)
    if loss_fun == "mse":
        return F.mse_loss(pred, true)
    raise ValueError(f"Unsupported cfg.model.loss_fun for ST-PINN loss: {cfg.model.loss_fun}")


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _compute_subset_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the base loss to one subset only, keeping mean reduction local."""
    mask = mask.bool().view(-1)
    if mask.numel() != pred.shape[0] or mask.numel() != true.shape[0]:
        raise ValueError(
            f"new_edge_mask has {mask.numel()} entries, but pred/true have "
            f"{pred.shape[0]} / {true.shape[0]} rows."
        )
    if not torch.any(mask):
        return _zero_loss(pred)
    return _compute_data_loss(pred[mask], true[mask])


def _get_lambda_new_current() -> float:
    """Linearly ramp the new-edge weight to reproduce the old training schedule."""
    schedule = str(getattr(cfg.model, "lambda_new_schedule", "linear")).lower()
    lambda_start = float(getattr(cfg.model, "lambda_new_start", 1.0))
    lambda_final = float(getattr(cfg.model, "lambda_new_final", 1.0))
    warmup_epochs = max(int(getattr(cfg.model, "lambda_new_warmup_epochs", 50)), 0)
    current_epoch = max(
        float(getattr(cfg.train, "current_epoch", getattr(cfg.optim, "max_epoch", 0))),
        0.0,
    )

    if schedule in ("constant", "none") or warmup_epochs == 0:
        return lambda_final
    if schedule != "linear":
        raise ValueError(f"Unsupported lambda_new schedule: {cfg.model.lambda_new_schedule}")

    progress = min(max(current_epoch / warmup_epochs, 0.0), 1.0)
    return lambda_start + progress * (lambda_final - lambda_start)


def _get_current_epoch() -> float:
    return max(
        float(getattr(cfg.train, "current_epoch", getattr(cfg.optim, "max_epoch", 0))),
        0.0,
    )


def _linear_interpolate(
    start_value: float,
    end_value: float,
    start_epoch: float,
    end_epoch: float,
    current_epoch: float,
) -> float:
    if end_epoch <= start_epoch:
        return end_value
    progress = min(max((current_epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)
    return start_value + progress * (end_value - start_value)


def _get_lambda_con_current() -> float:
    """
    Three-stage conservation schedule:

    1. keep lambda_con at 0 during representation learning;
    2. ramp to a small middle value;
    3. ramp to cfg.model.lambda_con for feasibility fine-tuning.
    """
    schedule = str(getattr(cfg.model, "lambda_con_schedule", "constant")).lower()
    lambda_final = float(getattr(cfg.model, "lambda_con", 0.0))
    current_epoch = _get_current_epoch()

    if schedule in ("constant", "none"):
        return lambda_final
    if schedule == "linear":
        warmup_epochs = max(float(getattr(cfg.model, "lambda_con_warmup_epochs", 50)), 0.0)
        return _linear_interpolate(0.0, lambda_final, 0.0, warmup_epochs, current_epoch)
    if schedule != "staged_linear":
        raise ValueError(f"Unsupported lambda_con schedule: {cfg.model.lambda_con_schedule}")

    zero_epochs = max(float(getattr(cfg.model, "lambda_con_zero_epochs", 50)), 0.0)
    mid_epoch = max(float(getattr(cfg.model, "lambda_con_mid_epoch", 120)), zero_epochs)
    final_epoch = max(
        float(getattr(cfg.model, "lambda_con_final_epoch", getattr(cfg.optim, "max_epoch", 200))),
        mid_epoch,
    )
    lambda_mid = float(getattr(cfg.model, "lambda_con_mid", min(lambda_final, 0.01)))
    lambda_mid = min(lambda_mid, lambda_final) if lambda_final >= 0.0 else max(lambda_mid, lambda_final)

    if current_epoch <= zero_epochs:
        return 0.0
    if current_epoch <= mid_epoch:
        return _linear_interpolate(0.0, lambda_mid, zero_epochs, mid_epoch, current_epoch)
    return _linear_interpolate(lambda_mid, lambda_final, mid_epoch, final_epoch, current_epoch)


def _compute_terminal_balance_residual(pred_scaled: torch.Tensor, batch) -> torch.Tensor:
    r"""
    Compute the terminal node imbalance on the reconfigured graph:

        r(f) = B^{new} f - d

    where
        f : predicted real-valued edge flow on G^{new}
        d : inherited node net-demand proxy from the pre-edit stationary state.
    """
    flow_mean = pred_scaled.new_tensor(float(cfg.dataset.flow_mean))
    flow_scale = pred_scaled.new_tensor(max(float(cfg.dataset.flow_std), 1e-6))

    f_real = pred_scaled * flow_scale + flow_mean
    f_real_flat = f_real.view(-1)

    num_nodes = int(batch.num_nodes)
    src = batch.edge_index_new[0]
    dst = batch.edge_index_new[1]

    inflow = pred_scaled.new_zeros(num_nodes)
    outflow = pred_scaled.new_zeros(num_nodes)
    inflow.scatter_add_(0, dst, f_real_flat)
    outflow.scatter_add_(0, src, f_real_flat)

    net_demand = batch.net_demand.to(device=pred_scaled.device, dtype=pred_scaled.dtype).view(-1)
    return inflow - outflow - net_demand


def _compute_conservation_loss(
    pred_scaled: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Optional conservation regularizer:

        rho^{term} = B^{new} f - d
        L_con = mean((rho^{term} / sigma_f)^2)

    This branch stays available for current ST-PINN experiments, but setting
    cfg.model.lambda_con = 0.0 reproduces the old diffusion_backbone_only
    optimization target exactly.
    """
    rho_terminal = _compute_terminal_balance_residual(pred_scaled, batch)
    flow_scale = pred_scaled.new_tensor(max(float(cfg.dataset.flow_std), 1e-6))
    loss_con = torch.mean((rho_terminal / flow_scale).pow(2))
    return loss_con, rho_terminal.unsqueeze(-1)


def compute_pinn_loss(
    pred: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Old diffusion_backbone_only supervision with optional conservation:

        L = L_data + lambda_con * L_con

    where

        L_data = lambda_old * L_old + lambda_new * L_new

    and L_old / L_new are computed on retained / newly added edges separately.
    """
    true = batch.y
    pred, true = _align_pred_true_shapes(pred, true)

    lambda_old = float(getattr(cfg.model, "lambda_old", 1.0))
    lambda_new_current = _get_lambda_new_current()

    if hasattr(batch, "new_edge_mask"):
        mask_new = batch.new_edge_mask.bool().view(-1)
        mask_old = ~mask_new
        loss_old = _compute_subset_loss(pred, true, mask_old)
        loss_new = _compute_subset_loss(pred, true, mask_new)
        loss_data = lambda_old * loss_old + lambda_new_current * loss_new
    else:
        loss_old = _compute_data_loss(pred, true)
        loss_new = _zero_loss(pred)
        lambda_new_current = 0.0
        loss_data = loss_old

    lambda_con = _get_lambda_con_current()
    loss_con, rho_terminal = _compute_conservation_loss(pred, batch)
    total_loss = loss_data + lambda_con * loss_con

    batch.loss_old = loss_old.detach()
    batch.loss_new = loss_new.detach()
    batch.lambda_new_current = pred.new_tensor(float(lambda_new_current)).detach()
    batch.loss_sup = loss_data.detach()
    batch.loss_data = loss_data.detach()
    batch.loss_con = loss_con.detach()
    batch.lambda_con = pred.new_tensor(float(lambda_con)).detach()
    batch.lambda_con_target = pred.new_tensor(float(getattr(cfg.model, "lambda_con", 0.0))).detach()
    batch.rho_terminal_abs_mean = rho_terminal.abs().mean().detach()
    batch.rho_terminal_abs_max = rho_terminal.abs().max().detach()
    batch.rho_terminal = rho_terminal.detach()

    return total_loss, pred
