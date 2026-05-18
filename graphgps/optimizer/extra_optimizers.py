import logging
import math
from typing import Iterator
from dataclasses import dataclass

import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.graphgym.optim import SchedulerConfig
import torch_geometric.graphgym.register as register


@register.register_optimizer('adagrad')
def adagrad_optimizer(params: Iterator[Parameter], base_lr: float,
                      weight_decay: float) -> Adagrad:
    return Adagrad(params, lr=base_lr, weight_decay=weight_decay)


@register.register_optimizer('adamW')
def adamW_optimizer(params: Iterator[Parameter], base_lr: float,
                   weight_decay: float) -> AdamW:
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)



@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
    num_warmup_epochs: int = 10
    wsd_stable_epochs: int = 0
    wsd_decay_epochs: int = 50
    wsd_decay_type: str = "cosine"
    train_mode: str = 'custom'
    eval_period: int = 1


@register.register_scheduler('plateau')
def plateau_scheduler(optimizer: Optimizer, patience: int,
                      lr_decay: float) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)


@register.register_scheduler('reduce_on_plateau')
def scheduler_reduce_on_plateau(optimizer: Optimizer, reduce_factor: float,
                                schedule_patience: int, min_lr: float,
                                train_mode: str, eval_period: int):
    if train_mode == 'standard':
        raise ValueError("ReduceLROnPlateau scheduler is not supported "
                         "by 'standard' graphgym training mode pipeline; "
                         "try setting config 'train.mode: custom'")

    if eval_period != 1:
        logging.warning("When config train.eval_period is not 1, the "
                        "optim.schedule_patience of ReduceLROnPlateau "
                        "may not behave as intended.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=reduce_factor,
        patience=schedule_patience,
        min_lr=min_lr,
        verbose=True
    )
    if not hasattr(scheduler, 'get_last_lr'):
        # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
        # pytorch1.10; we add it here for consistency with other schedulers.
        def get_last_lr(self):
            """ Return last computed learning rate by current scheduler.
            """
            return self._last_lr

        scheduler.get_last_lr = get_last_lr.__get__(scheduler)
        scheduler._last_lr = [group['lr']
                              for group in scheduler.optimizer.param_groups]

    def modified_state_dict(ref):
        """Returns the state of the scheduler as a :class:`dict`.
        Additionally modified to ignore 'get_last_lr', 'state_dict'.
        Including these entries in the state dict would cause issues when
        loading a partially trained / pretrained model from a checkpoint.
        """
        return {key: value for key, value in ref.__dict__.items()
                if key not in ['sparsifier', 'get_last_lr', 'state_dict']}

    scheduler.state_dict = modified_state_dict.__get__(scheduler)

    return scheduler


@register.register_scheduler('linear_with_warmup')
def linear_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


@register.register_scheduler('cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


@register.register_scheduler('wsd')
def warmup_stable_decay_scheduler(
        optimizer: Optimizer,
        num_warmup_epochs: int,
        max_epoch: int,
        min_lr: float,
        wsd_stable_epochs: int,
        wsd_decay_epochs: int,
        wsd_decay_type: str):
    scheduler = get_wsd_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch,
        min_lr=min_lr,
        stable_steps=wsd_stable_epochs,
        decay_steps=wsd_decay_epochs,
        decay_type=wsd_decay_type,
    )
    return scheduler


@register.register_scheduler('warmup_stable_decay')
def warmup_stable_decay_scheduler_alias(
        optimizer: Optimizer,
        num_warmup_epochs: int,
        max_epoch: int,
        min_lr: float,
        wsd_stable_epochs: int,
        wsd_decay_epochs: int,
        wsd_decay_type: str):
    return warmup_stable_decay_scheduler(
        optimizer=optimizer,
        num_warmup_epochs=num_warmup_epochs,
        max_epoch=max_epoch,
        min_lr=min_lr,
        wsd_stable_epochs=wsd_stable_epochs,
        wsd_decay_epochs=wsd_decay_epochs,
        wsd_decay_type=wsd_decay_type,
    )


@register.register_scheduler('polynomial_with_warmup')
def polynomial_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


def get_linear_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_wsd_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float,
        stable_steps: int = 0,
        decay_steps: int = 50,
        decay_type: str = "cosine",
        last_epoch: int = -1):
    """
    Warmup-Stable-Decay schedule.

    Phase 1 warms up linearly to each param group's initial LR, phase 2 keeps
    it stable, and phase 3 decays it to cfg.optim.min_lr.
    """
    num_training_steps = max(int(num_training_steps), 1)
    warmup_steps = min(max(int(num_warmup_steps), 0), num_training_steps)
    decay_steps = max(int(decay_steps), 0)
    stable_steps = max(int(stable_steps), 0)

    if stable_steps == 0:
        if decay_steps == 0:
            decay_steps = max(num_training_steps - warmup_steps, 1)
        stable_steps = max(num_training_steps - warmup_steps - decay_steps, 0)
    elif decay_steps == 0:
        decay_steps = max(num_training_steps - warmup_steps - stable_steps, 1)

    if warmup_steps + stable_steps + decay_steps > num_training_steps:
        stable_steps = max(num_training_steps - warmup_steps - decay_steps, 0)
    if warmup_steps + stable_steps + decay_steps > num_training_steps:
        decay_steps = max(num_training_steps - warmup_steps - stable_steps, 1)

    stable_end = warmup_steps + stable_steps
    decay_end = stable_end + decay_steps
    decay_type = str(decay_type).lower()
    if decay_type not in ("cosine", "linear"):
        raise ValueError("wsd_decay_type must be one of: 'cosine', 'linear'")

    def _make_lr_lambda(base_lr: float):
        min_ratio = float(min_lr) / max(float(base_lr), 1e-12)
        min_ratio = min(max(min_ratio, 0.0), 1.0)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return max(min_ratio, float(current_step) / float(max(1, warmup_steps)))
            if current_step < stable_end:
                return 1.0
            if current_step >= decay_end:
                return min_ratio

            progress = float(current_step - stable_end) / float(max(1, decay_steps))
            progress = min(max(progress, 0.0), 1.0)
            if decay_type == "linear":
                decay_multiplier = 1.0 - progress
            else:
                decay_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * decay_multiplier

        return lr_lambda

    lr_lambdas = [_make_lr_lambda(group["lr"]) for group in optimizer.param_groups]
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py
    
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
