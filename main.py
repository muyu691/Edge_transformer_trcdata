import datetime
import logging
import os

import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    makedirs_rm_exist,
    set_cfg,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import (
    OptimizerConfig,
    create_optimizer,
    create_scheduler,
)
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

import graphgps  # noqa: F401
from graphgps.logger import create_logger
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig


def agg_runs(out_dir, metric_best):
    logging.info("Skipping run aggregation.")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        wsd_stable_epochs=cfg.optim.wsd_stable_epochs,
        wsd_decay_epochs=cfg.optim.wsd_decay_epochs,
        wsd_decay_type=cfg.optim.wsd_decay_type,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    if len(cfg.run_multiple_splits) == 0:
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        if args.repeat != 1:
            raise NotImplementedError(
                "Running multiple repeats of multiple splits in one run is not supported."
            )
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == "__main__":
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    torch.set_num_threads(cfg.num_threads)

    for run_id, seed, split_index in zip(*run_loop_settings()):
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        logging.info(
            "[*] Run ID %s: seed=%s, split_index=%s",
            run_id,
            cfg.seed,
            cfg.dataset.split_index,
        )
        logging.info("    Starting now: %s", datetime.datetime.now())

        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info("Num parameters: %s", cfg.params)

        if cfg.train.mode == "standard":
            if cfg.wandb.use:
                logging.warning(
                    "[W] WandB logging is not supported with train.mode=standard; "
                    "use train.mode=custom instead."
                )
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as exc:
        logging.info("Failed when trying to aggregate multiple runs: %s", exc)

    if args.mark_done:
        os.rename(args.cfg_file, f"{args.cfg_file}_done")
    logging.info("[*] All done: %s", datetime.datetime.now())
