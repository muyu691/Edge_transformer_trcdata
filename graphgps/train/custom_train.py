import json
import logging
import os.path as osp
import time

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from torchmetrics.functional import mean_absolute_error

from constraint_violation.metrics import ConstraintViolationAccumulator
from graphgps.loss.flow_conservation_loss import compute_pinn_loss
from graphgps.metric_wrapper import flow_metric_space_tag, get_flow_metric_tensors, wmape
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name, match_edge_indices


def _compute_loss(pred, batch):
    return compute_pinn_loss(pred, batch)


def _collect_batch_loss_stats(batch):
    stats = {}
    for key in (
        "loss_old",
        "loss_new",
        "lambda_new_current",
        "loss_sup",
        "loss_data",
        "loss_con",
        "lambda_con",
        "lambda_con_target",
        "rho_terminal_abs_mean",
        "rho_terminal_abs_max",
        "h_e_std",
        "h_v_std",
        "f_scaled_std",
        "delta_f_scaled_std",
        "delta_f_scaled_abs_mean",
        "rho_v_std",
        "rho_v_abs_mean",
    ):
        if not hasattr(batch, key):
            continue
        value = getattr(batch, key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        stats[key] = float(value)
    return stats


class _RegressionAccumulator:
    def __init__(self):
        self.count = 0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_true = 0.0
        self.sum_true_sq = 0.0
        self.sum_abs_true = 0.0

    def update(self, pred, true):
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

    def as_dict(self):
        if self.count == 0:
            return {"count": 0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "wmape": 0.0}
        mae = self.sum_abs_error / self.count
        rmse = float(np.sqrt(self.sum_squared_error / self.count))
        denom = self.sum_true_sq - (self.sum_true ** 2) / self.count
        r2 = 0.0 if denom <= 1e-12 else 1.0 - (self.sum_squared_error / denom)
        wmape_value = 0.0 if self.sum_abs_true <= 1e-12 else self.sum_abs_error / self.sum_abs_true
        return {
            "count": int(self.count),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "wmape": float(wmape_value),
        }


class _MetricBundle:
    def __init__(self):
        self.all_edges = _RegressionAccumulator()
        self.old_edges = _RegressionAccumulator()
        self.new_edges = _RegressionAccumulator()

    def update(self, pred, true, new_edge_mask):
        pred = pred.view(-1)
        true = true.view(-1)
        mask_new = new_edge_mask.view(-1).bool()
        mask_old = ~mask_new
        self.all_edges.update(pred, true)
        self.old_edges.update(pred[mask_old], true[mask_old])
        self.new_edges.update(pred[mask_new], true[mask_new])

    def as_dict(self):
        return {
            "all_edges": self.all_edges.as_dict(),
            "old_edges": self.old_edges.as_dict(),
            "new_edges": self.new_edges.as_dict(),
        }


class _SplitMetrics:
    def __init__(self):
        self.normalized = _MetricBundle()
        self.real = _MetricBundle()

    def as_dict(self):
        return {
            "normalized": self.normalized.as_dict(),
            "real": self.real.as_dict(),
        }


def _maybe_cuda_synchronize(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _save_json(path, payload):
    payload = _sanitize_for_json(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    return value


def _save_training_history(perf, full_epoch_times):
    if not perf:
        return

    num_epochs = max(len(split_perf) for split_perf in perf)
    epochs_payload = []
    for epoch_idx in range(num_epochs):
        epoch_value = epoch_idx
        if epoch_idx < len(perf[0]) and isinstance(perf[0][epoch_idx], dict):
            epoch_value = int(perf[0][epoch_idx].get("epoch", epoch_idx))
        epoch_record = {
            "epoch": epoch_value,
            "epoch_seconds": float(full_epoch_times[epoch_idx]) if epoch_idx < len(full_epoch_times) else None,
            "train": perf[0][epoch_idx] if epoch_idx < len(perf[0]) else None,
            "val": perf[1][epoch_idx] if epoch_idx < len(perf[1]) else None,
            "test": perf[2][epoch_idx] if epoch_idx < len(perf[2]) else None,
        }
        epochs_payload.append(epoch_record)

    history_path = osp.join(cfg.run_dir, "history.json")
    _save_json(history_path, {"epochs": epochs_payload})
    logging.info("Saved training history to %s", history_path)


def _load_dataset_metadata():
    metadata_path = osp.join(cfg.dataset.dir, "dataset_meta.json")
    if not osp.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.warning("Failed to read dataset metadata from %s: %s", metadata_path, exc)
        return {}


def _selection_metric_name():
    if cfg.metric_best == "auto":
        return "validation.loss"
    return f"validation.{cfg.metric_best}"


def _select_best_epoch(val_perf):
    best_epoch = int(np.array([vp["loss"] for vp in val_perf]).argmin())
    best_metric_value = float(val_perf[best_epoch]["loss"])
    if cfg.metric_best != "auto":
        metric_values = np.array([vp[cfg.metric_best] for vp in val_perf])
        best_epoch = int(getattr(metric_values, cfg.metric_agg)())
        best_metric_value = float(val_perf[best_epoch][cfg.metric_best])
    return best_epoch, best_metric_value


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iteration, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        loss, pred_score = _compute_loss(pred, batch)
        batch_loss_stats = _collect_batch_loss_stats(batch)
        loss.backward()

        if ((iteration + 1) % batch_accumulation == 0) or (iteration + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()

        logger.update_stats(
            true=true.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **batch_loss_stats,
        )
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split="val"):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        loss, pred_score = _compute_loss(pred, batch)
        batch_loss_stats = _collect_batch_loss_stats(batch)
        logger.update_stats(
            true=true.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **batch_loss_stats,
        )
        time_start = time.time()


def _compute_new_edge_mask(batch):
    match_idx = match_edge_indices(
        edge_index_old=batch.edge_index_old,
        edge_index_new=batch.edge_index_new,
        total_nodes=batch.num_nodes,
    )
    return match_idx < 0


def _unpack_model_output(output):
    if isinstance(output, tuple) and len(output) == 3:
        return output[0], output[1]
    return output


@torch.no_grad()
def detailed_test_evaluation(loader, model, split="test"):
    model.eval()
    device = torch.device(cfg.accelerator)

    per_graph_wmapes = []
    all_preds_new = []
    all_trues_new = []
    all_preds_old = []
    all_trues_old = []
    total_time_ms = 0.0
    total_graphs = 0
    use_cuda = device.type == "cuda"

    for batch in loader:
        batch.to(device)
        total_graphs += batch.num_graphs

        if use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            output = model(batch)
            end_evt.record()
            torch.cuda.synchronize()
            batch_time_ms = start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            output = model(batch)
            batch_time_ms = (time.perf_counter() - t0) * 1000.0

        pred, true = _unpack_model_output(output)
        total_time_ms += batch_time_ms

        pred_real, true_real, _ = get_flow_metric_tensors(
            pred.detach().cpu().float(),
            true.detach().cpu().float(),
        )

        edge_batch = batch.batch[batch.edge_index_new[0]].detach().cpu()
        for graph_idx in range(batch.num_graphs):
            mask_graph = edge_batch == graph_idx
            pred_graph = pred_real[mask_graph]
            true_graph = true_real[mask_graph]
            if true_graph.numel() > 0:
                per_graph_wmapes.append(wmape(pred_graph, true_graph).item())

        if hasattr(batch, "new_edge_mask"):
            is_new_edge = batch.new_edge_mask.bool().detach().cpu()
        else:
            is_new_edge = _compute_new_edge_mask(batch).detach().cpu()

        if is_new_edge.any():
            all_preds_new.append(pred_real[is_new_edge])
            all_trues_new.append(true_real[is_new_edge])
        if (~is_new_edge).any():
            all_preds_old.append(pred_real[~is_new_edge])
            all_trues_old.append(true_real[~is_new_edge])

    q_values = None
    w_tensor = None
    if per_graph_wmapes:
        w_tensor = torch.tensor(per_graph_wmapes, dtype=torch.float64)
        q_values = torch.quantile(w_tensor, torch.tensor([0.25, 0.50, 0.75, 0.95], dtype=torch.float64))
        logging.info(
            "\n%s\n  [%s] Per-Graph WMAPE Distribution (%s graphs, %s)\n%s\n"
            "    Mean   WMAPE : %.6f\n"
            "    Std    WMAPE : %.6f\n"
            "    Min    WMAPE : %.6f\n"
            "    25%% percentile : %.6f\n"
            "    50%% percentile : %.6f\n"
            "    75%% percentile : %.6f\n"
            "    95%% percentile : %.6f\n"
            "    Max    WMAPE : %.6f\n%s",
            "=" * 70,
            split.upper(),
            len(per_graph_wmapes),
            flow_metric_space_tag(),
            "=" * 70,
            w_tensor.mean(),
            w_tensor.std(),
            w_tensor.min(),
            q_values[0],
            q_values[1],
            q_values[2],
            q_values[3],
            w_tensor.max(),
            "=" * 70,
        )

    def _edge_metrics(pred_list, true_list, label):
        if not pred_list:
            logging.info("    %s: No edges found in %s split.", label, split)
            return
        pred_tensor = torch.cat(pred_list)
        true_tensor = torch.cat(true_list)
        logging.info(
            "    %s: count=%7d  WMAPE=%.6f  MAE=%.4f",
            label,
            pred_tensor.shape[0],
            wmape(pred_tensor, true_tensor).item(),
            mean_absolute_error(pred_tensor.view(-1), true_tensor.view(-1)).item(),
        )

    logging.info(
        "\n%s\n  [%s] New-Edge vs Old-Edge Metrics (%s)\n%s",
        "=" * 70,
        split.upper(),
        flow_metric_space_tag(),
        "=" * 70,
    )
    _edge_metrics(all_preds_old, all_trues_old, "Old edges (retained)")
    _edge_metrics(all_preds_new, all_trues_new, "New edges (added)")
    _edge_metrics(all_preds_old + all_preds_new, all_trues_old + all_trues_new, "All edges (combined)")
    logging.info("%s", "=" * 70)

    avg_time_per_graph = total_time_ms / max(total_graphs, 1)
    logging.info(
        "\n%s\n  [%s] Inference Timing\n%s\n"
        "    Device               : %s\n"
        "    Total graphs         : %s\n"
        "    Total inference time : %.2f ms\n"
        "    Avg time per graph   : %.4f ms\n%s",
        "=" * 70,
        split.upper(),
        "=" * 70,
        device,
        total_graphs,
        total_time_ms,
        avg_time_per_graph,
        "=" * 70,
    )

    return {
        "wmape_percentiles": {
            "p25": float(q_values[0]) if q_values is not None else None,
            "p50": float(q_values[1]) if q_values is not None else None,
            "p75": float(q_values[2]) if q_values is not None else None,
            "p95": float(q_values[3]) if q_values is not None else None,
        },
        "wmape_mean": float(w_tensor.mean()) if w_tensor is not None else None,
        "avg_time_per_graph_ms": avg_time_per_graph,
        "total_graphs": total_graphs,
    }


@torch.no_grad()
def evaluate_baseline_aligned_split(loader, model, split="test"):
    model.eval()
    device = torch.device(cfg.accelerator)
    split_metrics = _SplitMetrics()

    per_graph_wmapes = []
    total_forward_seconds = 0.0
    total_graphs = 0
    loss_total = 0.0
    loss_old = 0.0
    loss_new = 0.0
    loss_sup = 0.0
    loss_con = 0.0
    lambda_new_current = 0.0
    num_batches = 0
    constraint_metrics = ConstraintViolationAccumulator()

    for batch in loader:
        batch.to(device)
        total_graphs += batch.num_graphs

        _maybe_cuda_synchronize(device)
        t0 = time.perf_counter()
        pred, true = _unpack_model_output(model(batch))
        _maybe_cuda_synchronize(device)
        total_forward_seconds += time.perf_counter() - t0

        loss, pred_score = _compute_loss(pred, batch)
        batch_loss_stats = _collect_batch_loss_stats(batch)
        loss_total += float(loss.detach().cpu().item())
        loss_old += float(batch_loss_stats.get("loss_old", 0.0))
        loss_new += float(batch_loss_stats.get("loss_new", 0.0))
        loss_sup += float(batch_loss_stats.get("loss_sup", loss.detach().cpu().item()))
        loss_con += float(batch_loss_stats.get("loss_con", 0.0))
        lambda_new_current += float(batch_loss_stats.get("lambda_new_current", 0.0))
        num_batches += 1

        pred_cpu = pred_score.detach().cpu().float()
        true_cpu = true.detach().cpu().float()
        pred_real, true_real, _ = get_flow_metric_tensors(pred_cpu, true_cpu)

        if hasattr(batch, "new_edge_mask"):
            is_new_edge = batch.new_edge_mask.bool().detach().cpu()
        else:
            is_new_edge = _compute_new_edge_mask(batch).detach().cpu()

        split_metrics.normalized.update(pred_cpu, true_cpu, is_new_edge)
        split_metrics.real.update(pred_real, true_real, is_new_edge)
        constraint_metrics.update(
            pred_real=pred_real.view(-1),
            edge_index_new=batch.edge_index_new.detach().cpu(),
            net_demand=batch.net_demand.detach().cpu(),
            node_batch=batch.batch.detach().cpu(),
            ptr=batch.ptr.detach().cpu(),
        )

        edge_batch = batch.batch[batch.edge_index_new[0]].detach().cpu()
        for graph_idx in range(batch.num_graphs):
            mask_graph = edge_batch == graph_idx
            pred_graph = pred_real[mask_graph]
            true_graph = true_real[mask_graph]
            if true_graph.numel() > 0:
                per_graph_wmapes.append(wmape(pred_graph, true_graph).item())

    denom = max(num_batches, 1)
    total_time_ms = total_forward_seconds * 1000.0
    avg_time_per_graph = total_time_ms / max(total_graphs, 1)
    metrics_dict = split_metrics.as_dict()

    w_tensor = None
    q_values = None
    if per_graph_wmapes:
        w_tensor = torch.tensor(per_graph_wmapes, dtype=torch.float64)
        q_values = torch.quantile(w_tensor, torch.tensor([0.25, 0.50, 0.75, 0.95], dtype=torch.float64))

    logging.info(
        "\n%s\n  [%s] Baseline-Aligned Metrics Summary\n%s\n"
        "  Normalized all edges : MAE=%.6f  RMSE=%.6f  R2=%.6f  WMAPE=%.6f\n"
        "  Real all edges       : MAE=%.6f  RMSE=%.6f  R2=%.6f  WMAPE=%.6f\n"
        "  Real old edges       : MAE=%.6f  RMSE=%.6f  R2=%.6f  WMAPE=%.6f\n"
        "  Real new edges       : MAE=%.6f  RMSE=%.6f  R2=%.6f  WMAPE=%.6f\n"
        "  Constraint violation : Con-MAE=%.6f  Con-RMSE=%.6f  Con-Max=%.6f  RelCon=%.6f  RelCon p95=%.6f\n"
        "  Inference timing     : total=%.2f ms  per_graph=%.4f ms  graphs=%s\n%s",
        "=" * 70,
        split.upper(),
        "=" * 70,
        metrics_dict["normalized"]["all_edges"]["mae"],
        metrics_dict["normalized"]["all_edges"]["rmse"],
        metrics_dict["normalized"]["all_edges"]["r2"],
        metrics_dict["normalized"]["all_edges"]["wmape"],
        metrics_dict["real"]["all_edges"]["mae"],
        metrics_dict["real"]["all_edges"]["rmse"],
        metrics_dict["real"]["all_edges"]["r2"],
        metrics_dict["real"]["all_edges"]["wmape"],
        metrics_dict["real"]["old_edges"]["mae"],
        metrics_dict["real"]["old_edges"]["rmse"],
        metrics_dict["real"]["old_edges"]["r2"],
        metrics_dict["real"]["old_edges"]["wmape"],
        metrics_dict["real"]["new_edges"]["mae"],
        metrics_dict["real"]["new_edges"]["rmse"],
        metrics_dict["real"]["new_edges"]["r2"],
        metrics_dict["real"]["new_edges"]["wmape"],
        constraint_metrics.as_dict()["con_mae"],
        constraint_metrics.as_dict()["con_rmse"],
        constraint_metrics.as_dict()["con_max"],
        constraint_metrics.as_dict()["relcon"],
        constraint_metrics.as_dict()["relcon_p95"],
        total_time_ms,
        avg_time_per_graph,
        total_graphs,
        "=" * 70,
    )

    if w_tensor is not None and q_values is not None:
        logging.info(
            "\n%s\n  [%s] Per-Graph WMAPE Distribution (%s graphs, %s)\n%s\n"
            "    Mean   WMAPE : %.6f\n"
            "    Std    WMAPE : %.6f\n"
            "    Min    WMAPE : %.6f\n"
            "    25%% percentile : %.6f\n"
            "    50%% percentile : %.6f\n"
            "    75%% percentile : %.6f\n"
            "    95%% percentile : %.6f\n"
            "    Max    WMAPE : %.6f\n%s",
            "=" * 70,
            split.upper(),
            len(per_graph_wmapes),
            flow_metric_space_tag(),
            "=" * 70,
            w_tensor.mean(),
            w_tensor.std(),
            w_tensor.min(),
            q_values[0],
            q_values[1],
            q_values[2],
            q_values[3],
            w_tensor.max(),
            "=" * 70,
        )

    return {
        "loss": {
            "loss_total": loss_total / denom,
            "loss_old": loss_old / denom,
            "loss_new": loss_new / denom,
            "lambda_new_current": lambda_new_current / denom,
            "loss_sup": loss_sup / denom,
            "loss_con": loss_con / denom,
        },
        "metrics": metrics_dict,
        "wmape_percentiles_real": {
            "p25": float(q_values[0]) if q_values is not None else None,
            "p50": float(q_values[1]) if q_values is not None else None,
            "p75": float(q_values[2]) if q_values is not None else None,
            "p95": float(q_values[3]) if q_values is not None else None,
        },
        "wmape_mean_real": float(w_tensor.mean()) if w_tensor is not None else None,
        "timing": {
            "forward_seconds_total": total_forward_seconds,
            "forward_milliseconds_per_graph": avg_time_per_graph,
            "num_graphs": total_graphs,
        },
        "constraint_violation": constraint_metrics.as_dict(),
    }


def _finalize_summary(loaders, model, perf, best_epoch, best_metric_value, full_epoch_times, inference_wall_clock_seconds=None):
    if len(loaders) < 3:
        return

    logging.info("\n%s", "=" * 70)
    logging.info("  Running detailed test evaluation ...")
    logging.info("%s", "=" * 70)
    detailed_test_evaluation(loaders[2], model, split="test")

    logging.info("\n%s", "=" * 70)
    logging.info("  Running baseline-aligned test evaluation ...")
    logging.info("%s", "=" * 70)
    final_summary = evaluate_baseline_aligned_split(loaders[2], model, split="test")

    summary_payload = {
        "model": cfg.model.type,
        "dataset_name": cfg.dataset.name,
        "network_name": getattr(cfg.dataset, "network_name", None),
        "dataset_dir": cfg.dataset.dir,
        "device_used": cfg.accelerator,
        "seed": int(cfg.seed),
        "params": int(getattr(cfg, "params", 0)),
        "best_epoch": int(best_epoch),
        "selection_metric": _selection_metric_name(),
        "selection_value": float(best_metric_value),
        "training_time": {
            "total_seconds": float(np.sum(full_epoch_times)) if full_epoch_times else 0.0,
            "average_epoch_seconds": float(np.mean(full_epoch_times)) if full_epoch_times else 0.0,
            "num_epochs": int(cfg.optim.max_epoch if full_epoch_times else 0),
        },
        "test_time": final_summary["timing"],
        "test_metrics": final_summary["metrics"],
        "test_loss": final_summary["loss"],
        "constraint_violation": final_summary["constraint_violation"],
        "wmape_percentiles_real": final_summary["wmape_percentiles_real"],
        "wmape_mean_real": final_summary["wmape_mean_real"],
        "dataset_metadata": _load_dataset_metadata(),
        "config": cfg_to_dict(cfg),
    }
    if perf is not None:
        summary_payload["validation_at_best_epoch"] = perf[1][best_epoch]
        summary_payload["test_logger_at_best_epoch"] = perf[2][best_epoch]
    if inference_wall_clock_seconds is not None:
        summary_payload["inference_wall_clock_seconds"] = float(inference_wall_clock_seconds)

    summary_path = osp.join(cfg.run_dir, "summary.json")
    _save_json(summary_path, summary_payload)
    logging.info("Saved aligned summary to %s", summary_path)


@register_train("custom")
def custom_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info("Checkpoint found, task already done")
    else:
        logging.info("Start from epoch %s", start_epoch)

    run = None
    if cfg.wandb.use:
        try:
            import wandb
        except Exception as exc:
            raise ImportError("WandB is not installed.") from exc
        wandb_name = cfg.wandb.name if cfg.wandb.name else make_wandb_name(cfg)
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ["val", "test"]
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        cfg.train.current_epoch = cur_epoch
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for index in range(1, num_splits):
                cfg.train.current_epoch = cur_epoch
                eval_epoch(loggers[index], loaders[index], model, split=split_names[index - 1])
                perf[index].append(loggers[index].write_epoch(cur_epoch))
        else:
            for index in range(1, num_splits):
                perf[index].append(perf[index][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == "reduce_on_plateau":
            scheduler.step(val_perf[-1]["loss"])
        else:
            scheduler.step()

        full_epoch_times.append(time.perf_counter() - start_time)

        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if run is not None:
            run.log(flatten_dict(perf), step=cur_epoch)

        if is_eval_epoch(cur_epoch):
            best_epoch, _ = _select_best_epoch(val_perf)
            best_metric = cfg.metric_best if cfg.metric_best != "auto" else "loss"
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()

            logging.info(
                "> Epoch %s: took %.1fs (avg %.1fs) | Best so far: epoch %s\t"
                "train_loss: %.4f\tval_loss: %.4f\ttest_loss: %.4f\tval_%s: %.4f",
                cur_epoch,
                full_epoch_times[-1],
                np.mean(full_epoch_times),
                best_epoch,
                perf[0][best_epoch]["loss"],
                perf[1][best_epoch]["loss"],
                perf[2][best_epoch]["loss"],
                best_metric,
                perf[1][best_epoch][best_metric],
            )

            if run is not None:
                run.log(
                    {
                        "best/epoch": best_epoch,
                        "best/train_loss": perf[0][best_epoch]["loss"],
                        "best/val_loss": perf[1][best_epoch]["loss"],
                        "best/test_loss": perf[2][best_epoch]["loss"],
                    },
                    step=cur_epoch,
                )
                run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

    logging.info("Avg time per epoch: %.2fs", np.mean(full_epoch_times))
    logging.info("Total train loop time: %.2fh", np.sum(full_epoch_times) / 3600)

    best_epoch_final, best_metric_value = _select_best_epoch(perf[1])
    if cfg.train.enable_ckpt and cfg.train.ckpt_best:
        try:
            load_ckpt(model, optimizer, scheduler, best_epoch_final)
            logging.info("Loaded best checkpoint from epoch %s for final evaluation.", best_epoch_final)
        except Exception as exc:
            logging.warning(
                "Failed to reload best checkpoint at epoch %s; using in-memory model. Error: %s",
                best_epoch_final,
                exc,
            )

    _save_training_history(perf, full_epoch_times)
    _finalize_summary(loaders, model, perf, best_epoch_final, best_metric_value, full_epoch_times)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if run is not None:
        run.finish()

    logging.info("Task done, results saved in %s", cfg.run_dir)


@register_train("inference-only")
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    del optimizer, scheduler
    num_splits = len(loggers)
    split_names = ["train", "val", "test"]
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()
    cfg.train.current_epoch = getattr(cfg.optim, "max_epoch", 0)

    for index in range(num_splits):
        eval_epoch(loggers[index], loaders[index], model, split=split_names[index])
        perf[index].append(loggers[index].write_epoch(cur_epoch))

    best_epoch = 0
    best_metric = cfg.metric_best if cfg.metric_best != "auto" else "loss"
    logging.info(
        "> Inference | train_loss: %.4f\tval_loss: %.4f\ttest_loss: %.4f\tval_%s: %.4f",
        perf[0][best_epoch]["loss"],
        perf[1][best_epoch]["loss"],
        perf[2][best_epoch]["loss"],
        best_metric,
        perf[1][best_epoch][best_metric],
    )

    total_inference_seconds = time.perf_counter() - start_time
    logging.info("Done! took: %.2fs", total_inference_seconds)

    _finalize_summary(
        loaders,
        model,
        perf,
        best_epoch,
        perf[1][best_epoch][best_metric],
        [],
        inference_wall_clock_seconds=total_inference_seconds,
    )

    for logger in loggers:
        logger.close()
