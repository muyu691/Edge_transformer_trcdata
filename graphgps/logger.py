import logging
import time

import numpy as np
import torch
from scipy.stats import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.graphgym import get_current_gpu_usage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import Logger
from torch_geometric.graphgym.utils.io import dict_to_json, dict_to_tb

from graphgps.metric_wrapper import get_flow_metric_tensors, wmape


class CustomLogger(Logger):
    def basic(self):
        stats_dict = {
            "loss": round(self._loss / self._size_current, max(8, cfg.round)),
            "lr": round(self._lr, max(8, cfg.round)),
            "params": self._params,
            "time_iter": round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats_dict["gpu_memory"] = gpu_memory
        return stats_dict

    def custom(self):
        stats_dict = super().custom()
        ordered_keys = [
            "loss_sup",
            "loss_data",
            "loss_con",
            "lambda_con",
            "rho_terminal_abs_mean",
            "rho_terminal_abs_max",
            "h_e_std",
            "h_v_std",
            "f_scaled_std",
            "delta_f_scaled_std",
            "delta_f_scaled_abs_mean",
            "rho_v_std",
            "rho_v_abs_mean",
        ]
        ordered = {}
        for key in ordered_keys:
            if key in stats_dict:
                ordered[key] = stats_dict.pop(key)
        ordered.update(stats_dict)
        return ordered

    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)

        if pred.ndim == 1 and true.ndim == 2:
            pred = pred.view(-1, 1)
        elif true.ndim == 1 and pred.ndim == 2:
            true = true.view(-1, 1)

        pred_real, true_real, _ = get_flow_metric_tensors(pred, true)

        def _stats(pred_tensor, true_tensor):
            spearman_val = eval_spearmanr(true_tensor.numpy(), pred_tensor.numpy())["spearmanr"]
            if np.isnan(spearman_val):
                spearman_val = 0.0
            return {
                "mae": round(float(mean_absolute_error(true_tensor, pred_tensor)), cfg.round),
                "r2": round(float(r2_score(true_tensor, pred_tensor, multioutput="uniform_average")), cfg.round),
                "mse": round(float(mean_squared_error(true_tensor, pred_tensor)), cfg.round),
                "rmse": round(float(mean_squared_error(true_tensor, pred_tensor, squared=False)), cfg.round),
                "wmape": round(float(wmape(pred_tensor, true_tensor)), cfg.round),
                "spearmanr": round(float(spearman_val), cfg.round),
            }

        normalized_stats = _stats(pred, true)
        real_stats = _stats(pred_real, true_real)

        result = dict(real_stats)
        for key, value in normalized_stats.items():
            result[f"{key}_norm"] = value
        for key, value in real_stats.items():
            result[f"{key}_real"] = value
        return result

    def update_stats(self, true, pred, loss, lr, time_used, params, dataset_name=None, **kwargs):
        del dataset_name
        assert true.shape[0] == pred.shape[0]
        batch_size = true.shape[0]

        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used

        for key, value in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = value * batch_size
            else:
                self._custom_stats[key] += value * batch_size

    def write_epoch(self, cur_epoch):
        start_time = time.perf_counter()
        basic_stats = self.basic()
        task_stats = self.regression()
        epoch_stats = {"epoch": cur_epoch, "time_epoch": round(self._time_used, cfg.round)}
        eta_stats = {
            "eta": round(self.eta(cur_epoch), cfg.round),
            "eta_hours": round(self.eta(cur_epoch) / 3600, cfg.round),
        }
        custom_stats = self.custom()

        if self.name == "train":
            stats_dict = {**epoch_stats, **eta_stats, **basic_stats, **task_stats, **custom_stats}
        else:
            stats_dict = {**epoch_stats, **basic_stats, **task_stats, **custom_stats}

        logging.info("%s: %s", self.name, stats_dict)
        dict_to_json(stats_dict, f"{self.out_dir}/stats.json")
        if cfg.tensorboard_each_run:
            dict_to_tb(stats_dict, self.tb_writer, cur_epoch)
        self.reset()
        if cur_epoch < 3:
            logging.info("...computing epoch stats took: %.2fs", time.perf_counter() - start_time)
        return stats_dict


def create_logger():
    return [
        CustomLogger(name=name, task_type="regression")
        for name in ["train", "val", "test"][: cfg.share.num_splits]
    ]


def eval_spearmanr(y_true, y_pred):
    """Compute Spearman Rho averaged across tasks."""
    results = []
    if y_true.ndim == 1:
        results.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for index in range(y_true.shape[1]):
            is_labeled = ~np.isnan(y_true[:, index])
            results.append(stats.spearmanr(y_true[is_labeled, index], y_pred[is_labeled, index])[0])
    return {"spearmanr": sum(results) / len(results)}
