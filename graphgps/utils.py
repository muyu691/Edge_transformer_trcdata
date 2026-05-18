import logging

import torch
from torch import Tensor
from yacs.config import CfgNode


def flatten_dict(metrics):
    """Flatten train/val/test metric histories for WandB logging."""
    prefixes = ["train", "val", "test"]
    result = {}
    for index, prefix in enumerate(prefixes):
        if index >= len(metrics) or not metrics[index]:
            continue
        stats = metrics[index][-1]
        result.update({f"{prefix}/{key}": value for key, value in stats.items()})
    return result


def match_edge_indices(
    edge_index_old: Tensor,
    edge_index_new: Tensor,
    total_nodes: int,
) -> Tensor:
    """Match each new-graph edge to the old-graph row index, or -1 if absent."""
    e_new = edge_index_new.size(1)
    match_idx = torch.full(
        (e_new,),
        fill_value=-1,
        dtype=torch.long,
        device=edge_index_new.device,
    )
    if edge_index_old.numel() == 0 or e_new == 0:
        return match_idx

    base = int(total_nodes)
    old_keys = edge_index_old[0].long() * base + edge_index_old[1].long()
    new_keys = edge_index_new[0].long() * base + edge_index_new[1].long()

    sorted_old_keys, perm = torch.sort(old_keys)
    positions = torch.searchsorted(sorted_old_keys, new_keys)
    valid = positions < sorted_old_keys.numel()
    if not valid.any():
        return match_idx

    valid_positions = positions[valid]
    matched = sorted_old_keys[valid_positions] == new_keys[valid]
    if matched.any():
        valid_rows = valid.nonzero(as_tuple=False).view(-1)
        match_idx[valid_rows[matched]] = perm[valid_positions[matched]]

    return match_idx


def cfg_to_dict(cfg_node, key_list=None):
    """Convert a YACS config node into a plain Python dictionary."""
    if key_list is None:
        key_list = []
    valid_types = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in valid_types:
            logging.warning(
                "Key %s with value type %s is not in the supported config export types %s",
                ".".join(key_list),
                type(cfg_node),
                valid_types,
            )
        return cfg_node

    cfg_dict = dict(cfg_node)
    for key, value in cfg_dict.items():
        cfg_dict[key] = cfg_to_dict(value, key_list + [key])
    return cfg_dict


def make_wandb_name(cfg):
    dataset_name = cfg.dataset.format
    if dataset_name.startswith("PyG-"):
        dataset_name = dataset_name[4:]
    if cfg.dataset.name != "none":
        dataset_name = f"{dataset_name}-{cfg.dataset.name}" if dataset_name else cfg.dataset.name

    model_name = cfg.model.type
    if cfg.name_tag:
        model_name = f"{model_name}.{cfg.name_tag}"

    return f"{dataset_name}.{model_name}.r{cfg.run_id}"
