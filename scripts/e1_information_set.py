"""E1: one network x information set x seed per process; streaming final test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import graphgps  # Register the existing GraphGym configuration.
from torch_geometric.graphgym.config import cfg, set_cfg
from graphgps.loader.dataset.network_pairs_topology import NetworkPairsTopologyDataset
from graphgps.network.topology_model import NetworkPairsTopologyModel, od_net_demand
from graphgps.loss.flow_conservation_loss import compute_pinn_loss
from graphgps.optimizer.extra_optimizers import get_wsd_schedule_with_warmup
from graphgps.train.custom_train import _RegressionAccumulator
from graphgps.utils import match_edge_indices
from create_sioux_data.sue_solver import compute_sue_fixed_point_gap, bpr_travel_time, SOLVER_VERSION

NETWORKS = {"siouxfalls": ("SiouxFalls", 32), "ema": ("EMA", 16), "anaheim": ("Anaheim", 8)}
MODES = ("od_only", "old_state", "hybrid", "persistence")
BASE_CONFIG = ROOT / "configs/GatedGCN/network-pairs-topology.yaml"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(tmp, path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_input(batch, mode):
    """A strict field allowlist, excluding labels and evaluator-only physics/OD."""
    names = ["edge_index_new", "edge_attr_new", "batch", "ptr"]
    if mode in ("old_state", "hybrid"):
        names += ["edge_index_old", "edge_attr_old", "flow_old"]
    if mode == "old_state":
        names += ["net_demand"]
    if mode in ("od_only", "hybrid"):
        names += ["od_matrix", "centroid_pos"]
    return Data(num_nodes=int(batch.num_nodes), **{key: batch[key] for key in names})


def persistence_input(batch):
    return Data(num_nodes=int(batch.num_nodes), edge_index_old=batch.edge_index_old,
                edge_index_new=batch.edge_index_new, flow_old=batch.flow_old)


def persistence(batch, mean, std):
    match = match_edge_indices(batch.edge_index_old, batch.edge_index_new, int(batch.num_nodes))
    prediction = batch.flow_old.new_zeros((match.numel(), 1))
    retained = match >= 0
    prediction[retained] = batch.flow_old[match[retained]] * std + mean
    return prediction


class PhysicalAccumulator:
    def __init__(self):
        self.regression = _RegressionAccumulator()
        self.graphs = self.negatives = self.invalid_negatives = 0
        self.relcon = self.suegap = self.tstt = self.floor = 0.0

    def update(self, pred, true, relcon, gap, tstt, floor):
        self.regression.update(pred, true)
        self.graphs += 1
        self.negatives += int((pred < 0).sum())
        self.invalid_negatives += int((pred < -1e-6).sum())
        self.relcon += relcon
        self.suegap += gap
        self.tstt += tstt
        self.floor += floor

    def result(self):
        if not self.graphs:
            return {"graphs": 0}
        result = self.regression.as_dict()
        return {"graphs": self.graphs, "edges": result["count"],
                "WMAPE": result["wmape"], "WMAPE_pct": 100 * result["wmape"],
                "RMSE": result["rmse"], "R2": result["r2"],
                "RelCon": self.relcon / self.graphs, "SUEGap": self.suegap / self.graphs,
                "TSTT_Error_pct": 100 * self.tstt / self.graphs,
                "negative_flow_fraction": self.negatives / result["count"],
                "negative_flow_count_below_minus_1e_6": self.invalid_negatives,
                "ground_truth_sue_gap_mean": self.floor / self.graphs}


def graph_physics(data, pred, true, attr_scaler, parameters):
    """One ephemeral CPU graph; no outer solve, no cached predictions/graphs."""
    pred = pred.double().view(-1)
    true = true.double().view(-1)
    if not torch.isfinite(pred).all() or not torch.isfinite(true).all():
        raise ValueError("Non-finite prediction or label; refusing misleading metrics.")
    physical = pred.clamp_min(0).numpy()
    q = data.od_matrix[0].double().numpy()
    positions = data.centroid_pos[0].numpy()
    demand = np.zeros(data.num_nodes)
    demand[positions] = q.sum(axis=0) - q.sum(axis=1)
    src, dst = data.edge_index_new.numpy()
    residual = np.bincount(dst, weights=physical, minlength=data.num_nodes) - np.bincount(
        src, weights=physical, minlength=data.num_nodes) - demand
    relcon = float(np.abs(residual).sum() / (np.abs(demand).sum() + 1e-12))
    capacity = attr_scaler.inverse_transform(data.edge_attr_new.double().numpy())[:, 0]
    t0 = data.free_flow_time_new.view(-1).double().numpy()
    nodes = data.node_ids.tolist()
    edges = [(nodes[int(u)], nodes[int(v)]) for u, v in zip(src, dst)]
    graph = nx.DiGraph(first_thru_node=int(data.first_thru_node))
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    # NetworkX groups edges by tail: never assume reconstructed iteration order.
    edge_position = {edge: i for i, edge in enumerate(edges)}
    order = np.array([edge_position[edge] for edge in graph.edges()], dtype=np.int64)
    common = dict(node_ids=nodes, centroid_nodes=[nodes[int(i)] for i in positions], model_parameters=parameters)
    floor = compute_sue_fixed_point_gap(graph, q, capacity[order], t0[order], true.numpy()[order], **common)
    # Permit float32 export round-off, but fail closed on a mismatched choice set.
    floor_limit = max(5 * float(parameters["convergence_threshold"]), 1e-7)
    if not np.isfinite(floor) or floor > floor_limit:
        raise ValueError(f"Ground-truth SUEGap {floor:.3g} > {floor_limit:.3g}; stop E1 and inspect label/evaluator consistency.")
    gap = compute_sue_fixed_point_gap(graph, q, capacity[order], t0[order], physical[order], **common)
    kwargs = dict(alpha=parameters["bpr_alpha"], beta=parameters["bpr_beta"])
    truth = true.numpy()
    tstt_true = float(np.dot(truth, bpr_travel_time(truth, capacity, t0, **kwargs)))
    tstt_pred = float(np.dot(physical, bpr_travel_time(physical, capacity, t0, **kwargs)))
    tstt = abs(tstt_pred - tstt_true) / (tstt_true + 1e-12)
    return relcon, gap, tstt, floor


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_test(dataset, model, mode, batch_size, device, mean, std, attrs, parameters):
    """One final metrics pass. Warmup forwards are unscored and excluded from timing."""
    if model is not None:
        model.eval()
    total = PhysicalAccumulator()
    by_mutation = {name: PhysicalAccumulator() for name in ("closure", "capacity_change", "new_link")}
    forward_seconds = 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    warmed = False
    for cpu_batch in loader:
        inputs = (persistence_input(cpu_batch) if mode == "persistence" else model_input(cpu_batch, mode)).to(device)
        if not warmed:
            for _ in range(5):
                if model is None:
                    persistence(inputs, mean, std)
                else:
                    model(inputs)
            synchronize(device)
            warmed = True
        synchronize(device)
        start = time.perf_counter()
        if model is None:
            prediction = persistence(inputs, mean, std)
        else:
            prediction, _ = model(inputs)
        synchronize(device)
        forward_seconds += time.perf_counter() - start
        prediction = prediction.detach().cpu().double()
        if model is not None:
            prediction = prediction * std + mean
        true = cpu_batch.y.double() * std + mean
        label_roundoff = 8 * np.finfo(np.float32).eps * (abs(mean) + abs(std) + 1)
        if torch.any(true < -label_roundoff):
            raise ValueError("Exported ground-truth flow is significantly negative.")
        true = true.clamp_min(0)  # Only float32 normalized-label round-off, not prediction clipping.
        edge_batch = cpu_batch.batch[cpu_batch.edge_index_new[0]]
        for i in range(cpu_batch.num_graphs):
            data = cpu_batch.get_example(i)
            keep = edge_batch == i
            pred_graph, true_graph = prediction[keep], true[keep]
            physical = graph_physics(data, pred_graph, true_graph, attrs, parameters)
            total.update(pred_graph, true_graph, *physical)
            by_mutation[data.mutation_type].update(pred_graph, true_graph, *physical)
        del inputs, prediction, true, cpu_batch
    if not total.graphs:
        raise ValueError("Empty test split.")
    return {**total.result(), "runtime_ms_per_graph": 1000 * forward_seconds / total.graphs,
            "mutation_type_breakdown": {key: value.result() for key, value in by_mutation.items()},
            "physics_projection": "max(raw_prediction,0); regression uses unchanged raw prediction",
            "ground_truth_sue_gap_limit": max(5 * float(parameters["convergence_threshold"]), 1e-7),
            "runtime_warmup_batches": 5, "runtime_batch_size": batch_size}


def load_split(root, split, indices, with_od):
    dataset = NetworkPairsTopologyDataset(str(root), split=split, load_od=with_od)
    actual = torch.as_tensor(dataset.data.sample_idx).view(-1).cpu().numpy()
    if not np.array_equal(actual, indices):
        raise ValueError(f"{split} PyG order does not match split_indices.npz.")
    if with_od and not dataset._od_segments:
        raise ValueError("Missing OD mmap sidecar; rebuild processed data for E1.")
    return dataset


def configure(args, metadata, mean, std):
    cfg.clear()
    set_cfg(cfg)
    cfg.merge_from_file(str(BASE_CONFIG))
    cfg.dataset.dir = str(args.dataset_dir)
    cfg.dataset.network_name = metadata["network_name"]
    for name in ("num_nodes", "num_edges_old", "num_edges_new", "od_dim"):
        setattr(cfg.dataset, name, int(metadata[name]))
    cfg.dataset.centroid_count = int(metadata["centroid_count"])
    cfg.dataset.flow_mean, cfg.dataset.flow_std = mean, std
    cfg.dataset.od_scale = float(metadata["od_scale"])
    cfg.topology_gnn.information_mode = args.mode if args.mode != "persistence" else "old_state"
    cfg.train.batch_size = args.batch_size
    cfg.train.eval_test_during_training = False
    cfg.seed = args.seed
    cfg.accelerator = args.device
    if args.smoke:
        cfg.optim.max_epoch = 2
        cfg.optim.num_warmup_epochs = 1
        cfg.optim.wsd_stable_epochs = 0
        cfg.optim.wsd_decay_epochs = 1
    # Base config is the sole architecture/training source, no mode-specific tuning.
    assert cfg.model.lambda_old == cfg.model.lambda_new_start == cfg.model.lambda_new_final == 1
    assert cfg.topology_gnn.local_backbone == "edge_transformer"
    assert cfg.model.loss_fun == "l1" and cfg.optim.scheduler == "wsd"
    required = {"hidden_dim": 128, "num_diffusion_steps": 4, "num_heads": 4,
                "num_edge_transformer_layers": 1, "ffn_type": "swiglu", "ffn_mult": "8/3",
                "norm_type": "rmsnorm", "norm_position": "pre", "edge_endpoint_mode": "fusion",
                "enable_global_attn": False, "dropout": .1, "share_diffusion_cell": True,
                "pressure_update_mode": "lwr", "alignment_mode": "full"}
    if any(getattr(cfg.topology_gnn, key) != value for key, value in required.items()):
        raise ValueError("Base configuration no longer matches the declared E1 backbone.")
    if cfg.optim.optimizer.lower() != "adamw" or cfg.optim.base_lr != .001 or cfg.optim.weight_decay != 1e-5:
        raise ValueError("Base configuration no longer matches the declared E1 optimizer.")
    if not args.smoke and cfg.optim.max_epoch != 200:
        raise ValueError("Official E1 requires 200 epochs.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_best(args, metadata, indices, output, device):
    with_od = args.mode != "old_state"
    train = load_split(args.dataset_dir, "train", indices["train_idx"], with_od)
    val = load_split(args.dataset_dir, "val", indices["val_idx"], with_od)
    if not len(train) or not len(val):
        raise ValueError("Training and validation splits must be nonempty.")
    seed_everything(args.seed)
    model = NetworkPairsTopologyModel(1, 1).to(device)
    seed_everything(args.seed)  # Encoder construction must not shift shuffle/dropout RNG.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
    scheduler = get_wsd_schedule_with_warmup(
        optimizer, cfg.optim.num_warmup_epochs, cfg.optim.max_epoch, cfg.optim.min_lr,
        stable_steps=cfg.optim.wsd_stable_epochs, decay_steps=cfg.optim.wsd_decay_epochs,
        decay_type=cfg.optim.wsd_decay_type)
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(cfg.optim.max_epoch):
        cfg.train.current_epoch = epoch
        model.train()
        train_loss, batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            inputs = model_input(batch, args.mode)
            pred, _ = model(inputs)
            batch.active_net_demand = inputs.active_net_demand
            loss, _ = compute_pinn_loss(pred, batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm_value)
            optimizer.step()
            train_loss += float(loss.detach())
            batches += 1
        validation = _RegressionAccumulator()
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                prediction, _ = model(model_input(batch, args.mode))
                validation.update(prediction, batch.y)
        score = validation.as_dict()["rmse"]
        if not np.isfinite(score):
            raise ValueError("Non-finite validation RMSE.")
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_rmse_norm": score}, output / "best.pt.tmp")
            os.replace(output / "best.pt.tmp", output / "best.pt")
        scheduler.step()
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": train_loss / batches,
                                     "val_rmse_norm": score, "best_epoch": best_epoch}) + "\n")
        print(f"epoch {epoch + 1}/{cfg.optim.max_epoch}: train_loss={train_loss/batches:.6f}, val_rmse_norm={score:.6f}, best={best_epoch}", flush=True)
    train_seconds = time.perf_counter() - started
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    train.close_od()
    val.close_od()
    return model, int(checkpoint["epoch"]), float(checkpoint["val_rmse_norm"]), train_seconds


def run(args):
    args.dataset_dir = Path(args.dataset_dir).resolve()
    network = args.network
    with (args.dataset_dir / "dataset_meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta["network_name"].lower() != network:
        raise ValueError("Network does not match dataset metadata.")
    if meta.get("od_sidecar_version") != 1 or not np.isfinite(meta.get("od_scale", np.nan)) or meta["od_scale"] <= 0:
        raise ValueError("Rebuild the processed dataset with E1 OD sidecars first.")
    parameters = meta["conversion"]["sue_model"]
    if parameters["solver_version"] != SOLVER_VERSION or parameters["reasonable_link_basis"] != "free_flow_time":
        raise ValueError("Dataset does not use the current fixed-free-flow SUE definition.")
    with np.load(args.dataset_dir / "split_indices.npz") as saved:
        indices = {key: saved[key].copy() for key in ("train_idx", "val_idx", "test_idx")}
    sizes = [len(indices[f"{name}_idx"]) for name in ("train", "val", "test")]
    if not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(sum(sizes))):
        raise ValueError("Split IDs overlap or omit samples.")
    if not args.smoke and sizes != [int(sum(sizes) * .6), int(sum(sizes) * .2), sum(sizes) - int(sum(sizes) * .6) - int(sum(sizes) * .2)]:
        raise ValueError("Official E1 requires the existing 60/20/20 split.")
    with (args.dataset_dir / "scalers/flow_scaler.pkl").open("rb") as handle:
        flow = pickle.load(handle)
    with (args.dataset_dir / "scalers/attr_scaler.pkl").open("rb") as handle:
        attrs = pickle.load(handle)
    mean, std = float(flow.mean_[0]), float(flow.scale_[0])
    configure(args, meta, mean, std)
    hashes = {name: file_digest(args.dataset_dir / name) for name in (
        "dataset_meta.json", "sample_sources.json", "split_indices.npz", "scalers/flow_scaler.pkl", "scalers/attr_scaler.pkl",
        "train_dataset.pt", "val_dataset.pt", "test_dataset.pt", "train_od.npy", "val_od.npy", "test_od.npy")}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    snapshot = cfg.dump()  # YAML string is an exact config snapshot.
    protocol_config = cfg.clone()
    protocol_config.topology_gnn.information_mode = "old_state"
    protocol_config.seed = 0
    protocol = {"dataset_fingerprint": fingerprint, "batch_size": args.batch_size,
                "device": args.device, "smoke": args.smoke, "base_config": protocol_config.dump(),
                "hybrid_od_fraction": 1.0}
    protocol_fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    network_output = Path(args.output_root) / network
    network_output.mkdir(parents=True, exist_ok=True)
    protocol_file = network_output / "protocol.json"
    if protocol_file.exists():
        with protocol_file.open(encoding="utf-8") as handle:
            if json.load(handle) != protocol:
                raise ValueError("Network protocol differs (data/batch/device/config). Use a separate output root; never mix E1 settings.")
    else:
        with protocol_file.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2)
    output = network_output / args.mode
    if args.mode != "persistence":
        output /= f"seed_{args.seed}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory already contains files: {output}; refusing to repeat test/overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", {"config": snapshot, "dataset_fingerprint": fingerprint})
    model, best_epoch, score, train_seconds = None, None, None, 0.0
    device = torch.device(args.device)
    if args.mode != "persistence":
        model, best_epoch, score, train_seconds = train_best(args, meta, indices, output, device)
    # Test dataset is first opened here, after best-validation checkpoint reload.
    test = load_split(args.dataset_dir, "test", indices["test_idx"], True)
    try:
        metrics = evaluate_test(test, model, args.mode, args.batch_size, device, mean, std, attrs, parameters)
    finally:
        test.close_od()
    summary = {"network": network, "information_mode": args.mode,
               "seed": args.seed if model is not None else None, "best_epoch": best_epoch,
               "selection_metric": "validation.rmse_norm", "selection_value": score,
               "params": sum(p.numel() for p in model.parameters()) if model is not None else 0,
               "train_time_seconds": train_seconds, **metrics,
               "dataset_split_path": str(args.dataset_dir / "split_indices.npz"),
               "dataset_fingerprint": fingerprint, "solver_equilibrium_definition": parameters,
               "protocol_fingerprint": protocol_fingerprint,
               "config_snapshot": snapshot, "hybrid_od_fraction": 1.0 if args.mode == "hybrid" else None,
               "device": str(device), "smoke": args.smoke, "formal_test_metric_passes": 1}
    write_json(output / "summary.json", summary)
    with (output / "mutation_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["mutation_type", "graphs", "WMAPE_pct", "RMSE", "RelCon", "SUEGap", "TSTT_Error_pct"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for mutation, values in metrics["mutation_type_breakdown"].items():
            writer.writerow({"mutation_type": mutation, **values})
    print(json.dumps({key: summary[key] for key in ("network", "information_mode", "WMAPE_pct", "RMSE", "R2", "RelCon", "SUEGap", "TSTT_Error_pct", "runtime_ms_per_graph")}, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, choices=NETWORKS)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Two-epoch code check; NOT an official E1 result")
    args = parser.parse_args()
    args.batch_size = args.batch_size if args.batch_size is not None else NETWORKS[args.network][1]
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.dataset_dir is None:
        args.dataset_dir = ROOT / "create_sioux_data/processed_data" / f"pyg_{args.network}_7000_e1"
    if args.output_root is None:
        args.output_root = ROOT / ("results/e1_smoke" if args.smoke else "results/e1")
    if not args.smoke and args.seed not in (42, 43, 44, 45, 46):
        parser.error("Official seeds are 42,43,44,45,46")
    return args


if __name__ == "__main__":
    run(parse_args())
