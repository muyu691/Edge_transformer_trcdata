"""CPU E1 regressions, using tiny verified fixtures in automatically removed temp dirs."""
import contextlib
import copy
import io
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import importlib
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from test_sharded_dataset_builder import make_pair
import e1_information_set as e1
from create_sioux_data import build_network_pairs_dataset as builder
from create_sioux_data import sue_solver
from graphgps.loader.master_loader import join_dataset_splits
from summarize_e1_information_set import summarize


class E1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="e1_regression_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = [make_pair(i) for i in range(10)]
        # Nontrivial node ordering tests centroid mapping without assuming [0,...,C-1].
        for pair in self.raw:
            pair["network_name"] = "SiouxFalls"
        for i in range(2):
            directory = self.root / f"batch_{i + 1:04d}"
            directory.mkdir()
            with (directory / "network_pairs_dataset.pkl").open("wb") as handle:
                pickle.dump({"pairs": self.raw[i*5:(i+1)*5]}, handle)
        self.dataset = self.root / "dataset"
        with contextlib.redirect_stdout(io.StringIO()):
            builder.run(SimpleNamespace(input_pkl=None, input_dir=self.root,
                                       output_dir=self.dataset, train_ratio=.6, val_ratio=.2, seed=42))
        self.meta = json.loads((self.dataset / "dataset_meta.json").read_text())
        self.indices = dict(np.load(self.dataset / "split_indices.npz"))
        with (self.dataset / "scalers/flow_scaler.pkl").open("rb") as handle:
            self.flow = pickle.load(handle)

    def args(self, mode):
        return SimpleNamespace(network="siouxfalls", mode=mode, seed=42, dataset_dir=self.dataset,
                               batch_size=2, device="cpu", output_root=self.root / "results", smoke=True)

    def test_sidecars_mapping_and_split_merge(self):
        expected = np.concatenate([self.raw[int(i)]["od_matrix"].ravel() for i in self.indices["train_idx"]])
        self.assertAlmostEqual(self.meta["od_scale"], expected[expected > 0].mean())
        splits = []
        for name in ("train", "val", "test"):
            dataset = e1.load_split(self.dataset, name, self.indices[f"{name}_idx"], True)
            for i in range(len(dataset)):
                data = dataset[i]
                np.testing.assert_array_equal(data.od_matrix[0], self.raw[data.sample_idx]["od_matrix"])
                self.assertNotIn("od_matrix", dataset.get_graph(i))
            self.assertTrue(all(isinstance(m, np.memmap) and not m.flags.writeable for m in dataset._od_maps.values()))
            splits.append(dataset)
        merged = join_dataset_splits(splits)
        self.assertNotIn("od_matrix", merged.data)
        for i in range(10):
            data = merged[i]
            np.testing.assert_array_equal(data.od_matrix[0], self.raw[data.sample_idx]["od_matrix"])
        batch = Batch.from_data_list([merged[0], merged[6]])
        self.assertEqual(tuple(batch.od_matrix.shape), (2, 2, 2))
        self.assertEqual(tuple(batch.centroid_pos.shape), (2, 2))
        # Explicitly verify mapping when centroids are NOT the first C local nodes.
        example = Data(num_nodes=4, od_matrix=torch.tensor([[[0., 5.], [2., 0.]]]),
                       centroid_pos=torch.tensor([[3, 1]]))
        mapped = Batch.from_data_list([example, example])
        torch.testing.assert_close(e1.od_net_demand(mapped), torch.tensor([0., 3., 0., -3.] * 2))

    def test_information_isolation_and_common_backbone(self):
        dataset = e1.load_split(self.dataset, "train", self.indices["train_idx"], True)
        batch = Batch.from_data_list([dataset[0], dataset[1]])
        backbone = None
        for mode in ("old_state", "od_only", "hybrid"):
            e1.configure(self.args(mode), self.meta, float(self.flow.mean_[0]), float(self.flow.scale_[0]))
            e1.seed_everything(42)
            model = e1.NetworkPairsTopologyModel(1, 1).eval()
            state = {k: v for k, v in model.state_dict().items() if not k.startswith("od_")}
            if backbone is None:
                backbone = state
            else:
                for key in backbone:
                    torch.testing.assert_close(backbone[key], state[key], rtol=0, atol=0)
            original = e1.model_input(batch, mode)
            with torch.no_grad():
                before, _ = model(original)
            changed = batch.clone()
            if mode == "old_state":
                self.assertFalse(hasattr(model, "od_encoder"))
                changed.od_matrix[:] = float("nan")
                self.assertNotIn("od_matrix", original)
            if mode == "od_only":
                for key in ("edge_index_old", "edge_attr_old", "flow_old", "new_edge_mask", "net_demand"):
                    self.assertNotIn(key, original)
                    del changed[key]
                torch.testing.assert_close(original.f_init_real, torch.zeros_like(original.f_init_real), atol=1e-5, rtol=0)
                torch.testing.assert_close(original.rho_v_history[0].view(-1), -e1.od_net_demand(batch))
            with torch.no_grad():
                after, _ = model(e1.model_input(changed, mode))
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            if mode in ("od_only", "hybrid"):
                q1 = batch.od_matrix.clone()
                q2 = q1.clone()
                q2[:, 0, 0] += 1
                q2[:, 0, 1] -= 1
                q2[:, 1, 0] -= 1
                q2[:, 1, 1] += 1
                # Same row/column sums, different full OD: encoder must distinguish them.
                self.assertFalse(torch.allclose(model.od_encoder(q1), model.od_encoder(q2)))

    def test_gap_one_loading_and_physics_floor(self):
        pair = self.raw[0]
        params = pair["sue_diagnostics_new"]
        graph = pair["G_prime"]
        capacity = np.array([graph[u][v]["capacity"] for u, v in graph.edges()])
        with patch.object(sue_solver, "markov_logit_sue_solver", side_effect=AssertionError("outer solve")), \
             patch.object(sue_solver, "verify_sue_solution", side_effect=AssertionError("label verifier")), \
             patch.object(sue_solver, "_markov_logit_network_loading", wraps=sue_solver._markov_logit_network_loading) as loading:
            gap = sue_solver.compute_sue_fixed_point_gap(graph, pair["od_matrix"], capacity, np.ones(2),
                                                        pair["flows_new"] * 2, node_ids=(1, 2), centroid_nodes=(1, 2), model_parameters=params)
            self.assertAlmostEqual(gap, .5, places=10)
            self.assertEqual(loading.call_count, 1)
        bad = dict(params, solver_version="wrong")
        with self.assertRaises(ValueError):
            sue_solver.compute_sue_fixed_point_gap(graph, pair["od_matrix"], capacity, np.ones(2),
                                                   pair["flows_new"], node_ids=(1, 2), centroid_nodes=(1, 2), model_parameters=bad)

    def test_persistence_alignment(self):
        data = Data(num_nodes=3, edge_index_old=torch.tensor([[0, 1], [1, 2]]),
                    edge_index_new=torch.tensor([[1, 0, 2], [2, 1, 0]]), flow_old=torch.tensor([[1.], [2.]]))
        torch.testing.assert_close(e1.persistence(data, 10., 3.), torch.tensor([[16.], [13.], [0.]]))

    def test_four_modes_two_epoch_smoke(self):
        for mode in e1.MODES:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = e1.run(self.args(mode))
            self.assertEqual(summary["graphs"], 2)
            self.assertEqual(summary["formal_test_metric_passes"], 1)
            self.assertLess(summary["ground_truth_sue_gap_mean"], 5e-5)
            if mode != "persistence":
                output = self.root / "results/siouxfalls" / mode / "seed_42"
                history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
                self.assertEqual(len(history), 2)
                self.assertTrue(all(not any("test" in key for key in row) for row in history))
                self.assertEqual(summary["best_epoch"], min(history, key=lambda row: row["val_rmse_norm"])["epoch"])

    def test_reuse_split_and_od_alignment(self):
        out = self.root / "reexport"
        with contextlib.redirect_stdout(io.StringIO()):
            builder.run(SimpleNamespace(input_pkl=None, input_dir=self.root, output_dir=out,
                                       train_ratio=.6, val_ratio=.2, seed=999,
                                       split_indices=self.dataset / "split_indices.npz"))
        with np.load(out / "split_indices.npz") as saved:
            for key, expected in self.indices.items():
                np.testing.assert_array_equal(saved[key], expected)
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(np.load(out / f"{split}_od.npy"), np.load(self.dataset / f"{split}_od.npy"))

    def test_streaming_metric_and_aggregation(self):
        accumulator = e1.PhysicalAccumulator()
        accumulator.update(torch.tensor([-2., 4.]), torch.tensor([1., 3.]), 2., 3., .2, 1e-5)
        accumulator.update(torch.tensor([10.]), torch.tensor([8.]), 4., 5., .4, 1e-5)
        result = accumulator.result()
        self.assertAlmostEqual(result["WMAPE"], 6 / 12)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(14 / 3))
        self.assertAlmostEqual(result["RelCon"], 3.)
        self.assertAlmostEqual(result["TSTT_Error_pct"], 30.)
        self.assertAlmostEqual(result["negative_flow_fraction"], 1 / 3)
        root = self.root / "aggregation"
        for seed, value in ((42, 10.), (43, 12.)):
            path = root / "siouxfalls/old_state" / f"seed_{seed}"
            path.mkdir(parents=True)
            summary = dict(result, network="siouxfalls", information_mode="old_state", seed=seed,
                           WMAPE_pct=value, runtime_ms_per_graph=1., runtime_batch_size=32, device="cpu",
                           dataset_fingerprint="same", protocol_fingerprint="same", smoke=False,
                           formal_test_metric_passes=1, mutation_type_breakdown={})
            (path / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "Missing"):
            summarize(root)
        with contextlib.redirect_stdout(io.StringIO()):
            rows = summarize(root, allow_partial=True)
        self.assertEqual(rows[0]["WMAPE_pct_mean"], 11.)
        self.assertAlmostEqual(rows[0]["WMAPE_pct_std"], np.sqrt(2.))

    def test_custom_train_never_evaluates_epoch_test(self):
        custom = importlib.import_module("graphgps.train.custom_train")
        e1.configure(self.args("old_state"), self.meta, float(self.flow.mean_[0]), float(self.flow.scale_[0]))
        e1.cfg.train.auto_resume = False
        e1.cfg.run_dir = str(self.root)
        e1.cfg.train.ckpt_clean = False
        loggers = [SimpleNamespace(write_epoch=lambda epoch: {"loss": 1. / (epoch + 1), "rmse_norm": 1. / (epoch + 1)},
                                   close=lambda: None) for _ in range(3)]
        optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters())
        scheduler = SimpleNamespace(step=lambda: None)
        with patch.object(custom, "train_epoch"), patch.object(custom, "eval_epoch") as evaluation, \
             patch.object(custom, "save_ckpt"), patch.object(custom, "load_ckpt", return_value=2), \
             patch.object(custom, "_save_training_history"), patch.object(custom, "_finalize_summary") as final:
            custom.custom_train(loggers, [[], [], []], None, optimizer, scheduler)
            self.assertEqual([call.kwargs["split"] for call in evaluation.call_args_list], ["val", "val"])
            self.assertEqual(final.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
