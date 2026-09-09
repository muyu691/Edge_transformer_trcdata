"""Small self-contained regression tests; no production data or full SUE solves."""

import contextlib
import copy
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "create_sioux_data"))
import build_network_pairs_dataset as builder
from solve_network_pairs import _bind_pair_certificates
from sue_solver import verify_sue_solution


def make_pair(number):
    """Two nodes with a unique legal path: analytical flows equal OD demands."""
    graph = nx.DiGraph(first_thru_node=1)
    graph.add_edge(1, 2, capacity=100. + number, speed=60., length=1., free_flow_time=1.)
    graph.add_edge(2, 1, capacity=120. + number, speed=60., length=1., free_flow_time=1.)
    changed = graph.copy()
    changed[1][2]["capacity"] *= 0.8
    demand = 10. + number * 7.
    pair = dict(G=graph, G_prime=changed, edge_list_old=list(graph.edges()),
                edge_list_new=list(changed.edges()), node_ids=(1, 2), centroid_nodes=(1, 2),
                network_name="Toy", mutation_type="capacity_change", sample_idx=number % 5,
                od_matrix=np.array([[0., demand], [demand / 2., 0.]]),
                flows_old=np.array([demand, demand / 2.]),
                flows_new=np.array([demand, demand / 2.]))
    for side, current in (("old", graph), ("new", changed)):
        pair[f"sue_diagnostics_{side}"] = verify_sue_solution(
            current, pair["od_matrix"],
            np.array([current[u][v]["capacity"] for u, v in current.edges()]),
            np.ones(2), pair[f"flows_{side}"], node_ids=(1, 2), centroid_nodes=(1, 2),
            loading_protocol="reasonable_links")
    _bind_pair_certificates(pair)
    return pair


class ShardedBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sharded_builder_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = [make_pair(i) for i in range(10)]
        self.write_shard(1, self.raw[:5])
        self.write_shard(2, self.raw[5:])

    def write_shard(self, number, pairs):
        path = self.root / f"batch_{number:04d}_seed_{41 + number}" / "network_pairs_dataset.pkl"
        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"pairs": pairs}, handle)
        return path

    def test_global_split_scalers_and_roundtrip(self):
        args = SimpleNamespace(input_pkl=None, input_dir=str(self.root),
                               output_dir=str(self.root / "export"), expected_samples=10,
                               train_ratio=0.6, val_ratio=0.2, seed=42)
        with contextlib.redirect_stdout(io.StringIO()):
            builder.run(args)
        out = Path(args.output_dir)
        sources = json.loads((out / "sample_sources.json").read_text())
        self.assertEqual([row["source_sample_idx"] for row in sources], list(range(5)) * 2)
        self.assertEqual([row["sample_idx"] for row in sources], list(range(10)))
        splits = builder.split_indices(10, seed=42)
        np.testing.assert_array_equal(splits[0], builder.split_indices(10, seed=42)[0])
        with (out / "scalers" / "attr_scaler.pkl").open("rb") as handle:
            attrs = pickle.load(handle)
        with (out / "scalers" / "flow_scaler.pkl").open("rb") as handle:
            flows = pickle.load(handle)
        expected_attrs = StandardScaler().fit(np.vstack([
            builder.extract_edge_attrs(self.raw[int(i)][graph], self.raw[int(i)][edges])
            for i in splits[0] for graph, edges in (("G", "edge_list_old"), ("G_prime", "edge_list_new"))]))
        expected_flows = StandardScaler().fit(np.concatenate([
            self.raw[int(i)][f"flows_{side}"] for i in splits[0] for side in ("old", "new")]).reshape(-1, 1))
        for actual, expected in ((attrs, expected_attrs), (flows, expected_flows)):
            np.testing.assert_allclose(actual.mean_, expected.mean_, rtol=1e-12)
            np.testing.assert_allclose(actual.var_, expected.var_, rtol=1e-12)
            self.assertEqual(actual.n_samples_seen_, expected.n_samples_seen_)
        datasets = []
        with np.load(out / "split_indices.npz") as saved:
            for name, indices in zip(("train", "val", "test"), splits):
                dataset = torch.load(out / f"{name}_dataset.pt", weights_only=False)
                datasets.extend(dataset)
                np.testing.assert_array_equal(saved[f"{name}_idx"], indices)
                self.assertEqual([d.sample_idx for d in dataset], indices.tolist())
                for position, data in enumerate(dataset):
                    row = sources[data.sample_idx]
                    self.assertEqual((row["split"], row["split_position"]), (name, position))
                    self.assertEqual(data.source_shard, row["source_shard"])
                    np.testing.assert_allclose(flows.inverse_transform(data.y.numpy()).ravel(),
                                               self.raw[data.sample_idx]["flows_new"], rtol=1e-6)
        self.assertEqual(len({d.sample_idx for d in datasets}), 10)
        batch = Batch.from_data_list(datasets)
        self.assertEqual(batch.sample_idx.tolist(), [d.sample_idx for d in datasets])
        self.assertEqual(batch.source_sample_idx.tolist(), [d.source_sample_idx for d in datasets])
        # Changing held-out features/labels cannot change fitted training statistics.
        changed = copy.deepcopy(self.raw)
        for i in np.concatenate(splits[1:]):
            changed[int(i)] = make_pair(1000 + int(i))
        with contextlib.redirect_stdout(io.StringIO()):
            changed_attrs, changed_flows = builder.fit_scalers(changed, splits[0])
        np.testing.assert_allclose(changed_attrs.mean_, attrs.mean_)
        np.testing.assert_allclose(changed_flows.mean_, flows.mean_)
        with self.assertRaisesRegex(ValueError, "not empty"):
            builder.run(args)

    def test_single_pickle_compatibility(self):
        path = self.write_shard(1, self.raw[:5])
        pairs = builder.ShardedPairs(input_pkl=path)
        sources, _ = builder.scan_pair_sources(pairs)
        self.assertEqual(len(sources), 5)

    def test_incomplete_and_changed_shards(self):
        pairs = builder.ShardedPairs(input_dir=self.root)
        (self.root / "batch_0003_seed_44").mkdir()
        with self.assertRaisesRegex(ValueError, "changed"):
            list(pairs)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            builder.ShardedPairs(input_dir=self.root)

    def test_reject_mixed_network_or_model(self):
        for field, value in (("network_name", "Another"), ("theta", 0.7)):
            changed = copy.deepcopy(self.raw[5:])
            for pair in changed:
                if field == "network_name":
                    pair[field] = value
                else:
                    for side in ("old", "new"):
                        graph = pair["G" if side == "old" else "G_prime"]
                        pair[f"sue_diagnostics_{side}"] = verify_sue_solution(
                            graph, pair["od_matrix"],
                            np.array([graph[u][v]["capacity"] for u, v in graph.edges()]),
                            np.ones(2), pair[f"flows_{side}"], theta=value,
                            node_ids=(1, 2), centroid_nodes=(1, 2), loading_protocol="reasonable_links")
                    _bind_pair_certificates(pair)
            self.write_shard(2, changed)
            with self.assertRaisesRegex(ValueError, "mixes"):
                builder.scan_pair_sources(builder.ShardedPairs(input_dir=self.root))

    def test_reject_duplicate_scenarios_and_tampered_labels(self):
        self.write_shard(2, self.raw[:5])
        with self.assertRaisesRegex(ValueError, "Duplicate base"):
            builder.scan_pair_sources(builder.ShardedPairs(input_dir=self.root))
        changed = copy.deepcopy(self.raw[5:])
        changed[0]["flows_new"][0] += 1
        self.write_shard(2, changed)
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            builder.scan_pair_sources(builder.ShardedPairs(input_dir=self.root))

    def test_empty_wrong_count_and_invalid_ratios(self):
        for train, val in ((0., 0.2), (0.9, 0.2), (0.6, -0.1), (float("nan"), 0.2)):
            with self.assertRaises(ValueError):
                builder.split_indices(10, train, val)
        args = SimpleNamespace(input_pkl=None, input_dir=str(self.root),
                               output_dir=str(self.root / "export"), expected_samples=7000)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Expected 7000"):
            builder.run(args)
        self.assertFalse(Path(args.output_dir).exists())
        self.write_shard(1, [])
        self.write_shard(2, [])
        with self.assertRaisesRegex(ValueError, "No verified"):
            builder.scan_pair_sources(builder.ShardedPairs(input_dir=self.root))


if __name__ == "__main__":
    unittest.main(verbosity=2)
