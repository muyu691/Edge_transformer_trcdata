"""In-memory loader for processed traffic network-pairs datasets."""

import logging
import os.path as osp

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset


class NetworkPairsTopologyDataset(InMemoryDataset):
    """Load one split of the processed network-pairs dataset."""

    _SPLIT_FILES = {
        'train': 'train_dataset.pt',
        'val': 'val_dataset.pt',
        'test': 'test_dataset.pt',
    }

    def _download(self):
        pass

    def _process(self):
        pass

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
        load_od=True,
    ) -> None:
        assert split in self._SPLIT_FILES, (
            f"split must be one of {tuple(self._SPLIT_FILES)}, got '{split}'"
        )
        self.split = split
        self.load_od = load_od
        self._od_maps = {}
        self._od_segments = []
        super().__init__(root, transform, pre_transform, pre_filter)

        pt_path = osp.join(self.processed_dir, self._SPLIT_FILES[split])
        if not osp.exists(pt_path):
            raise FileNotFoundError(
                f"Missing processed split file: {pt_path}\n"
                f"Run the data pipeline first:\n"
                f"  1. python solve_network_pairs.py --network_name <NetworkName>\n"
                f"  2. python build_network_pairs_dataset.py --output_dir {self.root}\n"
            )

        data_list = torch.load(pt_path, weights_only=False)
        logging.info(
            f"[NetworkPairsTopologyDataset] Loaded {split} split: "
            f"{len(data_list)} graphs from {pt_path}"
        )
        self.data, self.slices = self.collate(data_list)
        od_path = osp.join(self.processed_dir, f'{split}_od.npy')
        if osp.exists(od_path):
            self._od_segments = [(0, len(data_list), od_path)]

    def get_graph(self, idx):
        """Graph only: used by split merging, never materializes OD tensors."""
        return super().get(idx)

    def get(self, idx):
        data = self.get_graph(idx)
        if self.load_od:
            for start, end, path in self._od_segments:
                if start <= idx < end:
                    if path not in self._od_maps:
                        mapping = np.load(path, mmap_mode='r')
                        if mapping.dtype != np.float32 or mapping.shape != (end - start, int(data.centroid_count), int(data.centroid_count)):
                            raise ValueError(f'OD sidecar shape/dtype mismatch: {path}')
                        self._od_maps[path] = mapping
                    # Add a leading dimension so PyG batches this as [B,C,C].
                    data.od_matrix = torch.from_numpy(np.array(self._od_maps[path][idx - start], copy=True)).unsqueeze(0)
                    break
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_od_maps'] = {}  # Workers reopen read-only mmap, never pickle its contents.
        return state

    def close_od(self):
        """Release mmap handles explicitly (especially useful on Windows)."""
        for mapping in self._od_maps.values():
            mapping._mmap.close()
        self._od_maps.clear()

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return list(self._SPLIT_FILES.values())

    def download(self):
        raise FileNotFoundError(
            "NetworkPairs datasets are not downloaded automatically.\n"
            "Generate them locally first:\n"
            "  1. python solve_network_pairs.py --network_name <NetworkName>\n"
            f"  2. python build_network_pairs_dataset.py --output_dir {self.root}\n"
        )

    def process(self):
        pass

    def __repr__(self) -> str:
        return (
            f"NetworkPairsTopologyDataset("
            f"split={self.split}, "
            f"num_graphs={len(self)})"
        )
