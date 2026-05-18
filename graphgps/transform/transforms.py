import logging

from tqdm import tqdm


class MaskEdgeFeatureTransform:
    """
    Zero-out specific columns in edge_attr_old and edge_attr_new for
    input feature importance ablation.

    edge_attr column layout (from build_network_pairs_dataset.py):
      col 0 : capacity
      col 1 : speed
      col 2 : length
    """

    def __init__(self, mask_capacity: bool = False, mask_fft: bool = False):
        self.mask_capacity = mask_capacity
        self.mask_fft = mask_fft

    def __call__(self, data):
        cols_to_mask = []
        if self.mask_capacity:
            cols_to_mask.append(0)
        if self.mask_fft:
            cols_to_mask.append(1)

        if not cols_to_mask:
            return data

        for attr_name in ("edge_attr_old", "edge_attr_new"):
            attr = getattr(data, attr_name, None)
            if attr is None:
                continue
            attr = attr.clone()
            for col in cols_to_mask:
                if col < attr.size(1):
                    attr[:, col] = 0.0
            setattr(data, attr_name, attr)
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mask_capacity={self.mask_capacity}, mask_fft={self.mask_fft})"
        )


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Apply a persistent in-memory transform to every graph in the dataset."""
    if transform_func is None:
        return dataset

    data_list = [
        transform_func(dataset.get(index))
        for index in tqdm(
            range(len(dataset)),
            disable=not show_progress,
            mininterval=10,
            miniters=max(len(dataset) // 20, 1),
        )
    ]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    logging.info("Applied in-memory transform to %s graphs.", len(data_list))
    return dataset
