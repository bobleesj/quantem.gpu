"""Eligibility for exact GPU source comparisons against saved QEM counts."""

import numpy as np


def supports_counts(datasets, stored: np.dtype) -> bool:
    """Select native unsigned single-frame bitshuffle/LZ4 source chunks only."""
    for dataset in datasets:
        dtype = dataset.dtype
        if (dtype.kind != "u" or not dtype.isnative or dtype.itemsize not in (1, 2, 4)
                or stored.kind != "u" or stored.itemsize > dtype.itemsize
                or dataset.chunks != (1, *dataset.shape[1:])):
            return False
        properties = dataset.id.get_create_plist()
        if properties.get_nfilters() != 1:
            return False
        identifier, _, values, _ = properties.get_filter(0)
        if identifier != 32008 or len(values) < 5 or values[4] != 2:
            return False
    return bool(datasets) and all(dataset.dtype == datasets[0].dtype for dataset in datasets)
