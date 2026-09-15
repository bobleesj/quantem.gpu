"""Real Arina native dense reads retain the encoded loader's pixel correction."""
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.io import load
from quantem.gpu.io._metadata import read_pixel_mask


@pytest.mark.parametrize('selection', [dict(scan_region=(0, 1, 0, 512)),
                                       dict(random_positions=64, seed=7)])
def test_native_dense_median_matches_neighbors(selection):
    """Native dense scan selections replace only stored bad detector pixels."""
    source = os.environ.get('QUANTEM_MEDIAN_TEST_MASTER')
    if not source or not Path(source).is_file():
        pytest.skip('Set QUANTEM_MEDIAN_TEST_MASTER to a complete Arina master.')
    cp = pytest.importorskip('cupy')
    options = dict(backend='cuda', representation='dense', verbose=False, **selection)
    raw = cp.asnumpy(load(source, hot_pixel_correction='none', **options).data)
    result = load(source, **options)
    corrected = cp.asnumpy(result.data)
    mask = read_pixel_mask(source)
    assert mask is not None and np.any(mask)
    for row, col in np.argwhere(mask):
        neighbors = [
            raw[..., neighbor_row, neighbor_col].astype(np.float64)
            for neighbor_row in range(max(0, row - 1), min(mask.shape[0], row + 2))
            for neighbor_col in range(max(0, col - 1), min(mask.shape[1], col + 2))
            if mask[neighbor_row, neighbor_col] == 0
        ]
        expected = np.median(np.stack(neighbors), axis=0).astype(corrected.dtype)
        np.testing.assert_array_equal(corrected[..., row, col], expected)
    np.testing.assert_array_equal(corrected[..., mask == 0], raw[..., mask == 0])
    assert result.metadata['hot_pixel_correction']['method'] == 'median'
    assert result.metadata['hot_pixel_correction']['applied']
