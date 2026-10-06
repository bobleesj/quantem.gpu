"""Real Arina encoded reads retain the loader's stored-pixel median correction."""
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.formats.hdf5.master import read_pixel_mask
from quantem.gpu.io import load


@pytest.mark.parametrize('selection', [dict(scan_region=(0, 1, 0, 512))])
def test_native_dense_median_matches_neighbors(selection):
    """Encoded reads replace only stored bad detector pixels."""
    source = os.environ.get('QUANTEM_MEDIAN_TEST_MASTER')
    if not source or not Path(source).is_file():
        pytest.skip('Set QUANTEM_MEDIAN_TEST_MASTER to a complete Arina master.')
    pytest.importorskip('cupy')
    with load(source, backend='cuda', hot_pixel_correction='none', verbose=False) as raw_source:
        raw = raw_source.read(**selection).cpu().numpy()
    with load(source, backend='cuda', verbose=False) as result:
        corrected = result.read(**selection).cpu().numpy()
        correction = result.metadata['hot_pixel_correction']
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
    assert correction['method'] == 'median'
    assert correction['applied']
