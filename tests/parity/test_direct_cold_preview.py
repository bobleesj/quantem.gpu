"""Interactive direct ptychography must work before any fit or saved reconstruction."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import DirectPtycho
from quantem.gpu.io import load


def test_cold_preview_then_changed_defocus():
    """Open real scan data and adjust defocus without fitting first."""
    source = os.environ.get('QUANTEM_MEDIAN_TEST_MASTER')
    if not source or not Path(source).is_file():
        pytest.skip('Set QUANTEM_MEDIAN_TEST_MASTER to a complete Arina master.')
    pytest.importorskip('cupy')
    data = load(source, device='cuda', verbose=False)
    with data, DirectPtycho(data, backend='cuda', voltage_kV=300,
                   semiangle_mrad=30, scan_sampling_A=(0.373, 0.373),
                   scan_detector_rotation_deg=169.9) as direct:
        recipe = {'C10_nm': 12.8, 'C12_nm': 2.0, 'phi12_deg': math.degrees(0.64)}
        phase, loss = direct.preview(recipe)
        repeated, repeated_loss = direct.preview(recipe)
        np.testing.assert_array_equal(phase, repeated)
        assert loss == repeated_loss and np.isfinite(loss)
        changed, changed_loss = direct.preview(recipe | {'C10_nm': 30.0})
        assert np.isfinite(changed).all() and np.isfinite(changed_loss)
        assert np.max(np.abs(changed - phase)) > 0
        context = direct.preview_context(max(1, direct.num_bright_field_pixels // 4))
        partial, _ = direct.preview(recipe, compute_loss=False, context=context)
        assert np.isfinite(partial).all()
        restored, _ = direct.preview(recipe)
        np.testing.assert_array_equal(restored, phase)
