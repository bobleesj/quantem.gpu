"""Interactive SSB must work before any fit or saved reconstruction."""
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import SSB
from quantem.gpu.io import load


def test_cold_preview_then_changed_defocus():
    """Open real scan data and adjust defocus without fitting first."""
    source = os.environ.get('QUANTEM_MEDIAN_TEST_MASTER')
    if not source or not Path(source).is_file():
        pytest.skip('Set QUANTEM_MEDIAN_TEST_MASTER to a complete Arina master.')
    pytest.importorskip('cupy')
    data = load(source, backend='cuda', representation='dense',
                scan_region=(0, 64, 0, 64), verbose=False)
    with SSB.from_array(data.data, backend='cuda', voltage_kV=300,
                        semiangle_mrad=30, scan_sampling_A=0.373,
                        rotation_angle_deg=169.9) as ssb:
        recipe = dict(C10=12.8, C12=2.0, phi12=0.64)
        phase, loss = ssb.preview(recipe)
        repeated, repeated_loss = ssb.preview(recipe)
        np.testing.assert_array_equal(phase, repeated)
        assert loss == repeated_loss and np.isfinite(loss)
        changed, changed_loss = ssb.preview(recipe | {'C10': 30.0})
        assert np.isfinite(changed).all() and np.isfinite(changed_loss)
        assert np.max(np.abs(changed - phase)) > 0
        context = ssb.preview_context(max(1, ssb.num_bf // 4))
        partial, _ = ssb.preview(recipe, compute_loss=False, context=context)
        assert np.isfinite(partial).all()
        restored, _ = ssb.preview(recipe)
        np.testing.assert_array_equal(restored, phase)
