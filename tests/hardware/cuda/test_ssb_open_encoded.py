"""SSB.open keeps the acquisition ANS encoded and decodes only the bright-field crop.

The crop must select exactly the bright-field pixels, centre and radius of a session built from the fully decoded cube,
so the reconstruction is the same to float32 rounding. Real Arina acquisition (512 x 512 scan, 192 x 192 detector); skips
when ``QUANTEM_SSB_ARINA_MASTER`` is unset or no CUDA device is present.
"""

import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

# an Arina master of a real 512 x 512 x 192 x 192 acquisition, 300 kV, 30 mrad, 0.264 A scan step, rotation 158.9 deg
SOURCE = Path(os.environ.get("QUANTEM_SSB_ARINA_MASTER", ""))
SETTINGS = dict(backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.264, det_sampling=0.5554,
                rotation_angle_deg=158.9)


def test_open_bright_field_crop_matches_full_detector():
    if not SOURCE.is_file():
        pytest.skip("set QUANTEM_SSB_ARINA_MASTER to a real Arina master file")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    from quantem.gpu import SSB
    from quantem.gpu.io import load

    aberrations = {"C10": 16.2, "C12": 4.15, "phi12": 0.33}
    cropped = SSB.open(str(SOURCE), **SETTINGS)
    assert cropped._data.shape[-2:] != (192, 192)          # only the bright-field region was decoded
    crop_phase, crop_loss = cropped.preview(aberrations)
    crop_phase, crop_bf = cp.asnumpy(crop_phase), cropped.num_bf
    del cropped
    cp.get_default_memory_pool().free_all_blocks()
    full = SSB.from_array(cp.from_dlpack(load(str(SOURCE), verbose=False).read()), **SETTINGS)
    full_phase, full_loss = full.preview(aberrations)
    assert crop_bf == full.num_bf
    # validated 2026-09-24: max |phase difference| 6e-8 rad, identical loss
    np.testing.assert_allclose(crop_phase, cp.asnumpy(full_phase), atol=1e-6)
    assert abs(crop_loss - full_loss) <= 1e-6 * abs(full_loss)
