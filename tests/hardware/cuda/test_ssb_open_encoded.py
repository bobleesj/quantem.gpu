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


@pytest.mark.parametrize("det_sampling", [None, SETTINGS["det_sampling"]])
def test_open_bright_field_crop_matches_full_detector(det_sampling):
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
    settings = {**SETTINGS, "det_sampling": det_sampling}
    cropped = SSB.open(str(SOURCE), **settings)
    assert cropped._data.shape[-2:] != (192, 192)          # only the bright-field region was decoded
    factors = [1, 2, 3, 4] if det_sampling is None else [1]
    crop_results = {
        factor: cropped.preview(aberrations, upsampling_factor=factor)
        for factor in factors
    }
    crop_bf = cropped.num_bf
    cropped.close()
    del cropped
    cp.get_default_memory_pool().free_all_blocks()
    full = SSB.from_array(cp.from_dlpack(load(str(SOURCE), verbose=False).read()), **settings)
    assert crop_bf == full.num_bf
    for factor in factors:
        full_phase, full_loss = full.preview(aberrations, upsampling_factor=factor)
        crop_phase, crop_loss = crop_results[factor]
        np.testing.assert_allclose(crop_phase, full_phase, atol=1e-6)
        assert abs(crop_loss - full_loss) <= 1e-6 * abs(full_loss)
        assert crop_loss == crop_results[1][1]
    full.close()


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    return float((first * second).sum() / np.sqrt((first * first).sum() * (second * second).sum()))


def test_preview_is_in_the_same_scan_order_as_the_result():
    """Every preview path (large-scan chunked, reduced drag subset, thick sample) matches the fitted result's orientation.

    The fused column-IFFT kernels of the large-scan path accumulate column-major; before 2026-09-26 its previews came back
    transposed next to ``reconstruct()`` and the thick-sample preview, so the phase appeared to rotate when a viewer switched
    between them.
    """
    if not SOURCE.is_file():
        pytest.skip("set QUANTEM_SSB_ARINA_MASTER to a real Arina master file")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    from quantem.gpu import SSB

    aberrations = {"C10": 16.2, "C12": 4.15, "phi12": 0.33}
    session = SSB.open(str(SOURCE), **SETTINGS)
    result = cp.asnumpy(session.reconstruct(aberrations).phase).astype(np.float64)
    previews = {
        "chunked with loss": session.preview(aberrations)[0],
        "chunked without loss": session.preview(aberrations, compute_loss=False)[0],
        "full drag subset": session.preview(aberrations, context=session.preview_context(session.num_bf))[0],
        "thick sample": session.preview(aberrations, tilt_mrad=(0.0, 0.0), depth_spread_nm=1e-6)[0],
    }
    for name, phase in previews.items():
        phase = np.asarray(cp.asnumpy(phase), np.float64)
        # validated 2026-09-26 on a real 512 x 512 acquisition: 0.995 in scan order, -0.03 transposed
        assert _correlation(result, phase) > 0.98, name
        assert _correlation(result, phase) > _correlation(result, phase.T) + 0.5, name
