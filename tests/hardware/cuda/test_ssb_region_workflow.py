"""Fit a scan region, then apply those optics to the full acquisition."""

import math
import numpy as np
import pytest

from quantem.gpu import SSB

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_region_fit_matches_explicit_crop_and_preserves_full_reconstruction(tmp_path):
    source, _ = _synthetic_session()
    data = np.tile(source._data, (2, 2, 1, 1))
    source.close()
    source_path = tmp_path / "counts.npy"
    np.save(source_path, data)
    options = dict(
        backend="cuda", voltage_kV=300, semiangle_mrad=30,
        scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(3, 3),
        bright_field_center_px=(15.5, 15.5), bright_field_radius_px=10,
        aberrations={"C10_nm": 3, "C12_nm": 0.5, "phi12_deg": math.degrees(0.3)},
        source_path=source_path,
    )
    search = dict(tilt=False, num_trials=0, refinement=None, check_rotation=False, verbose=False)
    with SSB(data, **options) as full, SSB(
        # the region keeps the full scan's calibration, its estimated rotation included; a crop left to estimate
        # its own (6.9 instead of 4.9 degrees here) is a different fit
        data[64:192, 64:192], **options, scan_detector_rotation_deg=full.scan_detector_rotation_deg,
        scan_detector_rotation_flipped=full.scan_detector_rotation_flipped,
    ) as cropped:
        regional = full.find_aberrations(search="standard",
            scan_region=(64, 192, 64, 192), save_dir=tmp_path / "region", **search,
        )
        explicit = cropped.find_aberrations(search="standard", **search)
        np.testing.assert_array_equal(regional.phase.get(), explicit.phase.get())
        assert regional.fit_scan_region == (64, 192, 64, 192)
        assert regional.phase.shape == (128, 128)
        reconstructed = full.reconstruct(regional)
        assert reconstructed.phase.shape == (256, 256)
        assert reconstructed.fit_scan_region == regional.fit_scan_region
        assert full.reconstruct().phase.shape == (256, 256)
        assert full._data.shape == data.shape

        from quantem.gpu.ssb.persistence import load_result, result_paths

        restored = load_result(
            paths=result_paths(tmp_path / "region", "find_aberrations"),
            signature=regional.metadata["signature"], backend="cuda",
        )
        assert tuple(restored.fit_scan_region) == regional.fit_scan_region
        np.testing.assert_array_equal(restored.phase.get(), regional.phase.get())
