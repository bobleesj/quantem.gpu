"""Saved detector selection reproduces the fitted phase and finer output grids."""

import json

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_saved_fit_replays_exact_detector_selection(tmp_path):
    """Reopen a correction whose disk edge lies between neighbouring pixels."""
    from quantem.gpu import SSB

    rng = np.random.default_rng(42)
    data = rng.integers(1, 30, size=(128, 128, 16, 16), dtype=np.uint16)
    source = tmp_path / "detector.npy"
    np.save(source, data)
    settings = dict(
        backend="cuda", voltage_kV=300, semiangle_mrad=30,
        scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(5.0, 5.0),
        scan_detector_rotation_deg=163.8,
        source_path=str(source),
    )
    center = (7.5001, 7.5)
    radius = float(np.hypot(5 - center[0], 5 - center[1]))
    with SSB(
        data, **settings, bright_field_center_px=center,
        bright_field_radius_px=radius,
        aberrations={"C10_nm": 8.0, "C12_nm": 1.0, "phi12_deg": 21.0},
    ) as session:
        fitted = session.find_aberrations(
            num_trials=0, refinement=None, check_rotation=False,
            save_dir=tmp_path / "fit",
        )
        expected = {
            factor: session.preview(fitted.aberrations, upsampling_factor=factor, compute_loss=False)[0]
            for factor in (1, 3)
        }
        pixels = session.num_bright_field_pixels
    saved = json.loads((tmp_path / "fit" / "ssb-find_aberrations.json").read_text())["result"]
    assert saved["bright_field_radius_px"] == radius
    with SSB(
        data, **settings, bright_field_center_px=saved["bright_field_center_px"],
        bright_field_radius_px=saved["bright_field_radius_px"],
    ) as reopened:
        for factor, phase in expected.items():
            actual, _ = reopened.preview(
                saved["aberrations"], upsampling_factor=factor, compute_loss=False,
            )
            np.testing.assert_array_equal(actual, phase)
        assert reopened.num_bright_field_pixels == pixels
