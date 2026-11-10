"""Screen calibrated floating intensities without converting them to counts."""

import os

import numpy as np
import pytest

from quantem.gpu import detector, io, screening


def test_float_screening_keeps_negative_fractional_intensities(tmp_path):
    """A float acquisition and its saved QEM produce matching cached images."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND on a physical accelerator.")
    rows, cols = np.indices((24, 28))
    disk = np.hypot(rows - 11.5, cols - 13.5) < 7
    rng = np.random.default_rng(17)
    values = (
        np.where(disk, 60, -2)[None, None]
        + rng.integers(0, 8, (16, 16, 24, 28)) / 8
    ).astype(np.float32)
    source, saved = tmp_path / "scan.npy", tmp_path / "scan.qem"
    np.save(source, values)
    with io.load(source, backend=backend, verbose=False) as data:
        io.save(saved, data)
    result = screening.prepare(saved, backend=backend)
    cached = screening.prepare(saved, backend=backend)
    assert cached.from_cache
    assert "exact_accumulation" not in result.metadata
    assert result.metadata["intensity_accumulation"]["dtype"] == "float32"
    np.testing.assert_array_equal(result.mean_pattern, values.mean((0, 1)))
    np.testing.assert_array_equal(result.total_intensity, values.sum((2, 3)))
    mask = detector.detector_mask(
        result.bright_field_center_px, result.bright_field_edge_radius_px,
        np.inf, (24, 28),
    )
    np.testing.assert_array_equal(result.dark_field, values[..., mask].sum(-1))
    assert (result.dark_field < 0).all()
    np.testing.assert_array_equal(cached.dark_field, result.dark_field)
    np.testing.assert_array_equal(cached.dpc_phase, result.dpc_phase)
    assert np.isfinite(result.dpc_phase).all()

    # Replacing a QEM at the same path must invalidate its saved products.
    replacement = tmp_path / "replacement.qem"
    values += np.float32(1)
    np.save(source, values)
    with io.load(source, backend=backend, verbose=False) as data:
        io.save(replacement, data)
    replacement.replace(saved)
    changed = screening.prepare(saved, backend=backend)
    assert not changed.from_cache
    np.testing.assert_array_equal(changed.total_intensity, values.sum((2, 3)))

    # Invalid source frames must stay visible as an error, never zero-filled.
    invalid = tmp_path / "invalid.npy"
    values[0, 0] = np.nan
    np.save(invalid, values)
    with pytest.raises(ValueError, match="non-finite diffraction intensities"):
        screening.prepare(invalid, backend=backend)
