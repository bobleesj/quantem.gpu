"""MPS SSB after a rotation change: the same answer as a session prepared at that rotation."""

import math
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="MPS SSB requires Apple silicon"
)


def test_rotation_change_keeps_every_aperture_pixel_with_anisotropic_sampling():
    """With rows and columns sampled differently, set_rotation gives what a session prepared at the new rotation gives.

    A rotation keeps each bright-field pixel's scattering angle but turns the aperture's soft edge, one detector pixel
    wide along k, and that width changes with direction when the samplings differ. Preparation stored only the pixels
    inside the aperture at the first rotation, so after set_rotation the pixels that had entered it were missing while
    the bright-field count still included them.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    rng = np.random.default_rng(5)
    rows, cols = np.indices((64, 64)) - 31.5
    counts = rng.poisson(30.0, (128, 128, 64, 64)).astype(np.uint16)
    counts[:, :, np.hypot(rows, cols) < 30] += 300
    # the selection (30 px) reaches past the aperture (25 mrad: 25 px along rows, 17 px along columns)
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=25.0, scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(1.0, 1.5),
                    bright_field_radius_px=30.0, bright_field_center_px=(31.5, 31.5))
    aberrations = {"C10_nm": 4.0, "C12_nm": 1.0, "phi12_deg": math.degrees(0.4)}

    with SSB(counts, scan_detector_rotation_deg=0.0, **settings) as turned:
        turned.reconstruct(aberrations=aberrations)
        turned.set_rotation(45.0)
        phase, loss = turned.preview(aberrations)
    with SSB(counts, scan_detector_rotation_deg=45.0, **settings) as prepared:
        expected, expected_loss = prepared.preview(aberrations)

    np.testing.assert_array_equal(phase, expected)
    assert loss == expected_loss
