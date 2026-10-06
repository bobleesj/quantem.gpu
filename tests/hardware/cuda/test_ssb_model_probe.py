"""Use fitted SSB optics without rebuilding probe calibration in a notebook."""

import numpy as np
import pytest
import torch

from quantem.gpu.ssb.results import SSBResult

cp = pytest.importorskip("cupy")


def test_probe_matches_quantem_optics_and_centered_fourier_pair():
    from quantem.diffractive_imaging.complex_probe import evaluate_probe, polar_spatial_frequencies
    from quantem.gpu.optics.physics import wavelength_A_from_kV

    result = SSBResult(
        object_wave=cp.ones((128, 128), dtype=cp.complex64), backend="cuda",
        voltage_kV=300, semiangle_mrad=30, aberrations={"C10": -3.2, "C12": 1.4, "phi12": -0.6},
    )
    actual = result.probe(space="fourier")
    # Use the SSB wavelength in both models; core's rounded constants differ.
    wavelength = wavelength_A_from_kV(300)
    frequency, phi = polar_spatial_frequencies(
        actual.shape, result.probe_sampling_A,
        device=f"cuda:{actual.device.id}",
    )
    expected = evaluate_probe(
        frequency * wavelength, phi, 30, result.probe_sampling_mrad, wavelength,
        aberration_coefs={"C10": -32, "C12": 14, "phi12": -0.6},
    )
    expected = expected / expected.abs().square().sum().sqrt()
    expected = cp.from_dlpack(torch.fft.fftshift(expected))
    cp.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
    real = result.probe()
    transformed = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(real), norm="ortho"))
    cp.testing.assert_allclose(transformed, actual, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(float(cp.sum(abs(real) ** 2)), 1, atol=1e-6)


def test_reconstruction_keeps_model_probe_and_saves_calibration(tmp_path):
    from quantem.gpu.ssb.persistence import load_result, result_paths, save_result
    from .test_ssb_thick_sample import _synthetic_session

    ssb, _ = _synthetic_session()
    try:
        native = ssb.reconstruct(aberrations={"C10": -3, "C12": 0.5, "phi12": 0.4})
        fine = ssb.reconstruct(native, upsample=4)
        automatic = ssb.reconstruct(native, upsample="auto")
        explicit = ssb.reconstruct(native, upsample=2)
        assert automatic.upsample == 2
        cp.testing.assert_array_equal(automatic.object_wave, explicit.object_wave)
        assert automatic.phase_limits == native.phase_limits
        cp.testing.assert_array_equal(fine.probe(), native.probe())
        assert fine.probe_sampling_A == native.probe_sampling_A
        changed = ssb.reconstruct(native, aberrations={"C10": 5})
        assert not cp.allclose(changed.probe(), native.probe())
        paths = result_paths(tmp_path / "saved", "reconstruct")
        save_result(changed, paths=paths, signature={"test": "probe"}, input_metadata={})
        restored = load_result(paths=paths, signature={"test": "probe"}, backend="cuda")
        cp.testing.assert_array_equal(restored.probe(), changed.probe())
        ssb.close()
        cp.testing.assert_array_equal(restored.probe(space="fourier"), changed.probe(space="fourier"))
    finally:
        ssb.close()


def test_defocused_model_fits_inside_the_automatic_field():
    result = SSBResult(
        object_wave=cp.ones((128, 128), dtype=cp.complex64), backend="cuda",
        voltage_kV=300, semiangle_mrad=30, aberrations={"C10": 100, "C12": 5, "phi12": 0.3},
    )
    intensity = abs(result.probe()) ** 2
    border = intensity.sum() - intensity[8:-8, 8:-8].sum()
    assert float(border / intensity.sum()) < 1e-3


def test_phase_and_probe_views_keep_data_and_calibration():
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    result = SSBResult(
        object_wave=cp.exp(1j * cp.arange(64 * 64, dtype=cp.float32).reshape(64, 64) / 4096),
        backend="cuda", voltage_kV=300, semiangle_mrad=30,
        scan_sampling_A=(0.4, 0.5), aberrations={"C10": 10, "C12": 2, "phi12": 0.2},
    )
    phase = result.phase.copy()
    intensity = abs(result.probe()) ** 2
    figure = result.show()
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 1
    assert image_axes[0].images[0].get_array().shape[:2] == phase.shape
    np.testing.assert_array_equal(figure.get_size_inches(), (12, 12))
    assert figure.number not in plt.get_fignums()

    figure = result.show("probe")
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 2
    assert image_axes[0].images[0].get_array().shape[:2] == intensity.shape
    assert image_axes[1].images[0].get_array().shape[:2] == result.probe(space="fourier").shape
    assert any("Å" in text.get_text() for text in image_axes[0].findobj(Text))
    assert any("mrad" in text.get_text() for text in image_axes[1].findobj(Text))
    assert figure.number not in plt.get_fignums()
    cp.testing.assert_array_equal(result.phase, phase)
    cp.testing.assert_array_equal(abs(result.probe()) ** 2, intensity)

    # The finer result covers the same physical field, with crop positions
    # selected and marked by the library in each image's own pixel coordinates.
    from dataclasses import replace

    fine = replace(result, object_wave=cp.repeat(cp.repeat(result.object_wave, 4, axis=0), 4, axis=1),
                   upsample=4, scan_sampling_A=(0.1, 0.125))
    figure = result.show(compare=fine, axsize=(7, 7))
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 4
    assert image_axes[0].patches[0].get_xy() == (23.5, 23.5)
    assert image_axes[1].patches[0].get_xy() == (95.5, 95.5)
    assert image_axes[2].images[0].get_array().shape[:2] == (16, 16)
    assert image_axes[3].images[0].get_array().shape[:2] == (64, 64)
    colorbars = [axis for axis in figure.axes if not axis.images]
    assert all(np.allclose(axis.get_ylim(), result.phase_limits) for axis in colorbars)
    np.testing.assert_array_equal(figure.get_size_inches(), (14, 14))
    cp.testing.assert_array_equal(result.phase, phase)
