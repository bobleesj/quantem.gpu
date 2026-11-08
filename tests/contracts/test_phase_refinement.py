"""Workflow checks for portable frozen weights and current-only inference."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.gpu.direct_ptycho.neural_phase import NeuralPhaseModel
from quantem.gpu.direct_ptycho._phase_networks import Direction, ResidualUNet
from quantem.gpu.direct_ptycho._phase_refinement import refine


@pytest.mark.parametrize("record", [None, {"selected_step": 10, "history": [{"step": 10, "validation_gain": 0.1}]}])
def test_export_load_and_reuse_without_changing_frame_history(tmp_path, record):
    unet = ResidualUNet(2)
    head = Direction()
    with torch.no_grad():
        unet.last.weight.zero_()
        unet.last.bias.zero_()
        head.decode[-1].weight.zero_()
        head.decode[-1].bias.zero_()
    checkpoint = tmp_path / "refinement.pt"
    NeuralPhaseModel.export(
        checkpoint,
        unet=unet.state_dict(),
        readout=[0.0] * 18,
        directions=[head.state_dict()],
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        annulus_mrad=(32.0, 45.0),
        object_sampling_A=(0.2, 0.2),
        upsampling=1,
        training_record=record,
    )
    model = NeuralPhaseModel.load(checkpoint)
    assert model.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert model.training_history == tuple((record or {}).get("history", ()))
    saved = tmp_path / "optional-copy.pt"
    model.save(saved)
    restored = NeuralPhaseModel.load(saved)
    assert restored._weights_sha256 == model._weights_sha256
    assert restored.training_record == model.training_record
    assert model.selected_step == (record or {}).get("selected_step")
    table = model.report()
    assert table.empty == (record is None)
    if record:
        assert table.loc[10, "Selected"] and "0.10000000" in table._repr_html_()
    session = SimpleNamespace(
        backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.2, 0.2)
    )
    model._check(session, 1)
    with pytest.raises(ValueError, match="sampling and optics"):
        model._check(session, 2)

    # On a detector-supported sinusoid, zero learned weights should reproduce each direct image.
    row, col = torch.tensor([[2]]), torch.tensor([[3]])
    group = SimpleNamespace(
        rows=row,
        cols=col,
        gram=torch.ones(1, 1, 1).to(torch.complex64),
        prior=torch.ones(1, 1),
        rhs=torch.ones(1, 1).to(torch.complex64),
    )
    normal = SimpleNamespace(groups=[group], scale=torch.tensor(1.0), factor=1)
    positions = torch.arange(16)
    direct = torch.cos(2 * torch.pi * (2 * positions[:, None] + 3 * positions[None, :]) / 16)
    annular = direct.square()
    weights = model._net
    for current in (direct, -direct, direct):
        result = refine(
            normal,
            current,
            annular,
            (0.2, 0.2),
            weights["unet"],
            weights["heads"],
            weights.readout,
            weights.kernels,
        )
        torch.testing.assert_close(result, current, atol=3e-6, rtol=2e-6)


def test_periodic_annular_interpolation_preserves_nyquist():
    from scipy.signal import resample
    from quantem.gpu.direct_ptycho._phase_refinement import _resample_axis

    counts = np.array([[4, 0, 2, 0], [0, 2, 0, 4], [2, 0, 4, 0]], np.float32)
    result = _resample_axis(_resample_axis(torch.tensor(counts), 9, 0), 12, 1)
    np.testing.assert_allclose(result, resample(resample(counts, 9, axis=0), 12, axis=1), atol=1e-6)


def test_fit_regions_use_disjoint_supported_native_grids():
    from quantem.gpu.direct_ptycho._phase_training import _regions

    left, right = _regions((384, 256), 128)
    assert left == (128, 256, 0, 128) and right == (128, 256, 128, 256)
    from quantem.gpu.direct_ptycho._phase_training import _evaluation_region

    assert _evaluation_region((384, 256), {"train_region": left, "validation_region": right}) == (0, 128, 0, 128)
    with pytest.raises(ValueError, match="unused"):
        _evaluation_region((128, 256), {"train_region": (0, 128, 0, 128), "validation_region": (0, 128, 128, 256)})
    with pytest.raises(ValueError, match="FFT padding"):
        _regions((512, 512), 32)
    with pytest.raises(ValueError, match="at least"):
        _regions((128, 255), 128)


def test_held_equations_match_the_measured_fourier_residual():
    from quantem.gpu.direct_ptycho._phase_training import _gain, _joint_error

    # One observed coefficient of an amplitude + phase object. Score the
    # cached normal equations against the explicit squared diffraction residual.
    row = torch.tensor([[1]])
    col = torch.tensor([[2]])
    phase = torch.zeros((8, 8), dtype=torch.float64)
    phase[2, 3] = 0.37
    eta = phase.roll(1, 0) * 0.25
    phi = torch.fft.fft2(phase)[1, 2]
    amplitude = torch.fft.fft2(eta)[1, 2]
    measurement = torch.tensor(1.3 + 0.9j, dtype=torch.complex128)
    transfer = torch.tensor([0.7 - 0.5j, -0.2 + 1.0j], dtype=torch.complex128)

    def equations(columns):
        gram = torch.outer(columns.conj(), columns)[None]
        group = SimpleNamespace(rows=row, cols=col, gram=gram, rhs=(columns.conj() * measurement)[None])
        return SimpleNamespace(factor=1, groups=[group], signal_power=measurement.abs().square())

    explicit = (transfer[0] * amplitude + transfer[1] * phi - measurement).abs().square()
    expected = float(explicit / measurement.abs().square())
    assert _joint_error(phase, eta, equations(transfer)) == pytest.approx(expected)
    phase_error = ((transfer[1] * phi - measurement).abs() / measurement.abs()).square()
    gain = _gain(phase, torch.zeros_like(phase), equations(transfer[1:]))
    assert float(gain) == pytest.approx(float(1 - phase_error))


def test_annulus_follows_physical_axes_edges_and_detector_validity():
    from quantem.gpu.direct_ptycho._annular_detector import annular_mask, measured_validity

    session = SimpleNamespace(bright_field_center_px=(1, 3), detector_sampling_mrad=(2, 1))
    valid = np.ones((4, 7), dtype=bool)
    valid[1, 5] = False
    scan = SimpleNamespace(metadata={"pixel_mask": (~valid).astype(np.uint32)})
    # Loader-filled pixels remain excluded in both training and inference guides.
    np.testing.assert_array_equal(measured_validity(scan, np.ones_like(valid)), valid)
    mask = annular_mask(session, valid.shape, (2, 3), valid)
    assert set(zip(*np.nonzero(mask))) == {
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 1),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    }
    with pytest.raises(ValueError, match="no valid detector pixels"):
        annular_mask(session, valid.shape, (20, 30), valid)


@pytest.mark.parametrize("bounds", [0, "13", (2,), (1, 2, 3), (-1, 2), (2, 2), (2, 1), (0, np.inf)])
def test_invalid_collection_angles_name_the_physical_control(bounds):
    from quantem.gpu.direct_ptycho._annular_detector import annular_limits

    with pytest.raises(ValueError, match="annulus_mrad"):
        annular_limits(bounds)


def test_direct_neural_comparison_keeps_both_images_visible():
    """Show full fields, matched zooms and FFTs with the correct independent dampings."""
    import matplotlib.pyplot as plt
    from quantem.gpu.direct_ptycho.results import DirectPtychoResult

    phase = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 10000
    result = DirectPtychoResult(
        object_wave=np.exp(1j * phase), backend="mps", neural_phase=phase * 0.9,
        object_sampling_A=(0.1, 0.1), reconstruction_method="phase",
        reconstruction_options={"phase_damping": 0.02, "neural_phase_model": {"input_phase_damping": 0.03}},
    )
    figure = result.show(show_fft=True, cmap="gray")
    images = [axis for axis in figure.axes if axis.images]
    assert len(images) == 6
    assert "Phase damping 0.02" in images[0].get_title()
    assert "Input phase damping 0.03" in images[1].get_title()
    from matplotlib.colors import Normalize

    normalization = Normalize(*np.percentile(np.concatenate([result.phase.ravel(), result.neural_phase.ravel()]), [.5, 99.5]))
    for axis, values in zip(images[:2], (result.phase, result.neural_phase), strict=True):
        expected = plt.get_cmap("gray")(normalization(values))
        np.testing.assert_array_equal(axis.images[0].get_array(), expected)
    assert images[0].images[0].get_clim() == images[1].images[0].get_clim()
    assert figure.number not in plt.get_fignums()


def test_probe_phase_masks_dark_aperture_and_preserves_radian_scale(monkeypatch):
    from quantem.gpu.direct_ptycho.results import DirectPtychoResult

    probe = np.zeros((4, 4), dtype=np.complex64)
    probe[2, 2] = np.exp(0.75j)
    monkeypatch.setattr(DirectPtychoResult, "probe", lambda self, **kwargs: probe)
    monkeypatch.setattr(DirectPtychoResult, "probe_sampling_mrad", property(lambda self: (2.0, 3.0)))
    result = DirectPtychoResult(object_wave=np.ones((4, 4), complex), backend="mps")
    figure = result.show("probe_phase")
    image = figure.axes[0].images[0]
    assert image.get_clim() == (-np.pi, np.pi)
    assert image.get_array().mask.sum() == 15
    assert image.get_array()[2, 2] == pytest.approx(0.75)
    # The zero-angle aperture pixel remains exactly centered at (0, 0).
    assert tuple(image.get_extent()) == (-7.5, 4.5, 3.0, -5.0)
