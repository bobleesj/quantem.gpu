"""Streaming refits only defocus and astigmatism and holds every other coordinate."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.stream import _refit


@pytest.fixture(scope="module")
def cuda_runtime():
    """Skip CUDA registration checks where no CUDA runtime or device is available."""
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable for Direct Ptycho series tests")
    if count == 0:
        pytest.skip("A CUDA device is required for Direct Ptycho series tests")
    return cupy


class PeakedObjective:
    """Model agreement exp(-|z - peak|^2) with its exact gradient: its maximum is known exactly."""

    num_bright_field_pixels = 9000

    def __init__(self, peak):
        self.peak = peak
        self.seen = []

    def evaluate(self, points, order):
        return self.values(points)

    def values(self, points, stride=1):
        points = np.atleast_2d(points)
        self.seen.append(np.array(points))
        return np.exp(-np.sum((points - self.peak) ** 2, axis=1))

    def gradients(self, points, stride=1):
        points = np.atleast_2d(points)
        return -2 * (points - self.peak) * np.exp(-np.sum((points - self.peak) ** 2, axis=1))[:, None]


def test_refit_finds_the_low_order_peak_and_holds_the_rest():
    reference = np.linspace(-.5, .5, 28)
    peak = reference.copy()
    peak[:3] = (.31, -.22, .17)
    objective = PeakedObjective(peak)
    bounds = np.array([(-3., 3.), (-2., 2.), (-2., 2.)])
    coordinates, evaluations, stop = _refit(objective, reference, reference[:3].copy(), bounds)
    np.testing.assert_allclose(coordinates[:3], peak[:3], atol=1e-3)
    np.testing.assert_array_equal(coordinates[3:], reference[3:])
    assert all(np.array_equal(points[:, 3:], np.tile(reference[3:], (len(points), 1))) for points in objective.seen)
    assert evaluations < 300 and stop != "evaluation cap"


def lattice_image(rows, sampling_A, seed):
    """A periodic test specimen: a hexagonal lattice near MoS2's 0.35 Å⁻¹ (spots on the FFT grid, so
    a Fourier shift moves it exactly) over a smooth random background at periods past 3 Å, the
    aperiodic features the coarse registration needs."""
    import cupy

    generator = np.random.default_rng(seed)
    q = np.fft.fftfreq(rows, sampling_A)
    q_size = np.hypot(q[:, None], q[None, :])
    background = np.fft.ifft2(np.fft.fft2(generator.standard_normal((rows, rows))) * (q_size < 0.2) * (q_size > 0.02)).real
    index = np.arange(rows)
    radius = round(0.35 * rows * sampling_A)
    image = 3 * background / background.std()
    for row_g, col_g in ((0, radius), (round(radius * np.sin(np.pi / 3)), round(radius / 2)), (round(radius * np.sin(np.pi / 3)), -round(radius / 2))):
        image += np.cos(2 * np.pi * (row_g * index[:, None] + col_g * index[None, :]) / rows)
    return cupy.asarray(image)


@pytest.mark.usefixtures("cuda_runtime")
def test_place_recovers_each_shift_and_sums_the_frames_in_register():
    """Six noisy frames of one specimen, each moved a known sub-pixel amount further: each is
    registered against the frame before it to 0.02 px, and the sum in register is the specimen with
    the noise averaged down by sqrt(6)."""
    import cupy

    from quantem.gpu.drift import shifted
    from quantem.gpu.direct_ptycho.stream import (
        CONFIDENCE,
        _add,
        _guess,
        _lattice,
        _place,
        _start,
    )

    rows, sampling, noise = 256, (0.2, 0.2), 0.5
    specimen = lattice_image(rows, sampling[0], seed=1)
    truth = np.array([(0, 0), (1.3, -0.7), (6.45, 4.2), (13.1, 9.85), (19.6, 14.3), (26.25, 19.9)])
    generator = cupy.random.default_rng(2)
    frames = [shifted(specimen, shift) + noise * generator.standard_normal((rows, rows)) for shift in truth]
    lattice = _lattice(frames[0], sampling)
    canvas = _start(frames[0])
    found, confidence = [(0.0, 0.0)], []
    for index, frame in enumerate(frames[1:], 1):
        shift, score = _place(canvas, frame, lattice, sampling, 10.0, _guess(list(enumerate(found)), index))
        found.append(shift)
        confidence.append(score)
    assert min(confidence) > CONFIDENCE
    np.testing.assert_allclose(np.array(found), truth, atol=0.02)
    # the canvas grew to hold every frame (frame 1 sits a pixel right of frame 0): frame 0 at its origin
    assert canvas.total.shape == (rows + 26, rows + 21) and canvas.origin == (26, 20)
    combined = canvas.total / cupy.maximum(canvas.cover, 1)
    covered = (canvas.cover == len(frames)).get()
    region = combined[26:26 + rows, 20:20 + rows].get()[covered[26:26 + rows, 20:20 + rows]]
    expected = specimen.get()[covered[26:26 + rows, 20:20 + rows]]
    assert covered.sum() > 0.8 * rows**2
    assert np.std(region - expected) < 1.1 * noise / np.sqrt(len(frames))
    # a frame placed again with a given shift lands exactly where the search put it
    again = _start(frames[0])
    for frame, shift in zip(frames[1:], found[1:]):
        _add(again, frame, shift)
    cupy.testing.assert_array_equal(again.total, canvas.total)


@pytest.mark.usefixtures("cuda_runtime")
def test_align_replaces_the_live_shifts_with_the_pairwise_drift():
    """The final pass runs the pairwise drift on the phase stack, takes its displacement (Å) and
    rebuilds the combined image from it."""
    import cupy

    from quantem.gpu.drift import shifted
    from quantem.gpu.direct_ptycho.stream import DirectPtychoSeries, _add, _combined, _start

    rows, sampling = 256, (0.2, 0.2)
    specimen = lattice_image(rows, sampling[0], seed=3)
    truth = np.array([(0, 0), (2.3, 1.1), (4.45, 2.6), (6.1, 4.35)])
    generator = cupy.random.default_rng(4)
    phase = cupy.stack([shifted(specimen, shift) + 0.3 * generator.standard_normal((rows, rows)) for shift in truth]).astype(cupy.float32)
    series = DirectPtychoSeries(frames=tuple("abcd"), phase=phase, images={"phase_of_mean": phase}, methods=("phase_of_mean",), aberrations=(),
                       refits=(), fits=(), timings=(), reference=None, object_sampling_A=sampling,
                       shifts_A=np.zeros((4, 2)), confidence=(), placed=(True, False, False, False), combined={},
                       cover=None, sources=(), session={})
    found = series.align()
    # Its pairs span up to 6 px of a 256 px field, where register_spectra's Hann-window correction
    # leaves 0.04 px on this specimen (the live pass registers only the residual past its guess).
    np.testing.assert_allclose(series.shifts_A / sampling[0], truth, atol=0.05)
    np.testing.assert_array_equal(series.shifts_A, found.displacement_A)
    canvas = _start(phase[0])
    for frame, shift in zip(phase[1:], found.displacement_A[1:] / sampling[0]):
        _add(canvas, frame, tuple(shift))
    assert series.placed == (True,) * 4
    # frames go to drift as indices, so its plot has a numeric frame axis
    assert found.frames == [0, 1, 2, 3] and found.plot().axes
    cupy.testing.assert_array_equal(series.combined["phase_of_mean"], _combined(canvas))


class FrameObjective(PeakedObjective):
    """One frame's model agreement: its low-order peak is the frame's own, the rest the reference."""

    calls = 0

    def values(self, points, stride=1):
        # the refit reads values and exact gradients; every values call counts as refit work
        FrameObjective.calls += len(np.atleast_2d(points))
        return super().values(points, stride)


class FakeSession:
    """Stands in for DirectPtycho on one loaded frame: the SSB run returns the frame's lattice phase,
    a regularized run returns noise (so a frame registered on it would land elsewhere), and every run
    is recorded with its frame, method, optics and strengths. A session holding a frame counts as
    open until closed: the stream must release every frame's GPU state before reading the next."""

    backend, voltage_kV, semiangle_mrad, scan_sampling_A = "cuda", 300.0, 30.0, 0.2
    detector_sampling_mrad, bright_field_center_px, bright_field_radius_px = 0.5, (96.0, 96.0), 40.0
    runs = []
    open = 0

    def __init__(self, data=None, **settings):
        self.data = data
        FakeSession.open += data is not None

    def close(self):
        FakeSession.open -= self.data is not None
        self.data = None

    def _resolve_upsample(self, upsample):
        from quantem.gpu.direct_ptycho.workflow import DirectPtycho

        return DirectPtycho._resolve_upsample(self, upsample)

    def run(self, chosen, *, method, **options):
        import cupy

        strengths = {name: value for name, value in options.items() if name not in (
            "upsample", "compute_loss", "force", "verbose")}
        FakeSession.runs.append(SimpleNamespace(frame=self.data.name, method=method, chosen=chosen,
                                                optics=dict(chosen.aberrations), strengths=strengths,
                                                factor=options["upsample"]))
        if method == "phase_of_mean":
            return SimpleNamespace(phase=self.data.phase, object_sampling_A=(0.2, 0.2))
        generator = cupy.random.default_rng(sum(map(ord, self.data.name)) * 3 + "spc".index(method[0]))
        noise = generator.standard_normal((2, *self.data.phase.shape))
        return SimpleNamespace(phase=noise[0], amplitude=1 + 0.1 * noise[1], object_sampling_A=(0.2, 0.2))


def fake_stream(monkeypatch, tmp_path, rows=256, noise_frames=()):
    """Five frames of one specimen drifting a known amount, each with its own defocus peak, written
    as small files so their identities are real; returns the sources and the reference fit. A frame
    in noise_frames shows noise only, so its registration cannot be confident."""
    import cupy

    from quantem.gpu.drift import shifted
    from quantem.gpu.direct_ptycho import stream as stream_module

    specimen = lattice_image(rows, 0.2, seed=5)
    truth = [(0, 0), (1.3, -0.7), (3.45, 1.2), (5.1, 2.85), (6.6, 4.3)]
    generator = cupy.random.default_rng(6)
    reference = np.linspace(-.5, .5, 28)
    phases, peaks, sources = {}, {}, []
    for index, shift in enumerate(truth):
        source = tmp_path / "data" / f"frame_{index}_master.h5"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(bytes([index]) * 64)
        sources.append(source)
        content = 0 * specimen if index in noise_frames else shifted(specimen, shift)
        phases[source.name] = (content + 0.3 * generator.standard_normal((rows, rows))).astype(cupy.float32)
        peak = reference.copy()
        peak[:3] = (.1 * index, -.05 * index, .02 * index)
        peaks[source.name] = peak
    loaded = {}

    def load(source, device, verbose=True):
        # the file's content moves the frame's defocus peak, so a rewritten frame refits elsewhere
        loaded["name"], loaded["bytes"] = Path(source).name, len(Path(source).read_bytes())
        return SimpleNamespace(name=Path(source).name, phase=phases[Path(source).name], shape=(rows, rows, 96, 96),
                               metadata={}, close=lambda: None)

    monkeypatch.setattr(stream_module.io, "load", load)
    monkeypatch.setattr(stream_module, "_objective", lambda session, **kwargs: FrameObjective(
        peaks[loaded["name"]] + np.r_[1e-3 * loaded["bytes"], np.zeros(27)]))
    monkeypatch.setattr(stream_module, "public_params", lambda objective, z, order: (
        {"C10_nm": z[0], "C12_nm": z[1], "phi12_deg": z[2]}, (0.0, 0.0), 1.0))
    fit = SimpleNamespace(optimizer={"coordinates": reference.tolist(), "bounds": [(-3., 3.)] * 28},
                          scan_detector_rotation_deg=176.3, scan_detector_rotation_flipped=False,
                          phase_limits=(-1.0, 1.0), metadata={})
    return sources, fit


def streamed(sources, fit, save_dir, **options):
    from quantem.gpu.direct_ptycho.stream import series as series_of

    FrameObjective.calls = 0
    FakeSession.runs, FakeSession.open = [], 0
    series = series_of(FakeSession(), sources, fit, save_dir=save_dir, show=False, verbose=False, **options)
    return series, FrameObjective.calls


def assert_same_stream(left, right):
    import cupy

    assert left.aberrations == right.aberrations
    assert [{name: refit[name] for name in ("evaluations", "stop")} for refit in left.refits] ==\
        [{name: refit[name] for name in ("evaluations", "stop")} for refit in right.refits]
    np.testing.assert_array_equal(left.shifts_A, right.shifts_A)
    assert left.confidence == right.confidence
    cupy.testing.assert_array_equal(left.phase, right.phase)
    assert left.combined.keys() == right.combined.keys()
    for key in left.combined:
        cupy.testing.assert_array_equal(left.combined[key], right.combined[key])


@pytest.mark.usefixtures("cuda_runtime")
def test_stream_resumes_from_saved_records_without_refitting(monkeypatch, tmp_path):
    """A second call with the same fit and sources refits nothing: every frame is reconstructed from
    its saved optics and placed at its saved shift, and the series equals the first call exactly."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    first, first_calls = streamed(sources, fit, tmp_path / "results")
    assert first_calls > 0
    assert sorted(path.name for path in (tmp_path / "results" / "stream").iterdir()) ==\
        [f"frame_{index}.json" for index in range(5)]
    again, again_calls = streamed(sources, fit, tmp_path / "results")
    assert again_calls == 0
    assert all(refit["resumed"] for refit in again.refits)
    assert_same_stream(first, again)


@pytest.mark.usefixtures("cuda_runtime")
def test_an_interrupted_stream_continues_as_if_never_stopped(monkeypatch, tmp_path):
    """Streaming the first three frames, then all five, equals one uninterrupted run: the first three
    come from their records and the fourth refits from the third's saved low-order values."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    whole, _ = streamed(sources, fit, None)
    streamed(sources[:3], fit, tmp_path / "results")
    resumed, _ = streamed(sources, fit, tmp_path / "results")
    assert [refit["resumed"] for refit in resumed.refits] == [True] * 3 + [False] * 2
    assert_same_stream(whole, resumed)


@pytest.mark.usefixtures("cuda_runtime")
def test_records_of_another_source_or_fit_are_ignored(monkeypatch, tmp_path):
    """A record keys on the frame's file content and on the reference fit: a rewritten frame and every
    frame after it are fitted again, and a different reference fit refits them all."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    streamed(sources, fit, tmp_path / "results")
    sources[2].write_bytes(b"another acquisition")
    changed, _ = streamed(sources, fit, tmp_path / "results")
    assert [refit["resumed"] for refit in changed.refits] == [True, True, False, False, False]
    other = SimpleNamespace(**{**vars(fit), "optimizer": {**fit.optimizer, "coordinates": (np.array(fit.optimizer["coordinates"]) + 1e-9).tolist()}})
    refit, calls = streamed(sources, other, tmp_path / "results")
    assert calls > 0 and not any(record["resumed"] for record in refit.refits)


@pytest.mark.usefixtures("cuda_runtime")
def test_a_rewritten_frame_invalidates_every_later_record_even_if_interrupted(monkeypatch, tmp_path):
    """Frame 2 is rewritten and recomputed, then the run stops before frames 3 and 4 are: their old
    records came from the old frame 2, so the next call refits them and equals an uninterrupted run."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    streamed(sources, fit, tmp_path / "results")
    sources[2].write_bytes(b"another acquisition")
    streamed(sources[:3], fit, tmp_path / "results")
    resumed, _ = streamed(sources, fit, tmp_path / "results")
    assert [refit["resumed"] for refit in resumed.refits] == [True, True, True, False, False]
    whole, _ = streamed(sources, fit, None)
    assert_same_stream(whole, resumed)


@pytest.mark.usefixtures("cuda_runtime")
def test_a_corrupt_record_is_computed_again(monkeypatch, tmp_path):
    """A record cut short is treated as missing: that frame and the ones after it are refit, and the
    records are strict JSON (frame 0's confidence is null, not Infinity)."""
    import json

    sources, fit = fake_stream(monkeypatch, tmp_path)
    first, _ = streamed(sources, fit, tmp_path / "results")
    records = tmp_path / "results" / "stream"
    assert json.loads((records / "frame_0.json").read_text(), parse_constant=lambda name: 1 / 0)["confidence"] is None
    (records / "frame_1.json").write_text((records / "frame_1.json").read_text()[:40])
    again, _ = streamed(sources, fit, tmp_path / "results")
    assert [refit["resumed"] for refit in again.refits] == [True, False, False, False, False]
    assert_same_stream(first, again)


def placed_by_hand(images, shifts_A, placed, sampling=0.2, cover=False):
    """The combined image built frame by frame at the given shifts, the frames in ``placed`` only
    (or, with cover, how many frames reach each pixel)."""
    from quantem.gpu.direct_ptycho.stream import _add, _combined, _start

    canvas = _start(images[0])
    for image, shift, keep in zip(images[1:], shifts_A[1:], placed[1:]):
        if keep:
            _add(canvas, image, tuple(shift / sampling))
    return canvas.cover if cover else _combined(canvas)


@pytest.mark.usefixtures("cuda_runtime")
def test_stream_reconstructs_each_requested_method_at_the_frames_optics(monkeypatch, tmp_path):
    """method=("phase", "complex") runs SSB (for registration) and both regularized methods once per
    frame, each at that frame's refit optics with the strengths given; images and combined hold the
    phase of each method and the complex method's amplitude, every image moved by the same shifts."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method=("phase", "complex"), phase_damp=0.07,
                         amplitude_damp=0.2)
    assert list(series.images) == ["phase", "complex", "amplitude"]
    assert list(series.combined) == ["phase", "complex", "amplitude"]
    assert series.methods == ("phase", "complex")
    assert [(run.frame, run.method) for run in FakeSession.runs] == [
        (source.name, method) for source in sources for method in ("phase_of_mean", "phase", "complex")]
    for run in FakeSession.runs:
        index = [source.name for source in sources].index(run.frame)
        assert run.chosen is series.fits[index]
        assert run.optics == series.aberrations[index] == series.fits[index].aberrations
        assert run.strengths == ({} if run.method == "phase_of_mean" else
                                 {"phase_damp": 0.07, "amplitude_damp": 0.2})
    assert "evaluations" not in series.aberrations[0] and set(series.refits[0]) == {"evaluations", "stop", "resumed"}
    assert series.phase.shape == series.images["phase"].shape == (5, 256, 256)
    latest = series.latest
    assert latest.index == 4 and latest.neural_phase is None
    cupy.testing.assert_array_equal(latest.phase, series.images["phase"][4])
    cupy.testing.assert_array_equal(latest.complex_phase, series.images["complex"][4])
    cupy.testing.assert_array_equal(latest.amplitude, series.images["amplitude"][4])
    # the regularized images are the fake's noise, placed with the shifts measured on the SSB phase
    for key in series.images:
        live = placed_by_hand(series.images[key], np.array(series.shifts_A), series.placed)
        cupy.testing.assert_allclose(series.combined[key], live, atol=1e-6)
    table = series.report()
    assert {"Phase of the mean (s)", "Smoothed phase (s)", "Phase and amplitude (s)", "Near threshold"} <= set(table.columns)


@pytest.mark.usefixtures("cuda_runtime")
def test_a_capped_auto_factor_is_announced_once_per_series(monkeypatch, tmp_path, capsys):
    """At a 1 A step (300 kV, 30 mrad) the scan sampling supports 8x: auto is resolved once to 4x, one line
    says so, and every frame runs at 4x; an explicit 8 runs at 8x without the line."""
    from quantem.gpu.direct_ptycho.stream import series

    sources, fit = fake_stream(monkeypatch, tmp_path)
    monkeypatch.setattr(FakeSession, "scan_sampling_A", 1.0)
    for requested, factor, notice in (("auto", 4, "SSB: auto upsample capped at 4× (the scan sampling supports "
                                                  "8×); pass upsample=8 for the full factor.\n"),
                                      (8, 8, "")):
        FakeSession.runs, FakeSession.open = [], 0
        capsys.readouterr()
        series(FakeSession(), sources, fit, upsample=requested, show=False, verbose=False)
        assert capsys.readouterr().out == notice
        assert {run.factor for run in FakeSession.runs} == {factor}


@pytest.mark.usefixtures("cuda_runtime")
def test_registration_uses_the_ssb_phase_whatever_the_method(monkeypatch, tmp_path):
    """A stream asking for the phase method only still registers each frame on its SSB phase: the
    shifts and confidences equal an SSB stream's exactly, though the regularized images are noise."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path)
    ssb, _ = streamed(sources, fit, None)
    phase, _ = streamed(sources, fit, None, method="phase")
    np.testing.assert_array_equal(phase.shifts_A, ssb.shifts_A)
    assert phase.confidence == ssb.confidence and all(phase.placed)
    cupy.testing.assert_array_equal(phase.phase, ssb.phase)
    assert list(phase.combined) == ["phase"] and list(ssb.combined) == ["phase_of_mean"]
    cupy.testing.assert_array_equal(ssb.images["phase_of_mean"], ssb.phase)


@pytest.mark.usefixtures("cuda_runtime")
def test_reconstruct_rebuilds_a_method_from_the_saved_optics_and_shifts(monkeypatch, tmp_path):
    """After a stream, run("phase", ...) reloads each frame, runs the phase method once per
    frame at fits[k] with the new strength, and places it at shifts_A; nothing is refit."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None)
    FakeSession.runs, FrameObjective.calls = [], 0
    combined = series.run("phase", phase_damp=0.05)
    assert FrameObjective.calls == 0
    assert [(run.frame, run.method, run.strengths) for run in FakeSession.runs] == [
        (source.name, "phase", {"phase_damp": 0.05}) for source in sources]
    assert all(run.chosen is chosen for run, chosen in zip(FakeSession.runs, series.fits, strict=True))
    assert list(combined) == ["phase"] and series.methods == ("phase_of_mean", "phase")
    assert "phase damp 0.05" in series.show("phase").axes[0].get_title()
    cupy.testing.assert_array_equal(combined["phase"], series.combined["phase"])
    cupy.testing.assert_array_equal(series.combined["phase"],
                                    placed_by_hand(series.images["phase"], series.shifts_A, series.placed))


@pytest.mark.usefixtures("cuda_runtime")
def test_resume_with_methods_refits_nothing(monkeypatch, tmp_path):
    """Records hold optics and shifts, not images: a second call asking for other methods resumes
    every frame from its record with no refit and reconstructs the methods it asks for."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    streamed(sources, fit, tmp_path / "results")
    again, calls = streamed(sources, fit, tmp_path / "results", method=("phase", "complex"))
    assert calls == 0 and all(refit["resumed"] for refit in again.refits)
    assert [(run.frame, run.method) for run in FakeSession.runs] == [
        (source.name, method) for source in sources for method in ("phase_of_mean", "phase", "complex")]
    whole, _ = streamed(sources, fit, None, method=("phase", "complex"))
    assert_same_stream(whole, again)


@pytest.mark.usefixtures("cuda_runtime")
def test_near_threshold_flags_confidence_just_above_the_cut(monkeypatch, tmp_path):
    """Near threshold marks a frame whose confidence lies in [CONFIDENCE, 1.1 CONFIDENCE): placed,
    but close enough to the cut to deserve a look."""
    from quantem.gpu.direct_ptycho.stream import CONFIDENCE

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None)
    series.confidence = (float("inf"), CONFIDENCE, 1.05 * CONFIDENCE, 1.1 * CONFIDENCE, 0.5 * CONFIDENCE)
    assert series.report()["Near threshold"].tolist() == [False, True, True, False, False]


@pytest.mark.usefixtures("cuda_runtime")
def test_show_draws_each_combined_image_and_the_drift(monkeypatch, tmp_path):
    """Static: one image panel per combined image (phase, complex phase, amplitude) over the drift
    strip, the amplitude on its own limits; interactive: a Show2D of the latest scan, labelled with its fitted
    defocus and astigmatism, then the same images by name."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method=("phase", "complex"))
    figure = series.show()
    images = [axis for axis in figure.axes if axis.images]
    assert [axis.get_title().split("\n")[0] for axis in images] == [
        "Direct phase", "Complex-fit phase", "Relative amplitude"]
    assert len({round(axis.get_position().x0, 3) for axis in images}) == 2
    assert "phase damp 0.3" in images[0].get_title()
    assert "amplitude damp 0.3" in images[2].get_title()
    assert images[0].images[0].get_clim() == fit.phase_limits != images[2].images[0].get_clim()
    assert len(series.show("phase").axes) == len(figure.axes) - 4
    single, _ = streamed(sources[:1], fit, None, method=("phase", "complex"))
    single_figure = single.show()
    assert len(single_figure.axes) == len(figure.axes) - 2
    assert all("single scan" in axis.get_title() for axis in single_figure.axes if axis.images)
    viewer = series.show(interactive=True)
    assert type(viewer).__name__ == "Show2D" and viewer.n_images == 4
    assert viewer.labels[0].startswith(f"{series.frames[-1]}: C10 ")


@pytest.mark.usefixtures("cuda_runtime")
def test_every_session_is_closed_before_the_next_frame(monkeypatch, tmp_path):
    """Each frame's session (and reconstruct's) is closed once used, so no frame's GPU state waits for
    the cycle collector; none is left open or alive after the series."""
    import gc

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method="phase")
    assert FakeSession.open == 0
    series.run("complex")
    assert FakeSession.open == 0
    assert not [item for item in gc.get_objects() if type(item) is FakeSession and item.data is not None]


@pytest.mark.parametrize("stage", ["reconstruction", "registration", "view"])
@pytest.mark.parametrize("resident", [True, False])
def test_frame_failure_closes_workspace_and_preserves_borrowed_counts(monkeypatch, tmp_path, stage, resident):
    """CPU ownership test: a failed frame closes its session, but only its own file input."""
    from quantem.gpu.direct_ptycho import stream as stream_module

    closed = []
    data = SimpleNamespace(metadata={"series": "one"}, shape=(2, 2, 2, 2),
                           close=lambda: closed.append("file input"))
    fit = SimpleNamespace(optimizer={"coordinates": [0.] * 28, "bounds": [(-1., 1.)] * 28},
                          aberrations={"C10_nm": 0., "C12_nm": 0., "phi12_deg": 0.},
                          sample_tilt_mrad=(0., 0.), depth_spread_nm=0.,
                          scan_detector_rotation_deg=0., scan_detector_rotation_flipped=False,
                          phase_limits=(-1., 1.))
    monkeypatch.setattr(stream_module, "cp", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0), Device=lambda: SimpleNamespace(synchronize=lambda: None))))
    monkeypatch.setattr(stream_module.io, "load", lambda *args, **kwargs: data)
    monkeypatch.setattr(FakeSession, "_resolve_upsample", lambda self, upsample: 1)
    monkeypatch.setattr(stream_module, "_combined", lambda canvas: np.ones((2, 2)))
    monkeypatch.setattr(stream_module.DirectPtychoSeries, "_store",
                        lambda *args: (np.ones((1, 2, 2)), {"phase": np.ones((1, 2, 2))}))

    def run(*args):
        if stage == "reconstruction":
            raise RuntimeError(stage)
        return fit, {"phase_of_mean": np.ones((2, 2)), "phase": np.ones((2, 2))}, (1., 1.), {"ssb_s": 0., "phase_s": 0.}

    def place(step, *args):
        if stage == "registration":
            raise RuntimeError(stage)
        step.canvases = {key: SimpleNamespace(cover=np.ones((2, 2))) for key in ("phase_of_mean", "phase")}
        return (0., 0.), 1.

    def redraw(self):
        raise RuntimeError("view")

    monkeypatch.setattr(stream_module._FrameStep, "run", run)
    monkeypatch.setattr(stream_module._FrameStep, "place", place)
    monkeypatch.setattr(stream_module.DirectPtychoSeries, "_redraw", redraw)
    FakeSession.open = 0
    source = data if resident else tmp_path / "one_master.h5"
    with pytest.raises(RuntimeError, match=stage):
        stream_module.series(FakeSession(), [source], fit, refit="none", method="phase",
                             upsample=1, show=False, verbose=False)
    assert FakeSession.open == 0
    assert closed == ([] if resident else ["file input"])


@pytest.mark.usefixtures("cuda_runtime")
def test_a_frame_of_another_scan_shape_stops_the_stream(monkeypatch, tmp_path):
    """A frame scanned at another size cannot join the series: the stream stops right after reading
    it, naming both shapes, before any refit or reconstruction of it."""
    import pytest

    from quantem.gpu.direct_ptycho import stream as stream_module

    sources, fit = fake_stream(monkeypatch, tmp_path)
    load = stream_module.io.load

    def odd(source, device, verbose=True):
        data = load(source, device, verbose)
        if Path(source).name == sources[2].name:
            data.shape = (128, 256, 96, 96)
        return data

    monkeypatch.setattr(stream_module.io, "load", odd)
    with pytest.raises(ValueError, match=r"frame_2.*\(128, 256\).*\(256, 256\)"):
        streamed(sources, fit, None, method="phase")
    assert {run.frame for run in FakeSession.runs} == {sources[0].name, sources[1].name}


@pytest.mark.usefixtures("cuda_runtime")
def test_a_frame_below_confidence_stays_out_of_every_combined_image(monkeypatch, tmp_path):
    """A frame registered below CONFIDENCE is kept in the stacks but left out of every combined image,
    which each equal the by-hand sum of the confident frames. CONFIDENCE (0.04) is set from the
    measured series; on this specimen good pairs score 0.39 and the noise frame 0.053, so the cut is
    raised to 0.1 here to put the noise frame below it."""
    import cupy

    from quantem.gpu.direct_ptycho import stream as stream_module

    monkeypatch.setattr(stream_module, "CONFIDENCE", 0.1)
    sources, fit = fake_stream(monkeypatch, tmp_path, noise_frames=(3,))
    series, _ = streamed(sources, fit, None, method=("phase", "complex"))
    assert series.confidence[3] < 0.1 < min(series.confidence[4], *series.confidence[1:3])
    assert series.placed == (True, True, True, False, True)
    for key in series.combined:
        cupy.testing.assert_allclose(series.combined[key], placed_by_hand(series.images[key], series.shifts_A,
                                                                          series.placed), atol=1e-6)
    cupy.testing.assert_array_equal(series.cover, placed_by_hand(cupy.ones_like(series.phase), series.shifts_A,
                                                                 series.placed, cover=True))


@pytest.mark.usefixtures("cuda_runtime")
def test_reconstruct_after_align_places_every_frame_at_the_pairwise_shifts(monkeypatch, tmp_path):
    """After align, reconstruct places every frame, including one the live pass left out, at
    align's shifts_A, and report() shows the pairwise σ in place of the live threshold flag."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path, noise_frames=(3,))
    series, _ = streamed(sources, fit, None)
    found = series.align()
    combined = series.run("phase")
    assert series.placed == (True,) * 5
    np.testing.assert_array_equal(series.shifts_A, found.displacement_A)
    cupy.testing.assert_array_equal(combined["phase"], placed_by_hand(series.images["phase"], found.displacement_A,
                                                                      series.placed))
    table = series.report()
    assert "Near threshold" not in table.columns and "Live confidence" in table.columns
    np.testing.assert_array_equal(table[["σ row (Å)", "σ col (Å)"]].to_numpy(), found.sigma_A)


@pytest.mark.usefixtures("cuda_runtime")
def test_reconstruct_ssb_keeps_phase_the_ssb_stack_and_methods_are_unique(monkeypatch, tmp_path):
    """Asking twice for a method runs it once a frame; reconstructing SSB replaces phase too, so the
    registration stack and images["phase_of_mean"] stay one array."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method=("phase_of_mean", "phase", "phase_of_mean", "phase"))
    assert series.methods == ("phase_of_mean", "phase") and len(FakeSession.runs) == 2 * len(sources)
    series.run("phase_of_mean")
    assert series.phase is series.images["phase_of_mean"]


@pytest.mark.usefixtures("cuda_runtime")
def test_flagged_pixel_handling_is_announced_once_for_the_series(monkeypatch, tmp_path, capsys):
    """Each frame's load is quiet, so a long series prints no line per frame; what the loads did with
    the detector's flagged pixels still changes the data, so it is announced once, by outcome."""
    from quantem.gpu.resident.hot_pixels import hot_pixel_record
    from quantem.gpu.direct_ptycho import stream as stream_module
    from quantem.gpu.direct_ptycho.stream import series

    sources, fit = fake_stream(monkeypatch, tmp_path)
    load = stream_module.io.load
    quiet = []
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[3, 4] = mask[50, 60] = 1

    def flagged(source, device, verbose=True):
        quiet.append(verbose)
        data = load(source, device, verbose)
        data.metadata["hot_pixel_correction"] = hot_pixel_record(mask, "median", backend="cuda")
        return data

    monkeypatch.setattr(stream_module.io, "load", flagged)
    FakeSession.runs, FakeSession.open = [], 0
    series(FakeSession(), sources, fit, show=False, verbose=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Series:")]
    assert quiet == [False] * len(sources)
    assert lines == [f"Series: {len(sources)} of {len(sources)} frames, 2 bad pixels replaced by neighbor median "
                     '(hot_pixel_correction="none" keeps the recorded values).']


@pytest.mark.usefixtures("cuda_runtime")
def test_resident_scans_give_the_series_of_their_files_and_stay_the_callers(monkeypatch, tmp_path):
    """A live stream's scans (resident ANS datasets) give the series their recorded files give; the
    series reads no file for them and closes none of them, in the series and in reconstruct, and a
    save_dir, which identifies frames by file, is refused for them."""
    from quantem.gpu.direct_ptycho import stream as stream_module

    sources, fit = fake_stream(monkeypatch, tmp_path)
    from_files, _ = streamed(sources, fit, None, method="phase")
    closed, scans, objectives = [], [], {}
    for source in sources:
        scan = stream_module.io.load(source, device="cuda:0")
        scan.close = lambda name=source.name: closed.append(name)
        objectives[scan.name] = stream_module._objective(None)
        scans.append(scan)
    monkeypatch.setattr(stream_module, "_objective", lambda session, **kwargs: objectives[session.data.name])
    monkeypatch.setattr(stream_module.io, "load", lambda *args, **kwargs: pytest.fail("a resident scan was read"))
    from_scans, _ = streamed(scans, fit, None, method="phase")
    assert_same_stream(from_files, from_scans)
    assert from_scans.frames == tuple(f"scan {index + 1}" for index in range(len(scans)))
    from_scans.run("complex")
    assert closed == [] and FakeSession.open == 0
    with pytest.raises(ValueError, match="resident scans"):
        streamed(scans, fit, tmp_path / "records")


def test_series_without_a_fit_fits_this_session_once_and_uses_that_fit(monkeypatch):
    """ptycho.series(scans) needs no fit: the session's own scan is fitted once (quietly) and that fit drives the
    per-scan refit; a given fit is used as is."""
    from quantem.gpu.direct_ptycho import stream as stream_module
    from quantem.gpu.direct_ptycho.workflow import DirectPtycho

    fitted, calls, used = object(), [], []
    session = SimpleNamespace(_device=None, fit_aberrations=lambda **options: calls.append(options) or fitted)
    monkeypatch.setattr(stream_module, "series", lambda ptycho, sources, fit, **options: used.append(fit) or "series")
    assert DirectPtycho.series(session, ["a", "b"], verbose=False) == "series"
    assert calls == [{"show": False, "verbose": False}] and used == [fitted]
    given = object()
    DirectPtycho.series(session, ["a"], given)
    assert len(calls) == 1 and used[-1] is given


@pytest.mark.usefixtures("cuda_runtime")
def test_a_series_built_scan_by_scan_equals_the_series_of_all_scans(monkeypatch, tmp_path):
    """A series started empty and given each scan with add, as a live stream does when every scan completes,
    is the series of all of them at once (optics, refits, shifts, confidence, every stack and combined image),
    its stacks growing past their first size; each add returns that scan's report row, and an open interactive
    view is redrawn with every scan."""
    from quantem.gpu.direct_ptycho.stream import DirectPtychoSeries

    sources, fit = fake_stream(monkeypatch, tmp_path)
    at_once, _ = streamed(sources, fit, None, method="phase")
    FakeSession.runs, FakeSession.open = [], 0
    growing = DirectPtychoSeries._begin(FakeSession(), fit, refit="defocus", method="phase", upsample="auto", save_dir=None,
                               show=False, strengths={})
    rows = [growing.add(sources[0])]
    viewer = growing.show(interactive=True)
    rows += [growing.add(source) for source in sources[1:]]
    assert_same_stream(at_once, growing)
    assert [row["Frame"] for row in rows] == list(growing.frames)
    assert [row["C10 (nm)"] for row in rows] == [optics["C10_nm"] for optics in at_once.aberrations]
    assert len(growing._state.stacks["phase_of_mean"]) == 8 and growing.phase.shape[0] == len(sources)
    assert viewer.labels[0].startswith(f"{growing.frames[-1]}: ") and viewer.title == f"5 scans, {sum(growing.placed)} combined"
    assert FakeSession.open == 0


@pytest.mark.usefixtures("cuda_runtime")
def test_a_live_scan_is_named_by_its_acquisition(monkeypatch, tmp_path):
    """A resident scan carrying its acquisition id (series metadata) is reported under it."""
    from quantem.gpu.direct_ptycho import stream as stream_module

    sources, fit = fake_stream(monkeypatch, tmp_path)
    scans = []
    for index, source in enumerate(sources[:2]):
        scan = stream_module.io.load(source, device="cuda:0")
        scan.metadata = {"series": f"acquisition-{index:03d}"}
        scans.append(scan)
    objectives = {scan.name: stream_module._objective(None) for scan in scans}
    monkeypatch.setattr(stream_module, "_objective", lambda session, **kwargs: objectives[session.data.name])
    series, _ = streamed(scans, fit, None)
    assert series.frames == ("acquisition-000", "acquisition-001")


@pytest.mark.usefixtures("cuda_runtime")
def test_full_refit_fits_every_scan_independently(monkeypatch, tmp_path):
    """refit="full" runs fit_aberrations on each scan's session, rotation kept, and reconstructs at that fit's
    optics; the low-order refit is not run, and saving records (which resume a defocus refit) is refused."""
    from quantem.gpu.direct_ptycho import stream as stream_module

    sources, fit = fake_stream(monkeypatch, tmp_path)
    fits = []

    def fit_aberrations(self, **options):
        fits.append(options)
        value = 0.1 * len(fits)
        return SimpleNamespace(aberrations={"C10_nm": value, "C12_nm": -value, "phi12_deg": 10 * value},
                               sample_tilt_mrad=(0.0, 0.0), depth_spread_nm=1.0,
                               optimizer={"coordinates": np.zeros(28).tolist(), "evaluations": 7})

    monkeypatch.setattr(FakeSession, "fit_aberrations", fit_aberrations, raising=False)
    monkeypatch.setattr(stream_module, "_objective", lambda *args, **kwargs: pytest.fail("the low-order refit ran"))
    series, _ = streamed(sources, fit, None, refit="full")
    assert len(fits) == len(sources) and all(options["check_rotation"] is False for options in fits)
    assert [optics["C10_nm"] for optics in series.aberrations] == pytest.approx([0.1 * (index + 1) for index in range(5)])
    assert [run.optics["C10_nm"] for run in FakeSession.runs] == [optics["C10_nm"] for optics in series.aberrations]
    assert [refit["evaluations"] for refit in series.refits] == [7] * 5
    with pytest.raises(ValueError, match="defocus refits"):
        streamed(sources, fit, tmp_path / "records", refit="full")


@pytest.mark.usefixtures("cuda_runtime")
def test_fixed_fit_reconstructs_each_scan_without_changing_any_optics(monkeypatch, tmp_path):
    """Live additions equal the recorded series while all initial optics stay fixed."""
    from quantem.gpu.direct_ptycho import stream as stream_module

    sources, fit = fake_stream(monkeypatch, tmp_path)
    fit.aberrations = {"C10_nm": -12.5, "C12_nm": 1.25, "phi12_deg": 17.0, "C30_nm": 650.0}
    fit.sample_tilt_mrad, fit.depth_spread_nm = (1.2, -0.7), 3.5

    def unexpected(*args, **kwargs):
        pytest.fail("fixed optics must not enter either fit")

    monkeypatch.setattr(stream_module, "_objective", unexpected)
    monkeypatch.setattr(FakeSession, "fit_aberrations", unexpected, raising=False)
    whole, calls = streamed(sources, fit, None, refit="none", method=("phase_of_mean", "phase", "complex"))
    assert calls == 0
    for used in whole.fits:
        assert used.aberrations == fit.aberrations
        assert used.aberrations is not fit.aberrations
        assert used.sample_tilt_mrad == fit.sample_tilt_mrad
        assert used.depth_spread_nm == fit.depth_spread_nm
        assert used.scan_detector_rotation_deg == fit.scan_detector_rotation_deg
        assert used.scan_detector_rotation_flipped == fit.scan_detector_rotation_flipped
    assert all(item["evaluations"] == 0 and item["stop"] == "fixed initial fit" for item in whole.refits)
    assert all(item["refit_s"] == 0 for item in whole.timings)
    assert all(run.optics == fit.aberrations for run in FakeSession.runs)
    growing, _ = streamed([], fit, None, refit="none", method=("phase_of_mean", "phase", "complex"))
    for source in sources:
        row = growing.add(source)
        assert row["Refit evaluations"] == 0
    assert_same_stream(whole, growing)
    with pytest.raises(ValueError, match="defocus refits"):
        streamed(sources, fit, tmp_path / "records", refit="none")


@pytest.mark.usefixtures("cuda_runtime")
def test_no_scan_is_added_after_align_or_run(monkeypatch, tmp_path):
    """align and reconstruct rebuild the stacks and shifts adding a scan would extend: adding then stops clearly."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources[:3], fit, None)
    series.run("phase_of_mean")
    with pytest.raises(RuntimeError, match="start a new series"):
        series.add(sources[3])


@pytest.fixture
def rolling_bookkeeping(monkeypatch):
    """Tiny CPU stand-ins exercise ownership and publication, not reconstruction or registration math."""
    from contextlib import nullcontext

    from quantem.gpu.direct_ptycho import stream as stream_module
    from quantem.gpu.direct_ptycho.workflow import DirectPtycho

    class Device(nullcontext):
        def synchronize(self):
            pass

    array_names = ("asarray", "empty", "empty_like", "float32", "float64", "full_like", "maximum",
                   "median", "ones", "pad", "where", "zeros")
    fake_arrays = SimpleNamespace(**{name: getattr(np, name) for name in array_names},
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0), Device=Device),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None))
    monkeypatch.setattr(stream_module, "cp", fake_arrays)
    monkeypatch.setattr(stream_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(stream_module, "_lattice", lambda *args: None)
    monkeypatch.setattr(stream_module, "shifted", lambda image, shift: np.array(image, dtype=np.float64))
    monkeypatch.setattr(stream_module, "hot_pixel_outcome", lambda record: str(record))
    monkeypatch.setattr(FakeSession, "_device", None, raising=False)
    monkeypatch.setattr(FakeSession, "_resolve_upsample", lambda self, upsample: 1)
    fit = SimpleNamespace(optimizer={"coordinates": [0.] * 28, "bounds": [(-1., 1.)] * 28},
                          aberrations={"C10_nm": 0., "C12_nm": 0., "phi12_deg": 0.},
                          sample_tilt_mrad=(0., 0.), depth_spread_nm=0.,
                          scan_detector_rotation_deg=0., scan_detector_rotation_flipped=False,
                          phase_limits=(-1., 1.), metadata={})

    class Source:
        shape = (4, 4, 2, 2)

        def __init__(self, index):
            self.index = index
            self.metadata = {"series": "repeated label", "hot_pixel_correction": str(index)}

        def close(self):
            pytest.fail("resident counts belong to the caller")

    def run(step, session, optics, tilt, depth, *args):
        image = np.arange(16, dtype=np.float32).reshape(4, 4) + 100 * session.data.index
        chosen = SimpleNamespace(**vars(fit))
        chosen.source_index = session.data.index
        chosen.aberrations, chosen.sample_tilt_mrad, chosen.depth_spread_nm = optics, tilt, depth
        return chosen, {"phase_of_mean": image, "phase": image + 10}, (0.2, 0.2), {
            "ssb_s": float(session.data.index), "phase_s": 0.}

    anchors = []

    def place(canvas, phase, lattice, sampling, reach, guess, *, accumulate=True):
        index = int(phase[0, 0] / 100)
        anchors.append((index, canvas.last.whole))
        shift, score = (float(index), -float(index)), 0.01 if index in (3, 4, 5) else 0.5
        if score >= stream_module.CONFIDENCE:
            if accumulate:
                stream_module._add(canvas, phase, shift)
            else:
                canvas.last = stream_module._positioned(phase, shift)
        return shift, score

    monkeypatch.setattr(stream_module._FrameStep, "run", run)
    monkeypatch.setattr(stream_module, "_place", place)
    FakeSession.open = 0

    def begin(max_frames):
        return DirectPtycho.series(FakeSession(), fit=fit, refit="none", method="phase",
                                  max_frames=max_frames, show=False, verbose=False)

    return SimpleNamespace(begin=begin, Source=Source, anchors=anchors, module=stream_module)


@pytest.mark.parametrize("max_frames", [1, 3])
def test_rolling_results_keep_absolute_rows_and_release_evicted_owners(rolling_bookkeeping, max_frames):
    """Repeated scan names stay distinct, rejected windows stay empty, and old owners leave the series."""
    import gc
    import weakref

    setup = rolling_bookkeeping
    result = setup.begin(max_frames)
    owners = []
    for index in range(40):
        scan = setup.Source(index)
        owners.append(weakref.ref(scan))
        row = result.add(scan)
        if index == 0:
            first_snapshot = result.phase
        expected = tuple(range(max(0, index + 1 - max_frames), index + 1))
        assert result.frame_indices == expected and result.total_frames == index + 1
        assert tuple(result.report().index) == tuple(result.report(to_dict=True)) == expected
        assert row["Frame"] == "repeated label" and row["Refit evaluations"] == 0
        assert len(result.sources) == len(result.fits) == len(result.timings) == len(result.frames) == len(expected)
        assert tuple(item.index for item in result.sources) == tuple(item.source_index for item in result.fits) == expected
        assert tuple(row["ssb_s"] for row in result.timings) == expected
        assert len(result._state.flagged) == len(result._state.shifts_px) == len(expected)
        assert len(result._state.step.placed) <= 2
        assert result.phase.shape == (len(expected), 4, 4)
        np.testing.assert_array_equal(result.phase[:, 0, 0], np.array(expected) * 100)
        np.testing.assert_array_equal(result.shifts_A[:, 0], np.array(expected) * 0.2)
        if index == 5:
            assert not any(result.placed) and not result.cover.any()
            assert result._state.step.anchor.last.whole == (2, -2)
            images, labels, _, _ = result._view(["phase"])
            assert all(np.isfinite(image).all() for image in images)
            assert "rejected, not combined" in labels[0] and "confidence 0.010" in labels[0]
            assert "no accepted scans" in labels[1]
    gc.collect()
    assert all(owner() is None for owner in owners[:-max_frames])
    assert all(owner() is not None for owner in owners[-max_frames:])
    assert setup.anchors[5] == (6, (2, -2))  # The confident anchor survived outside the result window.
    np.testing.assert_array_equal(first_snapshot[0], np.arange(16).reshape(4, 4))
    assert len(result._state.stacks["phase_of_mean"]) == max_frames and FakeSession.open == 0
    assert result._state.step.canvases["phase_of_mean"].total.shape == (max_frames + 3, max_frames + 3)
    with pytest.raises(RuntimeError, match="max_frames=None"):
        result.align()


def test_failed_window_rebuild_preserves_published_history_and_anchor(rolling_bookkeeping, monkeypatch):
    """A publication failure closes its workspace without replacing the previous usable window."""
    setup = rolling_bookkeeping
    result = setup.begin(2)
    for index in range(2):
        result.add(setup.Source(index))
    phase, step, stacks = result.phase, result._state.step, result._state.stacks

    def failed(*args, **kwargs):
        raise RuntimeError("combined allocation failed")

    with monkeypatch.context() as patch:
        patch.setattr(setup.module, "_window_canvas", failed)
        with pytest.raises(RuntimeError, match="combined allocation failed"):
            result.add(setup.Source(2))
    assert result.phase is phase and result._state.step is step and result._state.stacks is stacks
    assert result.frame_indices == (0, 1) and result.total_frames == 2 and FakeSession.open == 0
    result.add(setup.Source(2))
    assert result.frame_indices == (1, 2) and result._state.step.anchor.last.whole == (2, -2)


def test_latest_frame_stays_consistent_after_window_eviction(rolling_bookkeeping):
    series = rolling_bookkeeping.begin(1)
    assert series.latest is None
    first_row = series.add(rolling_bookkeeping.Source(0))
    first = series.latest
    assert first.name == first_row["Frame"]
    assert first.index == 0 and first.displacement_A == (0.0, 0.0)

    for index in range(1, 4):
        series.add(rolling_bookkeeping.Source(index))
    latest = series.latest
    assert series.frame_indices == (3,) and latest.index == 3
    assert not latest.placed
    np.testing.assert_allclose(latest.displacement_A, (0.6, -0.6))
    assert latest.neural_phase is None and latest.complex_phase is None and latest.amplitude is None
    np.testing.assert_array_equal(latest.ssb_phase, series.phase[0])
    np.testing.assert_array_equal(latest.phase, series.images["phase"][0])
    np.testing.assert_array_equal(first.phase, np.arange(16).reshape(4, 4) + 10)


def test_rolling_report_waits_for_the_completed_frame(rolling_bookkeeping):
    """Reading a live report during publication waits for its matching absolute frame index."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    setup = rolling_bookkeeping
    result = setup.begin(2)
    published, resume, reading, finished = (Event() for _ in range(4))

    class PausedCount(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "count" and value == 1:
                published.set()
                if not resume.wait(2):
                    raise TimeoutError("the report test did not resume frame publication")
            super().__setattr__(name, value)

    result._state = PausedCount(**vars(result._state))

    def read():
        reading.set()
        report, indices = result.report(), result.frame_indices
        finished.set()
        return report, indices

    with ThreadPoolExecutor(max_workers=2) as workers:
        added = workers.submit(result.add, setup.Source(0))
        try:
            assert published.wait(2)
            observed = workers.submit(read)
            assert reading.wait(2)
            assert not finished.wait(0.05)
        finally:
            resume.set()
        assert added.result(timeout=2)["Frame"] == "repeated label"
        report, indices = observed.result(timeout=2)
    assert tuple(report.index) == indices == (0,) and result.total_frames == 1
    assert FakeSession.open == 0


def test_registration_without_overlap_explains_recovery_before_computing():
    """A separated field fails before attempting an FFT on an empty crop."""
    from quantem.gpu.direct_ptycho.stream import _place

    anchor = SimpleNamespace(last=SimpleNamespace(whole=(0, 0), piece=np.zeros((4, 4))))
    with pytest.raises(RuntimeError, match="No usable overlap.*reduce the interval"):
        _place(anchor, np.zeros((4, 4)), None, (0.2, 0.2), 30, (20, 20), accumulate=False)


@pytest.mark.usefixtures("cuda_runtime")
@pytest.mark.parametrize("max_frames", [1, 3])
def test_rolling_cuda_results_match_full_history_and_exact_window_combination(monkeypatch, tmp_path, max_frames):
    """Real CUDA registration keeps the same global drift through eviction and a rejected run."""
    import cupy

    from quantem.gpu.direct_ptycho import stream as stream_module

    sources, fit = fake_stream(monkeypatch, tmp_path)
    fit.aberrations = {"C10_nm": -12.5, "C12_nm": 1.25, "phi12_deg": 17., "C30_nm": 650.}
    fit.sample_tilt_mrad, fit.depth_spread_nm = (1.2, -0.7), 3.5
    options = dict(refit="none", method=("phase_of_mean", "phase", "complex"))
    full, _ = streamed([], fit, None, **options)
    window, _ = streamed([], fit, None, max_frames=max_frames, **options)
    for index, source in enumerate(sources * 3):
        # A deliberate confidence control expires every accepted frame, then permits recovery.
        monkeypatch.setattr(stream_module, "CONFIDENCE", float("inf") if 3 <= index <= 5 else 0.04)
        full.add(source)
        window.add(source)
        retained = tuple(range(max(0, index + 1 - max_frames), index + 1))
        assert window.frame_indices == retained
        assert tuple(window.report().index) == retained and len(window._state.step.placed) <= 2
        np.testing.assert_array_equal(window.shifts_A, full.shifts_A[list(retained)])
        assert window.confidence == tuple(full.confidence[i] for i in retained)
        assert window.placed == tuple(full.placed[i] for i in retained)
        for key in window.images:
            cupy.testing.assert_array_equal(window.images[key], full.images[key][list(retained)])
        cupy.testing.assert_array_equal(window._state.step.anchor.last.piece, full._state.step.canvases["phase_of_mean"].last.piece)
        assert window._state.step.anchor.last.whole == full._state.step.canvases["phase_of_mean"].last.whole
        for key, stack in full.images.items():
            # Start in scan-zero coordinates, add only retained contributions, then crop empty margins.
            canvas = stream_module._start(cupy.zeros_like(stack[0]))
            canvas.cover.fill(0)
            positions = dict(full._state.step.placed)
            for frame in retained:
                if full.placed[frame]:
                    if frame == 0:
                        canvas.total += stack[0]
                        canvas.cover += 1
                    else:
                        stream_module._add(canvas, stack[frame], positions[frame])
            rows = cupy.flatnonzero(canvas.cover.any(axis=1)).get()
            cols = cupy.flatnonzero(canvas.cover.any(axis=0)).get()
            if len(rows):
                region = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))
                expected, cover = stream_module._combined(canvas)[region], canvas.cover[region]
            else:
                expected = cupy.zeros_like(stack[0])
                cover = cupy.zeros_like(stack[0])
            cupy.testing.assert_array_equal(window.combined[key], expected)
            cupy.testing.assert_array_equal(window.cover, cover)
        if index == 5:
            assert not any(window.placed)
            images, labels, _, _ = window._view(list(window.images))
            assert all(np.isfinite(image).all() for image in images)
            assert "rejected, not combined" in labels[0]
        assert len(window._state.stacks["phase_of_mean"]) <= max_frames and FakeSession.open == 0
    assert window.total_frames == 15 and all(row["evaluations"] == 0 for row in window.refits)


@pytest.mark.usefixtures("cuda_runtime")
def test_rolling_cuda_resume_keeps_global_record_indices(monkeypatch, tmp_path):
    """A bounded defocus series resumes its complete disk record chain without retaining old images."""
    import json

    sources, fit = fake_stream(monkeypatch, tmp_path)
    first, _ = streamed(sources, fit, tmp_path / "records", max_frames=2)
    resumed, calls = streamed(sources, fit, tmp_path / "records", max_frames=2)
    assert calls == 0 and resumed.frame_indices == (3, 4) and resumed.total_frames == 5
    assert all(row["resumed"] for row in resumed.refits)
    assert_same_stream(first, resumed)
    records = [json.loads((tmp_path / "records" / "stream" / f"frame_{index}.json").read_text()) for index in range(5)]
    assert [record["identity"]["index"] for record in records] == list(range(5))


@pytest.mark.usefixtures("cuda_runtime")
def test_latest_keeps_native_detector_counts_separate_from_registered_phase(monkeypatch, tmp_path):
    import cupy

    from quantem.gpu.direct_ptycho import stream

    sources, fit = fake_stream(monkeypatch, tmp_path)
    counts = {p.name: cupy.full((256, 256), i + 2, dtype=cupy.uint32) for i, p in enumerate(sources)}

    def measure(session, scan, bounds):
        assert bounds == (32, 45)
        return counts[scan.name]

    monkeypatch.setattr(stream, "annular_counts", measure)
    series, _ = streamed([], fit, None, method="phase", max_frames=1, annulus_mrad=(32, 45), phase_damp=0.07)
    assert series.latest is None
    series.add(sources[0])
    first = series.latest
    series.add(sources[1])
    current = series.latest
    assert first.annulus_mrad == current.annulus_mrad == (32, 45)
    assert first.index == 0 and current.index == 1 and current.aberrations == series.fits[0].aberrations
    current.aberrations.clear()
    assert series.latest.aberrations == series.fits[0].aberrations != {}
    assert current.phase_damp == 0.07 and current.neural_phase is None
    cupy.testing.assert_array_equal(first.annular_counts, counts[sources[0].name])
    cupy.testing.assert_array_equal(current.annular_counts, counts[sources[1].name])
    assert "annular_counts" not in series.combined
