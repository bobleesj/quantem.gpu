"""Streaming refits only defocus and astigmatism and holds every other coordinate."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantem.gpu.direct_ptycho.stream import _refit


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


def test_align_replaces_the_live_shifts_with_the_pairwise_drift():
    """The final pass runs the pairwise drift on the phase stack, takes its displacement (Å) and
    rebuilds the combined image from it."""
    import cupy

    from quantem.gpu.drift import shifted
    from quantem.gpu.direct_ptycho.stream import SSBStream, _add, _combined, _start

    rows, sampling = 256, (0.2, 0.2)
    specimen = lattice_image(rows, sampling[0], seed=3)
    truth = np.array([(0, 0), (2.3, 1.1), (4.45, 2.6), (6.1, 4.35)])
    generator = cupy.random.default_rng(4)
    phase = cupy.stack([shifted(specimen, shift) + 0.3 * generator.standard_normal((rows, rows)) for shift in truth]).astype(cupy.float32)
    series = SSBStream(frames=tuple("abcd"), phase=phase, images={"ssb": phase}, methods=("ssb",), aberrations=(),
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
    cupy.testing.assert_array_equal(series.combined["ssb"], _combined(canvas))


class FrameObjective(PeakedObjective):
    """One frame's model agreement: its low-order peak is the frame's own, the rest the reference."""

    calls = 0

    def values(self, points, stride=1):
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

    def _resolve_upsampling_factor(self, upsampling_factor):
        from quantem.gpu.direct_ptycho.workflow import SSB

        return SSB._resolve_upsampling_factor(self, upsampling_factor)

    def run(self, chosen, *, method, **options):
        import cupy

        strengths = {name: value for name, value in options.items() if name not in (
            "upsampling_factor", "compute_loss", "force", "verbose")}
        FakeSession.runs.append(SimpleNamespace(frame=self.data.name, method=method, chosen=chosen,
                                                optics=dict(chosen.aberrations), strengths=strengths,
                                                factor=options["upsampling_factor"]))
        if method == "ssb":
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
    monkeypatch.setattr(stream_module, "_objective", lambda session, gradients=False: FrameObjective(
        peaks[loaded["name"]] + np.r_[1e-3 * loaded["bytes"], np.zeros(27)]))
    monkeypatch.setattr(stream_module, "public_params", lambda objective, z, order: (
        {"C10_nm": z[0], "C12_nm": z[1], "phi12_deg": z[2]}, (0.0, 0.0), 1.0))
    fit = SimpleNamespace(optimizer={"coordinates": reference.tolist(), "bounds": [(-3., 3.)] * 28},
                          scan_detector_rotation_deg=176.3, scan_detector_rotation_flipped=False,
                          phase_limits=(-1.0, 1.0), metadata={})
    return sources, fit


def streamed(sources, fit, save_dir, **options):
    from quantem.gpu.direct_ptycho.stream import stream

    FrameObjective.calls = 0
    FakeSession.runs, FakeSession.open = [], 0
    series = stream(FakeSession(), sources, fit, save_dir=save_dir, show=False, verbose=False, **options)
    return series, FrameObjective.calls


def assert_same_stream(left, right):
    import cupy

    assert left.aberrations == right.aberrations
    assert [{name: refit[name] for name in ("evaluations", "stop")} for refit in left.refits] == \
        [{name: refit[name] for name in ("evaluations", "stop")} for refit in right.refits]
    np.testing.assert_array_equal(left.shifts_A, right.shifts_A)
    assert left.confidence == right.confidence
    cupy.testing.assert_array_equal(left.phase, right.phase)
    assert left.combined.keys() == right.combined.keys()
    for key in left.combined:
        cupy.testing.assert_array_equal(left.combined[key], right.combined[key])


def test_stream_resumes_from_saved_records_without_refitting(monkeypatch, tmp_path):
    """A second call with the same fit and sources refits nothing: every frame is reconstructed from
    its saved optics and placed at its saved shift, and the series equals the first call exactly."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    first, first_calls = streamed(sources, fit, tmp_path / "results")
    assert first_calls > 0
    assert sorted(path.name for path in (tmp_path / "results" / "stream").iterdir()) == \
        [f"frame_{index}.json" for index in range(5)]
    again, again_calls = streamed(sources, fit, tmp_path / "results")
    assert again_calls == 0
    assert all(refit["resumed"] for refit in again.refits)
    assert_same_stream(first, again)


def test_an_interrupted_stream_continues_as_if_never_stopped(monkeypatch, tmp_path):
    """Streaming the first three frames, then all five, equals one uninterrupted run: the first three
    come from their records and the fourth refits from the third's saved low-order values."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    whole, _ = streamed(sources, fit, None)
    streamed(sources[:3], fit, tmp_path / "results")
    resumed, _ = streamed(sources, fit, tmp_path / "results")
    assert [refit["resumed"] for refit in resumed.refits] == [True] * 3 + [False] * 2
    assert_same_stream(whole, resumed)


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


def test_stream_reconstructs_each_requested_method_at_the_frames_optics(monkeypatch, tmp_path):
    """method=("phase", "complex") runs SSB (for registration) and both regularized methods once per
    frame, each at that frame's refit optics with the strengths given; images and combined hold the
    phase of each method and the complex method's amplitude, every image moved by the same shifts."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method=("phase", "complex"), phase_regularization=0.07,
                         amplitude_regularization=0.2)
    assert list(series.images) == ["phase", "complex", "amplitude"]
    assert list(series.combined) == ["phase", "complex", "amplitude"]
    assert series.methods == ("phase", "complex")
    assert [(run.frame, run.method) for run in FakeSession.runs] == [
        (source.name, method) for source in sources for method in ("ssb", "phase", "complex")]
    for run in FakeSession.runs:
        index = [source.name for source in sources].index(run.frame)
        assert run.chosen is series.fits[index]
        assert run.optics == series.aberrations[index] == series.fits[index].aberrations
        assert run.strengths == ({} if run.method == "ssb" else
                                 {"phase_regularization": 0.07, "amplitude_regularization": 0.2})
    assert "evaluations" not in series.aberrations[0] and set(series.refits[0]) == {"evaluations", "stop", "resumed"}
    assert series.phase.shape == series.images["phase"].shape == (5, 256, 256)
    # the regularized images are the fake's noise, placed with the shifts measured on the SSB phase
    for key in series.images:
        live = placed_by_hand(series.images[key], np.array(series.shifts_A), series.placed)
        cupy.testing.assert_allclose(series.combined[key], live, atol=1e-6)
    table = series.report()
    assert {"SSB (s)", "Phase-only (s)", "Amplitude + phase (s)", "Near threshold"} <= set(table.columns)


def test_a_capped_auto_factor_is_announced_once_per_stream(monkeypatch, tmp_path, capsys):
    """At a 1 A step (300 kV, 30 mrad) the scan sampling supports 8x: auto is resolved once to 4x, one line
    says so, and every frame runs at 4x; an explicit 8 runs at 8x without the line."""
    from quantem.gpu.direct_ptycho.stream import stream

    sources, fit = fake_stream(monkeypatch, tmp_path)
    monkeypatch.setattr(FakeSession, "scan_sampling_A", 1.0)
    for requested, factor, notice in (("auto", 4, "SSB: auto upsampling capped at 4× (the scan sampling supports "
                                                  "8×); pass upsampling_factor=8 for the full factor.\n"),
                                      (8, 8, "")):
        FakeSession.runs, FakeSession.open = [], 0
        capsys.readouterr()
        stream(FakeSession(), sources, fit, upsampling_factor=requested, show=False, verbose=False)
        assert capsys.readouterr().out == notice
        assert {run.factor for run in FakeSession.runs} == {factor}


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
    assert list(phase.combined) == ["phase"] and list(ssb.combined) == ["ssb"]
    cupy.testing.assert_array_equal(ssb.images["ssb"], ssb.phase)


def test_reconstruct_rebuilds_a_method_from_the_saved_optics_and_shifts(monkeypatch, tmp_path):
    """After a stream, reconstruct("phase", ...) reloads each frame, runs the phase method once per
    frame at fits[k] with the new strength, and places it at shifts_A; nothing is refit."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None)
    FakeSession.runs, FrameObjective.calls = [], 0
    combined = series.reconstruct("phase", phase_regularization=0.05)
    assert FrameObjective.calls == 0
    assert [(run.frame, run.method, run.strengths) for run in FakeSession.runs] == [
        (source.name, "phase", {"phase_regularization": 0.05}) for source in sources]
    assert all(run.chosen is chosen for run, chosen in zip(FakeSession.runs, series.fits, strict=True))
    assert list(combined) == ["phase"] and series.methods == ("ssb", "phase")
    cupy.testing.assert_array_equal(combined["phase"], series.combined["phase"])
    cupy.testing.assert_array_equal(series.combined["phase"],
                                    placed_by_hand(series.images["phase"], series.shifts_A, series.placed))


def test_resume_with_methods_refits_nothing(monkeypatch, tmp_path):
    """Records hold optics and shifts, not images: a second call asking for other methods resumes
    every frame from its record with no refit and reconstructs the methods it asks for."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    streamed(sources, fit, tmp_path / "results")
    again, calls = streamed(sources, fit, tmp_path / "results", method=("phase", "complex"))
    assert calls == 0 and all(refit["resumed"] for refit in again.refits)
    assert [(run.frame, run.method) for run in FakeSession.runs] == [
        (source.name, method) for source in sources for method in ("ssb", "phase", "complex")]
    whole, _ = streamed(sources, fit, None, method=("phase", "complex"))
    assert_same_stream(whole, again)


def test_near_threshold_flags_confidence_just_above_the_cut(monkeypatch, tmp_path):
    """Near threshold marks a frame whose confidence lies in [CONFIDENCE, 1.1 CONFIDENCE): placed,
    but close enough to the cut to deserve a look."""
    from quantem.gpu.direct_ptycho.stream import CONFIDENCE

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None)
    series.confidence = (float("inf"), CONFIDENCE, 1.05 * CONFIDENCE, 1.1 * CONFIDENCE, 0.5 * CONFIDENCE)
    assert series.report()["Near threshold"].tolist() == [False, True, True, False, False]


def test_show_draws_each_combined_image_and_the_drift(monkeypatch, tmp_path):
    """Static: one image panel per combined image (phase, complex phase, amplitude) over the drift
    strip, the amplitude on its own limits; interactive: a Show2D of the same images by name."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method=("phase", "complex"))
    figure = series.show()
    images = [axis for axis in figure.axes if axis.images]
    assert [axis.get_title().split("\n")[0] for axis in images] == [
        "Phase-only (phase)", "Amplitude + phase (phase)", "Amplitude + phase (amplitude)"]
    assert images[0].images[0].get_clim() == fit.phase_limits != images[2].images[0].get_clim()
    assert len(series.show("phase").axes) == len(figure.axes) - 4
    viewer = series.show(interactive=True)
    assert type(viewer).__name__ == "Show2D" and viewer.n_images == 3


def test_every_session_is_closed_before_the_next_frame(monkeypatch, tmp_path):
    """Each frame's session (and reconstruct's) is closed once used, so no frame's GPU state waits for
    the cycle collector; none is left open or alive after the series."""
    import gc

    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method="phase")
    assert FakeSession.open == 0
    series.reconstruct("complex")
    assert FakeSession.open == 0
    assert not [item for item in gc.get_objects() if type(item) is FakeSession and item.data is not None]


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


def test_reconstruct_after_align_places_every_frame_at_the_pairwise_shifts(monkeypatch, tmp_path):
    """After align, reconstruct places every frame, including one the live pass left out, at
    align's shifts_A, and report() shows the pairwise σ in place of the live threshold flag."""
    import cupy

    sources, fit = fake_stream(monkeypatch, tmp_path, noise_frames=(3,))
    series, _ = streamed(sources, fit, None)
    found = series.align()
    combined = series.reconstruct("phase")
    assert series.placed == (True,) * 5
    np.testing.assert_array_equal(series.shifts_A, found.displacement_A)
    cupy.testing.assert_array_equal(combined["phase"], placed_by_hand(series.images["phase"], found.displacement_A,
                                                                      series.placed))
    table = series.report()
    assert "Near threshold" not in table.columns and "Live confidence" in table.columns
    np.testing.assert_array_equal(table[["σ row (Å)", "σ col (Å)"]].to_numpy(), found.sigma_A)


def test_reconstruct_ssb_keeps_phase_the_ssb_stack_and_methods_are_unique(monkeypatch, tmp_path):
    """Asking twice for a method runs it once a frame; reconstructing SSB replaces phase too, so the
    registration stack and images["ssb"] stay one array."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    series, _ = streamed(sources, fit, None, method=("ssb", "phase", "ssb", "phase"))
    assert series.methods == ("ssb", "phase") and len(FakeSession.runs) == 2 * len(sources)
    series.reconstruct("ssb")
    assert series.phase is series.images["ssb"]


def test_flagged_pixel_handling_is_announced_once_for_the_series(monkeypatch, tmp_path, capsys):
    """Each frame's load is quiet, so a long series prints no line per frame; what the loads did with
    the detector's flagged pixels still changes the data, so it is announced once, by outcome."""
    from quantem.gpu.resident.hot_pixels import hot_pixel_record
    from quantem.gpu.direct_ptycho import stream as stream_module
    from quantem.gpu.direct_ptycho.stream import stream

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
    stream(FakeSession(), sources, fit, show=False, verbose=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Stream:")]
    assert quiet == [False] * len(sources)
    assert lines == [f"Stream: {len(sources)} of {len(sources)} frames, 2 bad pixels replaced by neighbor median "
                     '(hot_pixel_correction="none" keeps the recorded values).']
