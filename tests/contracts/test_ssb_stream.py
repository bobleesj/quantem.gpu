"""Streaming refits only defocus and astigmatism and holds every other coordinate."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantem.gpu.ssb.stream import _refit


class PeakedObjective:
    """Model agreement exp(-|z - peak|^2): its maximum is known exactly."""

    def __init__(self, peak):
        self.peak = peak
        self.seen = []

    def evaluate(self, points, order):
        self.seen.append(np.array(points))
        return np.exp(-np.sum((points - self.peak) ** 2, axis=1))


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
    from quantem.gpu.ssb.stream import (
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
    from quantem.gpu.ssb.stream import SSBStream, _add, _combined, _start

    rows, sampling = 256, (0.2, 0.2)
    specimen = lattice_image(rows, sampling[0], seed=3)
    truth = np.array([(0, 0), (2.3, 1.1), (4.45, 2.6), (6.1, 4.35)])
    generator = cupy.random.default_rng(4)
    phase = cupy.stack([shifted(specimen, shift) + 0.3 * generator.standard_normal((rows, rows)) for shift in truth]).astype(cupy.float32)
    series = SSBStream(frames=tuple("abcd"), phase=phase, aberrations=(), timings=(), reference=None,
                       object_sampling_A=sampling, shifts_A=np.zeros((4, 2)), confidence=(), combined=None)
    found = series.align()
    # Its pairs span up to 6 px of a 256 px field, where register_spectra's Hann-window correction
    # leaves 0.04 px on this specimen (the live pass registers only the residual past its guess).
    np.testing.assert_allclose(series.shifts_A / sampling[0], truth, atol=0.05)
    np.testing.assert_array_equal(series.shifts_A, found.displacement_A)
    canvas = _start(phase[0])
    for frame, shift in zip(phase[1:], found.displacement_A[1:] / sampling[0]):
        _add(canvas, frame, tuple(shift))
    cupy.testing.assert_array_equal(series.combined, _combined(canvas))


class FrameObjective(PeakedObjective):
    """One frame's model agreement: its low-order peak is the frame's own, the rest the reference."""

    calls = 0

    def evaluate(self, points, order):
        FrameObjective.calls += len(points)
        return super().evaluate(points, order)


class FakeSession:
    """Stands in for DirectPtycho on one loaded frame: run returns the frame's lattice phase."""

    backend, voltage_kV, semiangle_mrad, scan_sampling_A = "cuda", 300.0, 30.0, 0.2
    detector_sampling_mrad, bright_field_center_px, bright_field_radius_px = 0.5, (96.0, 96.0), 40.0

    def __init__(self, data=None, **settings):
        self.data = data

    def run(self, chosen, **options):
        return SimpleNamespace(phase=self.data.phase, object_sampling_A=(0.2, 0.2))


def fake_stream(monkeypatch, tmp_path, rows=256):
    """Five frames of one specimen drifting a known amount, each with its own defocus peak, written
    as small files so their identities are real; returns the sources and the reference fit."""
    import cupy

    from quantem.gpu.drift import shifted
    from quantem.gpu.ssb import stream as stream_module

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
        phases[source.name] = (shifted(specimen, shift) + 0.3 * generator.standard_normal((rows, rows))).astype(cupy.float32)
        peak = reference.copy()
        peak[:3] = (.1 * index, -.05 * index, .02 * index)
        peaks[source.name] = peak
    loaded = {}

    def load(source, device):
        # the file's content moves the frame's defocus peak, so a rewritten frame refits elsewhere
        loaded["name"], loaded["bytes"] = Path(source).name, len(Path(source).read_bytes())
        return SimpleNamespace(phase=phases[Path(source).name], metadata={})

    monkeypatch.setattr(stream_module.io, "load", load)
    monkeypatch.setattr(stream_module, "_objective", lambda session: FrameObjective(
        peaks[loaded["name"]] + np.r_[1e-3 * loaded["bytes"], np.zeros(27)]))
    monkeypatch.setattr(stream_module, "public_params", lambda objective, z, order: (
        {"C10_nm": z[0], "C12_nm": z[1], "phi12_deg": z[2]}, (0.0, 0.0), 1.0))
    fit = SimpleNamespace(optimizer={"coordinates": reference.tolist(), "bounds": [(-3., 3.)] * 28},
                          scan_detector_rotation_deg=176.3, scan_detector_rotation_flipped=False,
                          phase_limits=(-1.0, 1.0), metadata={})
    return sources, fit


def streamed(sources, fit, save_dir):
    from quantem.gpu.ssb.stream import stream

    FrameObjective.calls = 0
    series = stream(FakeSession(), sources, fit, save_dir=save_dir, show=False, verbose=False)
    return series, FrameObjective.calls


def assert_same_stream(left, right):
    import cupy

    keep = ("C10_nm", "C12_nm", "phi12_deg", "evaluations", "stop")
    assert [{name: optics[name] for name in keep} for optics in left.aberrations] == \
        [{name: optics[name] for name in keep} for optics in right.aberrations]
    np.testing.assert_array_equal(left.shifts_A, right.shifts_A)
    assert left.confidence == right.confidence
    cupy.testing.assert_array_equal(left.phase, right.phase)
    cupy.testing.assert_array_equal(left.combined, right.combined)


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
    assert all(optics["resumed"] for optics in again.aberrations)
    assert_same_stream(first, again)


def test_an_interrupted_stream_continues_as_if_never_stopped(monkeypatch, tmp_path):
    """Streaming the first three frames, then all five, equals one uninterrupted run: the first three
    come from their records and the fourth refits from the third's saved low-order values."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    whole, _ = streamed(sources, fit, None)
    streamed(sources[:3], fit, tmp_path / "results")
    resumed, _ = streamed(sources, fit, tmp_path / "results")
    assert [optics["resumed"] for optics in resumed.aberrations] == [True] * 3 + [False] * 2
    assert_same_stream(whole, resumed)


def test_records_of_another_source_or_fit_are_ignored(monkeypatch, tmp_path):
    """A record keys on the frame's file content and on the reference fit: a rewritten frame and every
    frame after it are fitted again, and a different reference fit refits them all."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    streamed(sources, fit, tmp_path / "results")
    sources[2].write_bytes(b"another acquisition")
    changed, _ = streamed(sources, fit, tmp_path / "results")
    assert [optics["resumed"] for optics in changed.aberrations] == [True, True, False, False, False]
    other = SimpleNamespace(**{**vars(fit), "optimizer": {**fit.optimizer, "coordinates": (np.array(fit.optimizer["coordinates"]) + 1e-9).tolist()}})
    refit, calls = streamed(sources, other, tmp_path / "results")
    assert calls > 0 and not any(optics["resumed"] for optics in refit.aberrations)


def test_a_rewritten_frame_invalidates_every_later_record_even_if_interrupted(monkeypatch, tmp_path):
    """Frame 2 is rewritten and recomputed, then the run stops before frames 3 and 4 are: their old
    records came from the old frame 2, so the next call refits them and equals an uninterrupted run."""
    sources, fit = fake_stream(monkeypatch, tmp_path)
    streamed(sources, fit, tmp_path / "results")
    sources[2].write_bytes(b"another acquisition")
    streamed(sources[:3], fit, tmp_path / "results")
    resumed, _ = streamed(sources, fit, tmp_path / "results")
    assert [optics["resumed"] for optics in resumed.aberrations] == [True, True, True, False, False]
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
    assert [optics["resumed"] for optics in again.aberrations] == [True, False, False, False, False]
    assert_same_stream(first, again)
