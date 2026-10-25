"""Streaming refits only defocus and astigmatism and holds every other coordinate."""

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
