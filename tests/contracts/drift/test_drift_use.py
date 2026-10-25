"""What a microscopist calls: drift on any form of a phase series (stack, NumPy, list, stream, SSB results) gives the
same Drift, and corrected moves every frame back onto frame 0 over the region all frames share, with the sign of
displacement_A (I_f(r) = I_0(r - d_f))."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

cupy = pytest.importorskip("cupy")

from quantem.gpu.drift import Drift, drift, running

from .test_drift_parity import DRIFT_DISPLACEMENT, DRIFT_PX, SAMPLING_A, specimen

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="registration runs on the GPU")

# whole pixels, both signs on both axes, so the Fourier shift is an exact circular one and the region is exact too
WHOLE_PX = [(0, 0), (7, -4), (-5, 9), (12, 3)]


@pytest.fixture(scope="module")
def images():
    return torch.stack([specimen(shift, seed=index) for index, shift in enumerate(DRIFT_PX)])


def test_every_input_form_gives_the_same_drift(images):
    stream = SimpleNamespace(phase=images, frames=[f"frame {index}" for index in range(len(images))], object_sampling_A=SAMPLING_A)
    results = [SimpleNamespace(phase=image, object_sampling_A=SAMPLING_A) for image in images]
    for found in (drift(images.cpu().numpy(), sampling_A=SAMPLING_A[0]), drift(cupy.asarray(images), sampling_A=SAMPLING_A),
                  drift(list(images), sampling_A=SAMPLING_A), drift(stream), drift(results)):
        np.testing.assert_array_equal(found.displacement_A, DRIFT_DISPLACEMENT)
    assert drift(stream).frames == stream.frames
    np.testing.assert_array_equal(running(stream).displacement_A, running(images, sampling_A=SAMPLING_A).displacement_A)


def test_corrected_puts_every_frame_on_frame_0_over_the_shared_region():
    """Noise-free frames moved by whole pixels: moved back, each equals frame 0 over the region every frame sees, which
    is frame 0's rows 5 to 384 - 12 and columns 4 to 384 - 9."""
    frames = torch.stack([specimen(shift, noise=0.0) for shift in WHOLE_PX])
    displacement_A = np.array(WHOLE_PX, dtype=float) * SAMPLING_A
    found = Drift(list(range(4)), displacement_A, np.zeros_like(displacement_A), [], np.zeros((0, 2)), sampling_A=SAMPLING_A)
    aligned, mean = found.corrected(frames)
    assert aligned.shape == (4, 384 - 12 - 5, 384 - 9 - 4)
    torch.testing.assert_close(aligned, frames[0, 5:372, 4:375].expand_as(aligned), rtol=0, atol=2e-5)
    torch.testing.assert_close(mean, aligned.mean(0))


def test_corrected_moves_frames_by_their_own_pixel_size():
    """A drift measured at 0.25 Å a pixel applied to frames on a rectangular grid at (0.1, 0.2) Å, moved by fractional,
    signed shifts: each frame is a band-limited field (whole-period Fourier modes of its box), so the Fourier shift is
    exact and every corrected frame equals frame 0 over the region all of them see."""
    shape, sampling_A = (200, 300), (0.1, 0.2)
    displacement_A = np.array([(0.0, 0.0), (0.34, -1.34), (-0.22, 1.1)])
    generator = np.random.default_rng(5)
    modes = [(generator.integers(-12, 13), generator.integers(-12, 13), generator.uniform(0, 2 * np.pi)) for _ in range(20)]
    rows = torch.arange(shape[0], dtype=torch.float64, device="cuda")[:, None]
    cols = torch.arange(shape[1], dtype=torch.float64, device="cuda")[None, :]

    def frame(shift_px):
        return sum(torch.cos(2 * np.pi * (k_row * (rows - shift_px[0]) / shape[0] + k_col * (cols - shift_px[1]) / shape[1]) + phase) for k_row, k_col, phase in modes)

    frames = torch.stack([frame(shift / np.asarray(sampling_A)) for shift in displacement_A]).float()
    found = Drift([0, 1, 2], displacement_A, np.zeros_like(displacement_A), [], np.zeros((0, 2)), sampling_A=(0.25, 0.25))
    aligned, _ = found.corrected(frames, sampling_A=sampling_A)
    assert aligned.shape == (3, 200 - 4 - 3, 300 - 6 - 7)  # rows 3:196 (ceil 2.2, floor 200 - 3.4), cols 7:294
    torch.testing.assert_close(aligned, frames[0, 3:196, 7:294].expand_as(aligned), rtol=0, atol=2e-5)
