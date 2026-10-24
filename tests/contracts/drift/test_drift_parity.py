"""quantem.gpu.drift reproduces myptycho's translation drift exactly: register, drift and running on a simulated twisted
MoS2 bilayer moved by a drift longer than its lattice, against myptycho's outputs frozen below (myptycho ab62529a,
time/tests/drift_baseline.py, quantem.gpu a26c62fa for running's correlation). The series is rebuilt here exactly as
myptycho's test_drift.specimen builds it, so the comparison needs no stored images."""

import numpy as np
import pytest
import torch

from quantem.gpu.drift import drift, register, running

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="registration runs on the GPU")

SAMPLING_A = (0.25, 0.25)
SHAPE = (384, 384)
LATTICE_A = 3.16  # MoS2
TWIST_DEG = 15.6
DRIFT_PX = [(0.0, 0.0), (11.37, -6.82), (21.91, -15.24), (30.43, -27.58), (36.06, -33.17), (45.72, -36.91)]

REGISTER_SHIFT = np.array([11.37, -6.82])
DRIFT_DISPLACEMENT = np.array([[0.0, 0.0], [2.8418303571428547, -1.7049999999999983], [5.476793154761898, -3.809999999999996], [7.606376488095238, -6.894999999999999],
                               [9.014151785714281, -8.292499999999995], [11.429107142857141, -9.227499999999997]])
DRIFT_SIGMA = np.array([[0.0, 0.0], [0.0004777490967708798, 2.083928928551165e-15], [0.00047003050991585354, 2.0502606567669946e-15],
                        [0.00047003050991585354, 2.0502606567669946e-15], [0.0005463269935819401, 2.3830639012590325e-15], [0.0005925748241821015, 2.5847957155561735e-15]])
DRIFT_CLOSURE = np.array([[-8.881784197001252e-16, 0.0], [0.0, 8.881784197001252e-16], [-4.440892098500626e-16, 0.0], [0.0, 0.0]])
RUNNING_DISPLACEMENT = np.array([[-0.0, -0.0], [2.8422731372147894, -1.7051563430478751], [5.476878838052514, -3.809854983639326], [7.6063067253080305, -6.8949980023489275],
                                 [9.014555308691044, -8.291966936261034], [11.42892174289986, -9.227166755438077]])


def reciprocal(angle_deg):
    """The three first-order spots of a hexagonal lattice turned by angle_deg, and the three second-order."""
    g = 2 / (np.sqrt(3) * LATTICE_A)
    first = [g * np.array([np.sin(np.radians(angle_deg + 120 * k)), np.cos(np.radians(angle_deg + 120 * k))]) for k in range(3)]
    second = [np.sqrt(3) * g * np.array([np.sin(np.radians(angle_deg + 30 + 120 * k)), np.cos(np.radians(angle_deg + 30 + 120 * k))]) for k in range(3)]
    return first, second


def specimen(shift_px, *, strength=(1.0, 1.0), noise=0.3, seed=0):
    """The bilayer's phase seen with the specimen moved by shift_px: two hexagonal lattices 15.6° apart (cosines at their
    first and second orders), aperiodic blobs 4 Å wide at fixed spots of the specimen, and white noise; evaluated exactly
    at the moved positions, nothing interpolated."""
    rows = (torch.arange(SHAPE[0], dtype=torch.float64, device="cuda")[:, None] - shift_px[0]) * SAMPLING_A[0]
    cols = (torch.arange(SHAPE[1], dtype=torch.float64, device="cuda")[None, :] - shift_px[1]) * SAMPLING_A[1]
    image = torch.zeros(SHAPE, dtype=torch.float64, device="cuda")
    for amplitude, angle in zip(strength, (0.0, TWIST_DEG)):
        first, second = reciprocal(angle)
        for g in first:
            image += amplitude * torch.cos(2 * np.pi * (g[0] * rows + g[1] * cols))
        for g in second:
            image += 0.4 * amplitude * torch.cos(2 * np.pi * (g[0] * rows + g[1] * cols))
    blobs = np.random.default_rng(1).uniform(-20, SHAPE[0] * SAMPLING_A[0] + 20, (40, 2))
    for row, col in blobs:
        image += 2.0 * torch.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2 * 4.0**2))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image += noise * torch.randn(SHAPE, generator=generator, device="cuda", dtype=torch.float64)
    return image.float()


@pytest.fixture(scope="module")
def images():
    return torch.stack([specimen(shift, seed=index) for index, shift in enumerate(DRIFT_PX)])


def test_register_matches_myptycho(images):
    np.testing.assert_array_equal(np.array(register(images[0], images[1], sampling_A=SAMPLING_A).shift_px), REGISTER_SHIFT)


def test_drift_matches_myptycho(images):
    found = drift(images, sampling_A=SAMPLING_A)
    np.testing.assert_array_equal(found.displacement_A, DRIFT_DISPLACEMENT)
    np.testing.assert_array_equal(found.sigma_A, DRIFT_SIGMA)
    np.testing.assert_array_equal(found.closure_A, DRIFT_CLOSURE)
    assert len(found.layers) == 2


def test_running_matches_myptycho(images):
    np.testing.assert_array_equal(running(images, sampling_A=SAMPLING_A).displacement_A, RUNNING_DISPLACEMENT)
