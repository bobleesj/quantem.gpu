"""quantem.gpu.drift reproduces myptycho's translation drift exactly: register, drift and running on a simulated twisted
MoS2 bilayer moved by a drift longer than its lattice, against myptycho's outputs frozen below (baseline made at myptycho
ab62529a by time/tests/drift_baseline.py, the script added in myptycho 62e5e259; quantem.gpu a26c62fa for running's
correlation). The series is rebuilt here exactly as
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
# each pair's confidence (here its margin, the lesser of the two) gates live placement against an absolute threshold, so it is frozen too
DRIFT_CONFIDENCE = [0.2100536028544108, 0.20261001586914062, 0.1592871348063151, 0.19475007057189941, 0.2013526757558187, 0.17547059059143066,
                    0.1954201857248942, 0.22008093198140463, 0.18632729848225912, 0.20540467898050943, 0.21632631619771323, 0.21744362513224283]
DRIFT_PAIR_RMS_A = 0.00037866103124989387
RUNNING_DISPLACEMENT = np.array([[-0.0, -0.0], [2.8422731372147894, -1.7051563430478751], [5.476878838052514, -3.809854983639326], [7.6063067253080305, -6.8949980023489275],
                                 [9.014555308691044, -8.291966936261034], [11.42892174289986, -9.227166755438077]])


def reciprocal(angle_deg):
    """The three first-order spots of a hexagonal lattice turned by angle_deg, and the three second-order."""
    g = 2 / (np.sqrt(3) * LATTICE_A)
    first = [g * np.array([np.sin(np.radians(angle_deg + 120 * k)), np.cos(np.radians(angle_deg + 120 * k))]) for k in range(3)]
    second = [np.sqrt(3) * g * np.array([np.sin(np.radians(angle_deg + 30 + 120 * k)), np.cos(np.radians(angle_deg + 30 + 120 * k))]) for k in range(3)]
    return first, second


def specimen(shift_px, *, strength=(1.0, 1.0), noise=0.3, seed=0, shape=SHAPE, sampling_A=SAMPLING_A):
    """The bilayer's phase seen with the specimen moved by shift_px: two hexagonal lattices 15.6° apart (cosines at their
    first and second orders), aperiodic blobs 4 Å wide at fixed spots of the specimen, and white noise; evaluated exactly
    at the moved positions, nothing interpolated."""
    rows = (torch.arange(shape[0], dtype=torch.float64, device="cuda")[:, None] - shift_px[0]) * sampling_A[0]
    cols = (torch.arange(shape[1], dtype=torch.float64, device="cuda")[None, :] - shift_px[1]) * sampling_A[1]
    image = torch.zeros(shape, dtype=torch.float64, device="cuda")
    for amplitude, angle in zip(strength, (0.0, TWIST_DEG)):
        first, second = reciprocal(angle)
        for g in first:
            image += amplitude * torch.cos(2 * np.pi * (g[0] * rows + g[1] * cols))
        for g in second:
            image += 0.4 * amplitude * torch.cos(2 * np.pi * (g[0] * rows + g[1] * cols))
    blobs = np.random.default_rng(1).uniform(-20, shape[0] * sampling_A[0] + 20, (40, 2))
    for row, col in blobs:
        image += 2.0 * torch.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2 * 4.0**2))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image += noise * torch.randn(shape, generator=generator, device="cuda", dtype=torch.float64)
    return image.float()


@pytest.fixture(scope="module")
def images():
    return torch.stack([specimen(shift, seed=index) for index, shift in enumerate(DRIFT_PX)])


def test_register_matches_myptycho(images):
    np.testing.assert_array_equal(np.array(register(images[0], images[1], sampling_A=SAMPLING_A).shift_px), REGISTER_SHIFT)


def test_drift_matches_myptycho(images):
    found = drift(images, sampling_A=SAMPLING_A)
    np.testing.assert_array_equal(found.displacement_A, DRIFT_DISPLACEMENT)
    # σ col and closure are rounding-level (1e-15): a LAPACK build may move them, so they are compared to 1e-12 Å
    np.testing.assert_allclose(found.sigma_A, DRIFT_SIGMA, rtol=0, atol=1e-12)
    np.testing.assert_allclose(found.closure_A, DRIFT_CLOSURE, rtol=0, atol=1e-12)
    assert [pair["confidence"] for pair in found.pairs] == DRIFT_CONFIDENCE
    assert [pair["margin"] for pair in found.pairs] == DRIFT_CONFIDENCE
    assert all(pair["kept"] for pair in found.pairs)
    assert found.pair_rms_A == DRIFT_PAIR_RMS_A
    assert len(found.layers) == 2


def test_rectangular_anisotropic_series_finds_the_drift():
    """Every other case is square at one sampling, where swapping rows and columns or the two samplings passes; here the
    field is 96 x 64 Å at 0.25 x 0.2 Å a pixel, and drift and register find the imposed drift (myptycho 62e5e259 gives
    the same numbers bit for bit)."""
    shape, sampling_A = (384, 320), (0.25, 0.2)
    images = torch.stack([specimen(shift, seed=index, shape=shape, sampling_A=sampling_A) for index, shift in enumerate(DRIFT_PX)])
    np.testing.assert_allclose(drift(images, sampling_A=sampling_A).displacement_A, np.array(DRIFT_PX) * sampling_A, rtol=0, atol=2e-3)
    assert register(images[0], images[1], sampling_A=sampling_A).shift_px == DRIFT_PX[1]


def test_running_matches_myptycho(images):
    """running correlates by quantem.gpu.parallax's cross_correlation_shift_batch_cp, as the baseline did at quantem.gpu
    a26c62fa: a failure here after a change there is that change, not drift's."""
    np.testing.assert_array_equal(running(images, sampling_A=SAMPLING_A).displacement_A, RUNNING_DISPLACEMENT)
