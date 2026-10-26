"""Fifth-order MPS agreement uses all terms and preserves batched candidates."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def objective():
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB
    from quantem.gpu.direct_ptycho.mps.fifth_order import Objective

    counts = np.random.default_rng(617).poisson(
        10, size=(128, 128, 16, 16)
    ).astype(np.uint16)
    with SSB(
        counts, backend="mps", voltage_kV=300.0, semiangle_mrad=30.0,
        scan_sampling_A=0.373, detector_sampling_mrad=4.0,
        scan_detector_rotation_deg=17.3,
        bright_field_center_px=(8.0, 8.0), bright_field_radius_px=7.0,
    ) as ssb:
        backend = ssb._backend()
        backend.cache_rotation(np.deg2rad(17.3))
        yield Objective(backend._prepared)


def test_all_fifth_order_coordinates_change_agreement(objective):
    """Each optical, tilt and depth coordinate participates in the model."""
    start = np.random.default_rng(619).uniform(-0.25, 0.25, 28)
    start[5] = 1.2
    candidates = np.tile(start, (29, 1))
    candidates[1:] += np.eye(28) * 0.15
    values = objective.evaluate(candidates, 5)
    assert np.isfinite(values).all()
    assert np.all(values > 0)
    assert np.all(values[1:] != values[0])


def test_batched_candidates_preserve_scores(objective):
    """Changing candidate batching retains every candidate and every BF pixel."""
    candidates = np.random.default_rng(619).uniform(-0.25, 0.25, (9, 28))
    candidates[:, 5] = 1.2
    batched = objective.evaluate(candidates, 5)
    single = np.array([objective.evaluate(z, 5)[0] for z in candidates])
    np.testing.assert_array_equal(batched, single)
    from quantem.gpu.direct_ptycho.mps.fifth_order import Objective

    small = Objective(objective.p, partial_bytes=1)
    np.testing.assert_array_equal(small.evaluate(candidates, 5), batched)


def test_matches_existing_higher_order_model(objective):
    """Compare the new GPU reduction with the established Metal optics model."""
    from quantem.gpu.direct_ptycho.higher_order import ORDERS
    from quantem.gpu.direct_ptycho.mps.higher_order_agreement import agreement_batch

    candidates = np.random.default_rng(619).uniform(-0.25, 0.25, (9, 28))
    candidates[:, 5] = 1.2
    primary, packed = objective.encode(candidates, 5)
    rows = np.empty((len(primary), 31))
    rows[:, 0] = primary[:, 5]
    rows[:, 1:3] = primary[:, 3:5]
    divisors = np.repeat([n + 1 for n, _ in ORDERS], 2)
    rows[:, 3:] = packed.reshape(len(primary), 28) * divisors
    expected = agreement_batch(objective.p, rows)
    np.testing.assert_allclose(objective.evaluate(candidates, 5), expected, rtol=1e-5)


def test_zero_higher_orders_match_primary_model(objective):
    """The fifth-order polynomial includes the existing primary model exactly."""
    from quantem.gpu.direct_ptycho.mps.thick_sample import thick_fit_batch

    z = np.array([[-0.4, 0.1, -0.08, 0.4, -0.6, 1.2]])
    primary, _ = objective.encode(z, 1)
    expected = thick_fit_batch(objective.p, primary)
    np.testing.assert_allclose(objective.evaluate(z, 1), expected, rtol=3e-6)
