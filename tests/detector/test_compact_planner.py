"""Exact detector planning on the retained 192 by 192 interaction tree."""

from pathlib import Path

import numpy as np
import pytest

from quantem.gpu._compact.planner import NativePlanner


@pytest.fixture
def metadata():
    with np.load(Path(__file__).with_name("data") / "compact_planner.npz") as saved:
        return {name: saved[name] for name in saved.files}


def _reference(metadata, current, previous):
    """Express the whole planner using NumPy reductions and broadcasting."""
    leaf_of = metadata["leaf_of"]
    weights = metadata["column_cost"]
    tile_cost = float(metadata["tile_cost"])
    seeds = [("zero", np.zeros_like(current)), ("total", metadata["valid"])]
    if previous is not None:
        seeds.append(("previous", previous))
    candidates = []
    for seed, baseline in seeds:
        difference = current - baseline
        totals = {
            sign: np.bincount(
                leaf_of, weights=weights * (difference == sign), minlength=1184
            )
            for sign in (-1, 0, 1)
        }
        leaf = np.zeros(1184, np.int8)
        leaf[(totals[1] > totals[0] + tile_cost) & (totals[-1] == 0)] = 1
        leaf[(totals[-1] > totals[0] + tile_cost) & (totals[1] == 0)] = -1
        positive = np.bincount(leaf_of[difference > 0], minlength=1184)
        negative = np.bincount(leaf_of[difference < 0], minlength=1184)
        leaf[(positive > 0) & (negative > 0)] = 0

        parent = leaf[:576].copy()
        children = leaf[576:].reshape(152, 4)
        parent[metadata["parents"]] = children[np.arange(152), metadata["omitted"]]
        child_difference = (
            np.take_along_axis(children, metadata["stored_positions"], axis=1)
            - parent[metadata["parents"], None]
        )
        cells = parent.reshape(6, 4, 6, 4).transpose(0, 2, 1, 3).reshape(36, 16)
        options = np.array([0, -1, 1], np.int8)
        costs = (cells[:, None, :] != options[None, :, None]).sum(axis=2)
        costs += (options != 0)[None, :]
        coarse = options[costs.argmin(axis=1)]
        correction = cells - coarse[:, None]
        fine = np.concatenate(
            (
                correction.reshape(6, 6, 4, 4).transpose(0, 2, 1, 3).ravel(),
                child_difference.ravel(),
            )
        )
        residual = (difference - leaf[leaf_of]) * metadata["valid"]
        cost = weights[residual != 0].sum() + tile_cost * (
            np.count_nonzero(fine) + np.count_nonzero(coarse)
        )
        candidates.append((seed, residual, fine, coarse, float(cost)))
    return min(candidates, key=lambda candidate: candidate[-1])


def _masks(valid):
    detector_row, detector_col = np.indices((192, 192), dtype=np.float64)
    poses = [
        (95.5, 95.5, 40.0, 80.0),
        (95.75, 96.25, 40.5, 80.125),
        (96.125, 96.375, 40.375, 81.25),
        (0.0, 0.0, 0.0, 30.5),
        (191.0, 191.0, 10.25, 95.0),
        (192.5, -1.25, 0.0, 80.5),
        (95.5, 95.5, 0.0, 0.0),
        (95.0, 95.0, 0.0, 0.0),
        (95.5, 95.5, 0.0, 300.0),
    ]
    masks = []
    for row, col, inner, outer in poses:
        squared_radius = (detector_row - row) ** 2 + (detector_col - col) ** 2
        mask = (squared_radius >= inner**2) & (squared_radius <= outer**2)
        masks.append(mask.ravel().astype(np.int8) * valid)
    return masks + masks[::-1]


def test_translated_fractional_detectors_match_numpy(metadata):
    """Fresh and incremental plans preserve empty, boundary, and large moves."""
    planner = NativePlanner.from_metadata(**metadata)
    previous = None
    for mask in _masks(metadata["valid"]):
        for baseline in (None, previous):
            actual = planner(mask, baseline)
            reference = _reference(metadata, mask, baseline)
            assert actual[0] == reference[0]
            for coefficients, expected in zip(actual[1:4], reference[1:4]):
                np.testing.assert_array_equal(coefficients, expected)
            # NumPy's pairwise sum differs from the serial C++ cost sum by
            # at most 4.55e-13 on these poses; scientific coefficients are exact.
            np.testing.assert_allclose(actual[-1], reference[-1], rtol=0, atol=1e-12)
        previous = mask


def test_planned_sums_preserve_every_integer_count(metadata):
    """Hierarchical coefficients reproduce direct sums including large counts."""
    planner = NativePlanner.from_metadata(**metadata)
    counts = (
        np.random.default_rng(173)
        .integers(0, 65536, size=(5, 36864), dtype=np.uint16)
        .astype(np.int64)
    )
    previous = None
    for mask in _masks(metadata["valid"]):
        seed, residual, fine, coarse, _ = planner(mask, previous)
        parent = fine[:576].reshape(24, 24) + np.repeat(
            np.repeat(coarse.reshape(6, 6), 4, axis=0), 4, axis=1
        )
        leaf = np.empty(1184, np.int8)
        leaf[:576] = parent.ravel()
        children = leaf[576:].reshape(152, 4)
        children[:] = parent.ravel()[metadata["parents"], None]
        children[np.arange(152)[:, None], metadata["stored_positions"]] += fine[
            576:
        ].reshape(152, 3)
        reconstructed = residual + leaf[metadata["leaf_of"]] * metadata["valid"]
        if seed == "total":
            reconstructed += metadata["valid"]
        elif seed == "previous":
            reconstructed += previous
        np.testing.assert_array_equal(reconstructed, mask)
        np.testing.assert_array_equal(counts @ reconstructed, counts @ mask)
        previous = mask
