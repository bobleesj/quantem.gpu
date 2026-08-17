import json
from importlib.resources import files

import numpy as np
import pytest

from quantem.gpu.display.geometry import rotate_stack_inplane
from quantem.gpu.display.reference import dequantize_uint8


def _goldens() -> dict:
    path = files("quantem.gpu").joinpath("display/goldens/parity.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _goldens()["quantized"], ids=lambda case: case["name"])
def test_uint8_dequantization_matches_shared_golden(case: dict) -> None:
    result = dequantize_uint8(
        np.asarray(case["bytes"], dtype=np.uint8),
        case["low"],
        case["high"],
    )
    np.testing.assert_allclose(result, case["expected"], rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (np.nan, 7.0, [0.0, 7.0]),
        (-2.0, np.inf, [-2.0, -2.0]),
        (4.0, -4.0, [4.0, 4.0]),
    ],
)
def test_uint8_dequantization_range_edge_policy(
    low: float,
    high: float,
    expected: list[float],
) -> None:
    result = dequantize_uint8(np.asarray([0, 255], dtype=np.uint8), low, high)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("case", _goldens()["rotation"], ids=lambda case: case["name"])
def test_rotation_matches_shared_golden(case: dict) -> None:
    source = np.asarray(case["input"], dtype=np.float32).reshape(case["shape"])
    result = rotate_stack_inplane(source, case["angle_degrees"])
    np.testing.assert_allclose(
        result.ravel(),
        np.asarray(case["expected"], dtype=np.float32),
        rtol=0,
        atol=2e-6,
    )


def test_rotation_identity_preserves_original_object() -> None:
    source = np.arange(15, dtype=np.int16).reshape(1, 3, 5)
    assert rotate_stack_inplane(source, 360) is source


@pytest.mark.parametrize("bad", [True, None, np.nan, np.inf, -np.inf, "bad"])
def test_rotation_rejects_invalid_angles(bad: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        rotate_stack_inplane(np.zeros((1, 3, 5), dtype=np.float32), bad)


def test_rotation_rejects_non_stack_shape() -> None:
    with pytest.raises(ValueError, match="frames, rows, columns"):
        rotate_stack_inplane(np.zeros((3, 5), dtype=np.float32), 10)


def test_rotation_preserves_constant_and_nonfinite_policy() -> None:
    constant = np.full((2, 5, 3), -7.25, dtype=np.float32)
    np.testing.assert_allclose(rotate_stack_inplane(constant, 31), constant)
    nonfinite = np.zeros((1, 3, 5), dtype=np.float32)
    nonfinite[0, 1, 2] = np.nan
    result = rotate_stack_inplane(nonfinite, 17)
    assert np.isnan(result).any()


@pytest.mark.parametrize("case", _goldens()["reciprocal"], ids=lambda case: case["name"])
def test_reciprocal_coordinate_goldens_are_self_consistent(case: dict) -> None:
    row_frequency = case["row_offset"] / (case["rows"] * case["row_sampling"])
    column_frequency = case["column_offset"] / (
        case["columns"] * case["column_sampling"]
    )
    spatial_frequency = np.hypot(row_frequency, column_frequency)
    d_spacing = 1 / spatial_frequency if spatial_frequency > 0 else None
    expected = case["expected"]
    np.testing.assert_allclose(
        [row_frequency, column_frequency, spatial_frequency],
        expected[:3],
        rtol=0,
        atol=1e-12,
    )
    if expected[3] is None:
        assert d_spacing is None
    else:
        assert d_spacing == pytest.approx(expected[3])
