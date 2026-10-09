import numpy as np
import pytest

from quantem.gpu.display import colormap_lut, colormap_names
from tests.parity.display_reference import colorize, histogram, normalize, transform


def _fixture() -> np.ndarray:
    return np.asarray(
        [-7, -3, -1, 0, 0.25, 0.5, 0.75, 1, 3, 7],
        dtype=np.float32,
    )


def test_signed_log_reference_preserves_negative_and_positive_signal() -> None:
    values = np.asarray([-7, -3, 0, 3, 7], dtype=np.float32)
    expected = np.copysign(np.log1p(np.abs(values)), values).astype(np.float32)
    np.testing.assert_array_equal(transform(values, "log"), expected)

    got = normalize(values, -7, 7, "log")
    np.testing.assert_allclose(
        got,
        np.asarray([0, 1 / 6, 1 / 2, 5 / 6, 1], dtype=np.float32),
        rtol=0,
        atol=2e-7,
    )


def test_histogram_uses_exact_256_bin_edges_and_preserves_count() -> None:
    values = np.asarray([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)
    bins = histogram(values, 0, 1)
    assert bins.dtype == np.uint32
    assert int(bins.sum()) == values.size
    np.testing.assert_array_equal(np.flatnonzero(bins), [0, 64, 128, 192, 255])
    np.testing.assert_array_equal(bins[np.flatnonzero(bins)], 1)


def test_gray_colorize_uses_exact_floor_lut_indices() -> None:
    values = np.asarray([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)
    rgba = colorize(values, colormap_lut("gray"), 0, 1)
    np.testing.assert_array_equal(
        rgba,
        np.asarray(
            [
                [0, 0, 0, 255],
                [63, 63, 63, 255],
                [127, 127, 127, 255],
                [191, 191, 191, 255],
                [255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_constant_range_uses_midpoint_display_convention(scale: str) -> None:
    values = np.asarray([-7, 0, 7], dtype=np.float32)

    np.testing.assert_array_equal(
        normalize(values, 3, 3, scale),
        np.full(values.shape, 0.5, dtype=np.float32),
    )
    bins = histogram(values, 3, 3, scale)
    assert int(bins.sum()) == values.size
    np.testing.assert_array_equal(np.flatnonzero(bins), [128])
    rgba = colorize(values, colormap_lut("gray"), 3, 3, scale)
    np.testing.assert_array_equal(
        rgba,
        np.tile(np.asarray([127, 127, 127, 255], dtype=np.uint8), (3, 1)),
    )


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_nonfinite_display_policy_is_explicit(scale: str) -> None:
    values = np.asarray([np.nan, -np.inf, np.inf, -1, 0, 1], dtype=np.float32)
    expected = np.asarray([0, 0, 1, 0, 0.5, 1], dtype=np.float32)

    np.testing.assert_array_equal(normalize(values, -1, 1, scale), expected)
    bins = histogram(values, -1, 1, scale)
    assert int(bins.sum()) == 3
    np.testing.assert_array_equal(np.flatnonzero(bins), [0, 128, 255])
    rgba = colorize(values, colormap_lut("gray"), -1, 1, scale)
    np.testing.assert_array_equal(
        rgba,
        np.asarray(
            [
                [0, 0, 0, 255],
                [0, 0, 0, 255],
                [255, 255, 255, 255],
                [0, 0, 0, 255],
                [127, 127, 127, 255],
                [255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_float32_extreme_range_does_not_overflow_normalization(scale: str) -> None:
    limit = np.finfo(np.float32).max
    values = np.asarray([-limit, -1, -0.0, 0.0, 1, limit], dtype=np.float32)
    normalized = normalize(values, -limit, limit, scale)

    assert np.all(np.isfinite(normalized))
    assert normalized[0] == 0
    assert normalized[-1] == 1
    assert normalized[2] == 0.5
    assert normalized[3] == 0.5
    if scale == "linear":
        np.testing.assert_allclose(normalized[1:5], 0.5, rtol=0, atol=1e-7)
    else:
        assert normalized[1] < 0.5 < normalized[4]


def test_python_and_metal_expose_identical_colormap_names() -> None:
    assert colormap_names() == (
        "gray",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "magenta",
        "hot",
        "hsv",
        "turbo",
        "RdBu",
        "cividis",
        "seismic",
        "RdBu_r",
        "twilight",
        "twilight_shifted",
    )
