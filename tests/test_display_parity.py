from importlib.resources import files

import numpy as np
import pytest

from quantem.gpu.display import colormap_lut, colormap_names
from quantem.gpu.display.reference import colorize, histogram, normalize, transform


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


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_cuda_display_matches_reference(scale: str) -> None:
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")
    from quantem.gpu.display import cuda

    values = _fixture()
    low, high = -7.0, 7.0
    expected_normalized = normalize(values, low, high, scale)
    expected_histogram = histogram(values, low, high, scale)
    expected_rgba = colorize(values, colormap_lut("viridis"), low, high, scale)

    device_values = cp.asarray(values)
    cp.testing.assert_allclose(
        cuda.normalize(device_values, low, high, scale),
        cp.asarray(expected_normalized),
        rtol=0,
        atol=2e-7,
    )
    cp.testing.assert_array_equal(
        cuda.histogram(device_values, low, high, scale),
        cp.asarray(expected_histogram),
    )
    cp.testing.assert_array_equal(
        cuda.colorize(
            device_values,
            cp.asarray(colormap_lut("viridis")),
            low,
            high,
            scale,
        ),
        cp.asarray(expected_rgba),
    )


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_cuda_constant_and_nonfinite_display_matches_reference(scale: str) -> None:
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")
    from quantem.gpu.display import cuda

    cases = [
        (np.asarray([-7, 0, 7], dtype=np.float32), 3.0, 3.0),
        (
            np.asarray([np.nan, -np.inf, np.inf, -1, 0, 1], dtype=np.float32),
            -1.0,
            1.0,
        ),
        (
            np.asarray(
                [-np.finfo(np.float32).max, 0, np.finfo(np.float32).max],
                dtype=np.float32,
            ),
            -float(np.finfo(np.float32).max),
            float(np.finfo(np.float32).max),
        ),
    ]
    for values, low, high in cases:
        device_values = cp.asarray(values)
        cp.testing.assert_array_equal(
            cuda.normalize(device_values, low, high, scale),
            cp.asarray(normalize(values, low, high, scale)),
        )
        cp.testing.assert_array_equal(
            cuda.histogram(device_values, low, high, scale),
            cp.asarray(histogram(values, low, high, scale)),
        )
        cp.testing.assert_array_equal(
            cuda.colorize(
                device_values,
                cp.asarray(colormap_lut("gray")),
                low,
                high,
                scale,
            ),
            cp.asarray(colorize(values, colormap_lut("gray"), low, high, scale)),
        )


def test_webgpu_display_sources_are_packaged_and_share_colormap_data() -> None:
    root = files("quantem.gpu")
    colormaps = root.joinpath("display/webgpu/colormaps.ts").read_text(
        encoding="utf-8"
    )
    fft = root.joinpath("display/webgpu/fft.ts").read_text(encoding="utf-8")
    fft_metrics = root.joinpath("display/webgpu/fftMetrics.ts").read_text(
        encoding="utf-8"
    )
    geometry = root.joinpath("display/webgpu/geometry.ts").read_text(
        encoding="utf-8"
    )
    quantization = root.joinpath("display/webgpu/quantization.ts").read_text(
        encoding="utf-8"
    )
    stats = root.joinpath("display/webgpu/stats.ts").read_text(encoding="utf-8")

    assert "MetalDisplayKernels/Resources/colormaps.json" in colormaps
    assert "export class GPUColormapEngine" in colormaps
    assert "computeHistogramBatch" in colormaps
    assert "uploadUint8Data" in colormaps
    assert "SCALED_UINT8_COLORMAP_SHADER" in colormaps
    assert 'dataKind: "f32" | "u8"' in colormaps
    assert "renderSlotScaledToImageBitmapAsync" in colormaps
    assert "renderPanelSlotsToImageBitmapAsync" in colormaps
    assert "renderSharedGridToImageBitmapAsync" in colormaps
    assert "renderCombinedPanelRegionsToImageBitmapAsync" in colormaps
    assert "renderSlotDirectWithGpuRangeToImageBitmapAsync" in colormaps
    assert "await this.device.queue.onSubmittedWorkDone()" in colormaps
    assert "let stride = grid.x * 256u" in colormaps
    assert "this.device.limits.maxComputeWorkgroupsPerDimension" in colormaps
    assert "log(1.0 + max(val, 0.0))" not in colormaps
    assert "val = -log(1.0 - val)" in colormaps
    assert "export class WebGPUFFT" in fft
    assert "export function autoEnhanceFFT" in fft
    assert "reciprocalCoordinatesFromShiftedOffset" in fft
    assert "export function shiftedMagnitude" in fft
    assert "export function computeFftQualityMetrics" in fft_metrics
    assert "export function cropMaskedRegion" in geometry
    assert "export async function rotateStackInPlaneWebGPU" in geometry
    assert "export async function sampleLineProfileUint8WebGPU" in geometry
    assert "export async function dequantizeUint8WebGPU" in quantization
    assert "export function applyLogScale" in stats


def test_python_metal_and_webgpu_expose_identical_colormap_names() -> None:
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
