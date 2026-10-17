"""Phase reduction preserves the original GPU result across FFT layouts."""

import platform

import numpy as np
import pytest


@pytest.fixture
def mx():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple silicon GPU")
    return pytest.importorskip("mlx.core")


def _original_phase_sum(mx, obj):
    """Frozen pre-optimization Metal reduction, including its layout copy."""
    batch, chunk, ny, nx = obj.shape
    kernel = mx.fast.metal_kernel(
        name=f"reference_phase_sum_{batch}_{chunk}_{ny}_{nx}",
        input_names=["obj"],
        output_names=["out"],
        compile_options={"math_mode": "fast"},
        source=f"""
            uint elem = thread_position_in_grid.x;
            constexpr uint PLANE = {ny * nx};
            if (elem >= {batch}u * PLANE) return;
            uint batch = elem / PLANE;
            uint pixel = elem - batch * PLANE;
            size_t base = size_t(batch) * {chunk}u * PLANE + pixel;
            float sum = 0.0f;
            for (uint bf = 0; bf < {chunk}u; ++bf) {{
                auto z = obj[base + size_t(bf) * PLANE];
                sum += metal::atan2(z.imag, z.real);
            }}
            out[elem] = sum;
        """,
    )
    return kernel(
        inputs=[obj],
        grid=(batch * ny * nx, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, ny, nx)],
        output_dtypes=[mx.float32],
    )[0]


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize(
    "layout", ["contiguous", "column_fft", "transpose", "slice", "reverse", "broadcast"]
)
def test_phase_reduction_matches_original_gpu_for_strided_objects(mx, batch, layout):
    from quantem.gpu.ssb.mps.kernels.phase_sums import phase_sum_from_complex

    rng = np.random.default_rng(73)
    shape = (batch, 7, 24, 32)
    source = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    source[..., 0, 0] = 0
    obj = mx.array(source)
    if layout == "column_fft":
        obj = mx.fft.ifft(mx.fft.ifft(obj, axis=-1), axis=-2)
    elif layout == "transpose":
        obj = mx.transpose(obj, (0, 1, 3, 2))
    elif layout == "slice":
        obj = obj[:, ::2, 1::2, ::3]
    elif layout == "reverse":
        obj = obj[:, ::-1, ::-1, ::-1]
    elif layout == "broadcast":
        obj = mx.broadcast_to(obj[:, :1, :1, :], shape)
    reference = np.asarray(_original_phase_sum(mx, obj))
    result = np.asarray(phase_sum_from_complex(mx, obj))
    np.testing.assert_array_equal(result, reference)
    assert result.dtype == np.float32
    if batch == 1:
        np.testing.assert_array_equal(phase_sum_from_complex(mx, obj[0]), reference[0])
