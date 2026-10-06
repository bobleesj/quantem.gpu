"""Metal kernels that sum the phase (and its square) of corrected object planes over bright-field pixels."""

from functools import lru_cache

from quantem.gpu.ssb.mps.hardware import require_mlx


@lru_cache(maxsize=16)
def _phase_sums_kernel(batch: int, chunk: int, ny: int, nx: int):
    mx = require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        uint total = BATCH * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / PLANE;
        uint pixel = elem - batch * PLANE;
        size_t base = ((size_t)batch * (size_t)CHUNK * (size_t)PLANE) + (size_t)pixel;
        float s = 0.0f;
        float sq = 0.0f;
        for (uint bf = 0; bf < CHUNK; ++bf) {{
            auto z = obj[base + (size_t)bf * (size_t)PLANE];
            float a = metal::atan2(z.imag, z.real);
            s += a;
            sq += a * a;
        }}
        sum_out[elem] = s;
        sumsq_out[elem] = sq;
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_sums_n{int(batch)}_b{int(chunk)}_{int(ny)}_{int(nx)}",
        input_names=["obj"],
        output_names=["sum_out", "sumsq_out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def phase_sums_from_complex(mx, obj_chunk):
    """Metal fused atan2/sum/sumsq over BF pixels for a chunked object stack."""
    shape = tuple(int(x) for x in obj_chunk.shape)
    if len(shape) == 3:
        chunk, ny, nx = shape
        obj = obj_chunk[None, :, :, :]
        squeeze = True
        batch = 1
    elif len(shape) == 4:
        batch, chunk, ny, nx = shape
        obj = obj_chunk
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D object chunk, got shape {shape}.")
    kernel = _phase_sums_kernel(int(batch), int(chunk), int(ny), int(nx))
    outputs = kernel(
        inputs=[obj],
        template=[],
        grid=(int(batch) * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(batch), int(ny), int(nx)), (int(batch), int(ny), int(nx))],
        output_dtypes=[mx.float32, mx.float32],
    )
    if squeeze:
        return outputs[0][0], outputs[1][0]
    return outputs[0], outputs[1]


@lru_cache(maxsize=16)
def _phase_sum_kernel(batch: int, chunk: int, ny: int, nx: int):
    mx = require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        uint total = BATCH * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / PLANE;
        uint pixel = elem - batch * PLANE;
        size_t base = ((size_t)batch * (size_t)CHUNK * (size_t)PLANE) + (size_t)pixel;
        float s = 0.0f;
        for (uint bf = 0; bf < CHUNK; ++bf) {{
            auto z = obj[base + (size_t)bf * (size_t)PLANE];
            s += metal::atan2(z.imag, z.real);
        }}
        sum_out[elem] = s;
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_sum_n{int(batch)}_b{int(chunk)}_{int(ny)}_{int(nx)}",
        input_names=["obj"],
        output_names=["sum_out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def phase_sum_from_complex(mx, obj_chunk):
    """Metal fused atan2/sum over BF pixels for a chunked object stack."""
    shape = tuple(int(x) for x in obj_chunk.shape)
    if len(shape) == 3:
        chunk, ny, nx = shape
        obj = obj_chunk[None, :, :, :]
        squeeze = True
        batch = 1
    elif len(shape) == 4:
        batch, chunk, ny, nx = shape
        obj = obj_chunk
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D object chunk, got shape {shape}.")
    kernel = _phase_sum_kernel(int(batch), int(chunk), int(ny), int(nx))
    output = kernel(
        inputs=[obj],
        template=[],
        grid=(int(batch) * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(batch), int(ny), int(nx))],
        output_dtypes=[mx.float32],
    )[0]
    if squeeze:
        return output[0]
    return output
