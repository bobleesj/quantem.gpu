"""Accurate complex64 Fourier transforms on the Metal command stream.

The Fourier butterflies retain three float32 terms until the final complex64
output. This avoids accumulating rounding through the transform. No detector
values, waves or gradients leave the GPU. Scalar twiddles are fixed metadata.
"""

import math
import os
import struct
from functools import lru_cache

import torch
from quantem.gpu.direct_ptycho._resident_precision import HEADER


def _parts(value):
    high = struct.unpack("f", struct.pack("f", value))[0]
    middle = struct.unpack("f", struct.pack("f", value - high))[0]
    return high, middle, value - high - middle


@lru_cache(maxsize=16)
def _plan(size, device):
    if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
        raise RuntimeError("Accurate MPS FFT requires PYTORCH_MPS_FAST_MATH=0.")
    power = size if size & (size - 1) == 0 else size // 3
    if power & (power - 1) or size not in (power, 3 * power) or size > 1024:
        raise ValueError(
            "Accurate MPS FFT requires a power of two or three times a power of two, up to 1024."
        )
    terms = []
    for k in range(size):
        angle = -2 * math.pi * k / size
        terms.extend(_parts(math.cos(angle)))
        terms.extend(_parts(math.sin(angle)))
    twiddle = torch.tensor(terms, dtype=torch.float32, device=device)
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n"
        + HEADER
        + r"""
struct Wide { Real r, i; };
inline Wide plus_w(Wide a, Wide b) { return {a.r + b.r, a.i + b.i}; }
inline Wide minus_w(Wide a, Wide b) { return {a.r - b.r, a.i - b.i}; }
inline Wide times_w(Wide a, Wide b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
inline Wide root_w(device const float* table, uint k, bool inverse) {
    return {load_real(table, 2*k), inverse ? -load_real(table, 2*k+1) : load_real(table, 2*k+1)};
}
inline Wide get_w(threadgroup const float* p, uint k) {
    Wide value;
    value.r.a=p[6*k]; value.r.b=p[6*k+1]; value.r.c=p[6*k+2];
    value.i.a=p[6*k+3]; value.i.b=p[6*k+4]; value.i.c=p[6*k+5];
    return value;
}
inline void set_w(threadgroup float* p, uint k, Wide v) {
    p[6*k]=v.r.a; p[6*k+1]=v.r.b; p[6*k+2]=v.r.c;
    p[6*k+3]=v.i.a; p[6*k+4]=v.i.b; p[6*k+5]=v.i.c;
}
inline uint reverse_index(uint x) {
    uint y = 0;
    for (uint j=0; j<BITS; ++j) { y = (y << 1) | (x & 1); x >>= 1; }
    return y;
}
kernel void transform(
    device const float* input, device float* output,
    device const float* table, device const float* scale,
    constant long& rows, constant long& columns,
    constant long& axis, constant long& inverse,
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]])
{
    threadgroup float work[6*N];
    // Axis 1 reads complex64 rows and emits expanded rows; axis 0 reads
    // expanded columns and rounds the complete two-dimensional transform.
    uint lines = axis == 1 ? rows : columns;
    uint batch = group / lines, line = group % lines;
    uint source = axis == 1 ? batch*rows*columns + line*columns + tid
                            : batch*rows*columns + tid*columns + line;
    Wide value = axis == 1 ? Wide{Real(input[2*source]), Real(input[2*source+1])}
                           : Wide{load_real(input, 2*source), load_real(input, 2*source+1)};
    uint part = tid % RADIX, digit = tid / RADIX;
    set_w(work, part*M + reverse_index(digit), value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=2; span<=M; span*=2) {
        if (tid < N/2) {
            uint part = tid / (M/2), pair = tid % (M/2);
            uint j = pair % (span/2), first = part*M + (pair/(span/2))*span + j;
            Wide a = get_w(work,first);
            Wide b = times_w(get_w(work,first+span/2), root_w(table, j*N/span, inverse));
            set_w(work,first,plus_w(a,b)); set_w(work,first+span/2,minus_w(a,b));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    Wide result = get_w(work,tid % M);
    for (uint part=1; part<RADIX; ++part)
        result = plus_w(result, times_w(get_w(work,part*M+tid%M), root_w(table, (part*tid)%N, inverse)));
    if (axis == 1) {
        store_real(output, 2*source, result.r); store_real(output, 2*source+1, result.i);
    } else {
        Real factor = load_real(scale, 0);
        output[2*source] = scalar(result.r * factor);
        output[2*source+1] = scalar(result.i * factor);
    }
}
"""
    )
    source = (
        f"#define N {size}\n#define M {power}\n#define RADIX {size // power}\n#define BITS {power.bit_length() - 1}\n"
        + source
    )
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError(
            "Accurate MPS FFT needs PyTorch 2.8 or newer; update the MPS environment."
        )
    return torch.mps.compile_shader(source), twiddle


def supported(shape: tuple[int, ...]) -> bool:
    """Whether both transform axes fit the exact mixed-radix kernel."""

    def valid(n):
        m = n if n & (n - 1) == 0 else n // 3
        return n >= 2 and n <= 1024 and m > 0 and m & (m - 1) == 0 and n in (m, 3 * m)

    return len(shape) >= 2 and all(valid(n) for n in shape[-2:])


def _transform(value, inverse, norm):
    rows, columns = value.shape[-2:]
    count = rows * columns
    scale = (
        1 / math.sqrt(count)
        if norm == "ortho"
        else (
            1 / count
            if (inverse and norm in (None, "backward"))
            or (not inverse and norm == "forward")
            else 1
        )
    )
    packed_scale = torch.tensor(_parts(scale), dtype=torch.float32, device=value.device)
    value = value.resolve_conj().contiguous()
    intermediate = torch.empty(
        (*value.shape, 6), dtype=torch.float32, device=value.device
    )
    output = torch.empty_like(value)
    batches = value.numel() // count
    row_lib, row_table = _plan(columns, str(value.device))
    col_lib, col_table = _plan(rows, str(value.device))
    row_lib.transform(
        torch.view_as_real(value),
        intermediate,
        row_table,
        packed_scale,
        rows,
        columns,
        1,
        int(inverse),
        threads=batches * rows * columns,
        group_size=columns,
    )
    col_lib.transform(
        intermediate,
        torch.view_as_real(output),
        col_table,
        packed_scale,
        rows,
        columns,
        0,
        int(inverse),
        threads=batches * columns * rows,
        group_size=rows,
    )
    return output


class _Fourier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, inverse, norm):
        ctx.inverse, ctx.norm = inverse, norm
        return _transform(value, inverse, norm)

    @staticmethod
    def backward(ctx, gradient):
        opposite = {
            None: "forward",
            "backward": "forward",
            "forward": "backward",
            "ortho": "ortho",
        }[ctx.norm]
        return _Fourier.apply(gradient, not ctx.inverse, opposite), None, None


def fft2(
    value: torch.Tensor, *, inverse: bool = False, norm: str | None = None
) -> torch.Tensor:
    """Transform the last two axes without intermediate complex64 rounding.

    Input and output are complex64 MPS tensors. Normalization and its
    adjoint follow ``torch.fft``. The requested grid is retained exactly.

    Parameters
    ----------
    value : torch.Tensor
        Complex64 MPS tensor, with the last two axes each a power of two or
        three times a power of two, between 2 and 1024 pixels.
    inverse : bool, optional
        Compute the inverse transform.
    norm : str or None, optional
        ``"backward"``, ``"forward"`` or ``"ortho"``, as in Torch.

    Returns
    -------
    torch.Tensor
        Complex64 transform on MPS, with a differentiable GPU adjoint.
    """
    if value.device.type != "mps" or value.dtype != torch.complex64:
        raise TypeError("Accurate MPS FFT requires a complex64 MPS tensor.")
    if value.ndim < 2 or not supported(value.shape):
        raise ValueError(
            "Unsupported MPS FFT shape; retain the existing transform for this grid."
        )
    if norm not in (None, "backward", "forward", "ortho"):
        raise ValueError("FFT norm must be backward, forward, ortho or None.")
    return _Fourier.apply(value, inverse, norm)
