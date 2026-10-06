"""Twiddle tables and digit-reversal macros shared by the fused Metal inverse-FFT kernels."""

from functools import lru_cache

import numpy as np

from quantem.gpu.ssb.mps.kernels import get_fft_config


def twiddle_512(mx):
    """Return IDFT twiddles for 512-point custom MPS FFT kernels."""
    return twiddle_n(mx, 512)


def twiddle_n(mx, n: int):
    """Return IDFT twiddles for custom power-of-two MPS FFT kernels."""
    n = int(n)
    return mx.array(
        np.exp(2j * np.pi * np.arange(n, dtype=np.float32) / n).astype(
            np.complex64
        )
    )


@lru_cache(maxsize=1)
def twiddle_512_metal_header() -> str:
    """Return the exact complex64 twiddle bits as a Metal constant table."""
    values = np.exp(
        2j * np.pi * np.arange(512, dtype=np.float32) / 512
    ).astype(np.complex64)
    bits = values.view(np.uint32).reshape(512, 2)
    entries = ",".join(
        f"uint2(0x{int(real):08x}u,0x{int(imag):08x}u)"
        for real, imag in bits
    )
    return f"constant uint2 SSB_TWIDDLE_512[512] = {{{entries}}};"


def small_fft_macros(n: int) -> tuple[str, str, int, bool]:
    """Return Metal digit-reversal macros for fused radix-4 IFFTs."""
    config = get_fft_config(n)
    if config.specialized:
        raise ValueError(
            f"MPS {config.size}x{config.size} uses its specialized FFT path."
        )
    return (
        config.digit_reverse_define,
        config.digit_reverse_undef,
        config.radix4_max,
        config.has_final_radix2,
    )
