"""Compatibility imports; canonical implementation: ``quantem.gpu.ssb.backends.cuda.kernels.fft1024``."""

from quantem.gpu.ssb.backends.cuda.kernels.fft1024 import (
    CustomFFT1024 as CustomFFT1024,
    _FFT1024_KERNELS as _FFT1024_KERNELS,
    _TWIDDLE_DECL as _TWIDDLE_DECL,
    get_custom_fft_1024 as get_custom_fft_1024,
)
