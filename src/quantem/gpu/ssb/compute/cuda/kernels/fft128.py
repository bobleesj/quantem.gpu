"""Compatibility imports; canonical implementation: ``quantem.gpu.ssb.backends.cuda.kernels.fft128``."""

from quantem.gpu.ssb.backends.cuda.kernels.fft128 import (
    CustomFFT128 as CustomFFT128,
    _FFT128_KERNELS as _FFT128_KERNELS,
    _TWIDDLE_DECL as _TWIDDLE_DECL,
    get_custom_fft_128 as get_custom_fft_128,
)
