"""Compatibility imports; canonical implementation: ``quantem.gpu.ssb.backends.cuda.kernels.fft512``."""

from quantem.gpu.ssb.backends.cuda.kernels.fft512 import (
    CustomFFT512 as CustomFFT512,
    _FFT512_KERNELS as _FFT512_KERNELS,
    _TWIDDLE_DECL as _TWIDDLE_DECL,
    get_custom_fft_512 as get_custom_fft_512,
)
