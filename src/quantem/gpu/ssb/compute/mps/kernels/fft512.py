"""MPS/Metal SSB FFT configuration for specialized 512x512 kernels."""

from .common import MPSFFTConfig

CONFIG = MPSFFTConfig(
    size=512,
    digit_reverse_define="",
    digit_reverse_undef="",
    radix4_max=0,
    has_final_radix2=False,
    specialized=True,
)

__all__ = ["CONFIG"]
