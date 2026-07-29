"""Shared configuration for fixed-size MPS/Metal SSB FFT kernels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MPSFFTConfig:
    """Compile-time Metal FFT configuration for one square scan size."""

    size: int
    digit_reverse_define: str
    digit_reverse_undef: str
    radix4_max: int
    has_final_radix2: bool
    specialized: bool = False


__all__ = ["MPSFFTConfig"]
