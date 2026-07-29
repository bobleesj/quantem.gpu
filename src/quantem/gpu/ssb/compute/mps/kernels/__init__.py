"""Deterministic fixed-size MPS/Metal SSB kernel registry."""

from .common import MPSFFTConfig
from .fft128 import CONFIG as FFT128
from .fft256 import CONFIG as FFT256
from .fft512 import CONFIG as FFT512
from .fft1024 import CONFIG as FFT1024


MPS_FFT_CONFIGS: dict[int, MPSFFTConfig] = {
    128: FFT128,
    256: FFT256,
    512: FFT512,
    1024: FFT1024,
}


def get_fft_config(scan_size: int) -> MPSFFTConfig:
    """Return the Metal FFT configuration for one square scan size."""

    size = int(scan_size)
    try:
        return MPS_FFT_CONFIGS[size]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in MPS_FFT_CONFIGS)
        raise ValueError(
            f"MPS SSB supports square scan sizes {supported}; got {size}."
        ) from exc


__all__ = ["MPS_FFT_CONFIGS", "MPSFFTConfig", "get_fft_config"]
