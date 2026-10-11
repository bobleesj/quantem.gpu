"""Deterministic fixed-size CUDA SSB kernel registry."""

from quantem.gpu.ssb.cuda.kernels.common import CustomFFTBase
from quantem.gpu.ssb.cuda.kernels.fft128 import get_custom_fft_128
from quantem.gpu.ssb.cuda.kernels.fft256 import get_custom_fft_256
from quantem.gpu.ssb.cuda.kernels.fft512 import get_custom_fft_512
from quantem.gpu.ssb.cuda.kernels.fft1024 import get_custom_fft_1024

# the CUDA backend pads every scan to one of these square sizes before a kernel set is requested
_FACTORIES = {
    128: get_custom_fft_128,
    256: get_custom_fft_256,
    512: get_custom_fft_512,
    1024: get_custom_fft_1024,
}


def get_fft_kernel(scan_size: int) -> CustomFFTBase:
    """Return the CUDA FFT kernel set for one supported square scan size."""
    return _FACTORIES[int(scan_size)]()


__all__ = ["get_fft_kernel"]
