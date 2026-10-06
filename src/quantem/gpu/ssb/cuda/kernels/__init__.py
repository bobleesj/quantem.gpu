"""Deterministic fixed-size CUDA SSB kernel registry."""

from collections.abc import Callable

from quantem.gpu.ssb.cuda.kernels.common import CustomFFTBase
from quantem.gpu.ssb.cuda.kernels.fft128 import get_custom_fft_128
from quantem.gpu.ssb.cuda.kernels.fft256 import get_custom_fft_256
from quantem.gpu.ssb.cuda.kernels.fft512 import get_custom_fft_512
from quantem.gpu.ssb.cuda.kernels.fft1024 import get_custom_fft_1024

CUDA_FFT_FACTORIES: dict[int, Callable[[], CustomFFTBase]] = {
    128: get_custom_fft_128,
    256: get_custom_fft_256,
    512: get_custom_fft_512,
    1024: get_custom_fft_1024,
}


def get_fft_kernel(scan_size: int) -> CustomFFTBase:
    """Return the CUDA FFT kernel set for one supported square scan size."""

    size = int(scan_size)
    try:
        factory = CUDA_FFT_FACTORIES[size]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in CUDA_FFT_FACTORIES)
        raise ValueError(
            f"CUDA SSB supports square scan sizes {supported}; got {size}."
        ) from exc
    return factory()


__all__ = ["CUDA_FFT_FACTORIES", "get_fft_kernel"]
