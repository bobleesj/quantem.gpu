"""Deterministic fixed-size CUDA SSB kernel registry."""

from functools import cache

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.ssb.cuda.kernels.common import CustomFFTBase
from quantem.gpu.ssb.cuda.kernels.fft128 import CustomFFT128
from quantem.gpu.ssb.cuda.kernels.fft256 import CustomFFT256
from quantem.gpu.ssb.cuda.kernels.fft512 import CustomFFT512
from quantem.gpu.ssb.cuda.kernels.fft1024 import CustomFFT1024

# the CUDA backend pads every scan to one of these square sizes before a kernel set is requested
_KERNEL_SETS = {
    128: CustomFFT128,
    256: CustomFFT256,
    512: CustomFFT512,
    1024: CustomFFT1024,
}


def get_fft_kernel(scan_size: int) -> CustomFFTBase:
    """Return the CUDA FFT kernel set for one supported square scan size on the current CuPy device."""
    return _kernel_set(int(scan_size), cp.cuda.runtime.getDevice())


@cache
def _kernel_set(scan_size: int, device: int) -> CustomFFTBase:
    """Compile one size's kernel set on ``device``, once per process.

    A kernel set holds the function handles and the twiddle table of the CUDA context it was compiled in. One set
    per process made a session on a second GPU launch the first GPU's kernels: a fit on cuda:1 returned wrong
    aberrations with loss 0 and no error.
    """
    with cp.cuda.Device(device):
        return _KERNEL_SETS[scan_size]()


__all__ = ["get_fft_kernel"]
