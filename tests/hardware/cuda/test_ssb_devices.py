"""CUDA SSB on the GPU that holds the scan, which need not be the current one."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _require_cuda(count: int = 1):
    try:
        visible = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    if visible < count:
        pytest.skip(f"Needs {count} visible CUDA GPUs; GPU 1 is kept for reconstruction, so run it on a free pair.")


def test_fft_kernel_sets_are_built_for_the_current_gpu(monkeypatch):
    """A kernel set holds the function handles and twiddle table of the CUDA context it was compiled in.

    One set per process made a session on a second GPU launch the first GPU's kernels: a fit on cuda:1 returned
    wrong aberrations with loss 0 and no error. The set is now looked up by size and current device.
    """
    _require_cuda()
    from quantem.gpu.ssb.cuda import kernels

    requested = []
    monkeypatch.setattr(kernels, "_kernel_set", lambda size, device: requested.append((size, device)))
    monkeypatch.setattr(kernels.cp.cuda.runtime, "getDevice", lambda: 1)
    kernels.get_fft_kernel(512)

    assert requested == [(512, 1)]


def test_a_fit_on_the_second_gpu_equals_the_fit_on_the_first():
    """The same counts on cuda:0 and cuda:1 give the same aberrations and loss, whichever GPU is current."""
    _require_cuda(2)
    from quantem.gpu import SSB

    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rows, cols = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rows, cols) < 10] += 200
    settings = dict(backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(3.0, 3.0))
    fits = []
    for device in (0, 1):
        with cp.cuda.Device(device):
            data = cp.asarray(counts)
        with SSB(data, **settings) as ssb:
            fits.append(ssb.find_aberrations(search="primary", tilt=False, num_trials=8, check_rotation=False, verbose=False))

    assert dict(fits[1].aberrations) == dict(fits[0].aberrations)
    assert fits[1].loss == fits[0].loss > 0
