import pytest


def test_cuda_backend_does_not_own_explore_ui() -> None:
    """Interactive ptychography UI is owned by quantem.widget.ShowPtycho."""
    pytest.importorskip("cupy")
    from quantem.gpu.ssb.compute.cuda.backend import CudaSSBBackend

    assert not hasattr(CudaSSBBackend, "explore")
