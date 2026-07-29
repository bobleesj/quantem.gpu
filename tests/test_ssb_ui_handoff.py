import pytest


def test_ssb_explore_points_to_showptycho_ui() -> None:
    """Interactive ptychography UI is owned by quantem.widget.ShowPtycho."""
    pytest.importorskip("cupy")
    from quantem.gpu.ssb.compute.cuda.backend import CudaSSBBackend

    ssb = CudaSSBBackend.__new__(CudaSSBBackend)
    with pytest.raises(RuntimeError, match=r"quantem\.widget\.ShowPtycho\(ssb\)"):
        ssb.explore()
