import pytest


def test_ssb_explore_points_to_showptycho_ui() -> None:
    """Interactive ptychography UI is owned by quantem.widget.ShowPtycho."""
    from quantem.gpu.ssb.reconstruction import SSB

    ssb = SSB.__new__(SSB)
    with pytest.raises(RuntimeError, match=r"quantem\.widget\.ShowPtycho\(ssb\)"):
        ssb.explore()
