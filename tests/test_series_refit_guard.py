"""2026-11-16: a defocus refit that jumps or ends on its bound keeps the last frame's optics in a series."""

import numpy as np

from quantem.gpu.direct_ptycho import stream


def _step():
    step = stream._FrameStep.__new__(stream._FrameStep)
    step.bounds = np.array([[-500.0, 500.0], [-50.0, 50.0], [-50.0, 50.0]])
    return step


def test_a_small_defocus_change_is_trusted():
    assert _step().untrusted([12.0, 1.0, -2.0], {"C10_nm": 12.0}, {"C10_nm": 4.0}) == ""


def test_a_defocus_jump_keeps_the_last_frames_optics():
    said = _step().untrusted([180.0, 1.0, -2.0], {"C10_nm": 180.0}, {"C10_nm": 4.0})
    assert "defocus moved 176 nm" in said


def test_a_refit_on_its_bound_is_not_trusted():
    assert "search bound" in _step().untrusted([500.0, 1.0, -2.0], {"C10_nm": 500.0}, None)


def test_the_first_frame_has_nothing_to_compare_with():
    assert _step().untrusted([30.0, 0.0, 0.0], {"C10_nm": 30.0}, None) == ""
