"""The joint fit, the resident search and the parameter slices share one depth-spread range: no GPU needed."""

import inspect

from quantem.gpu.ssb import thick_sample_fit


def test_the_depth_spread_search_range_is_one_public_constant():
    """The joint fit, the resident search and the parameter slices search the depth spread over one range."""
    from quantem.gpu.ssb import DEPTH_SPREAD_RANGE_NM, _resident_search
    from quantem.gpu.ssb.units import ENGINE_PER_NM

    assert DEPTH_SPREAD_RANGE_NM == (2.0, 60.0)
    assert thick_sample_fit.DEPTH_SPREAD_RANGE_A == tuple(value * ENGINE_PER_NM for value in DEPTH_SPREAD_RANGE_NM)
    signature = inspect.signature(_resident_search.fit_resident)
    assert signature.parameters["depth_spread_range_A"].default == thick_sample_fit.DEPTH_SPREAD_RANGE_A
