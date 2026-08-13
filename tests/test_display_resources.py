import numpy as np
import pytest

from quantem.gpu.display import colormap_lut, colormap_names, metal_source


def test_metal_source_exposes_display_contract():
    source = metal_source()
    for function in (
        "quantem_display_vertex",
        "quantem_display_fragment",
        "quantem_range_u32",
        "quantem_histogram_u32",
    ):
        assert function in source


@pytest.mark.parametrize("name", colormap_names())
def test_colormap_lut_shape_range_and_alpha(name):
    lut = colormap_lut(name)
    assert lut.shape == (256, 4)
    assert lut.dtype == np.float32
    assert np.isfinite(lut).all()
    assert ((0 <= lut) & (lut <= 1)).all()
    np.testing.assert_array_equal(lut[:, 3], 1)


def test_gray_lut_endpoints():
    lut = colormap_lut("gray")
    np.testing.assert_array_equal(lut[0], [0, 0, 0, 1])
    np.testing.assert_array_equal(lut[-1], [1, 1, 1, 1])


def test_unknown_colormap_explains_valid_choices():
    with pytest.raises(ValueError, match="Choose one of: gray, viridis"):
        colormap_lut("not-a-colormap")
