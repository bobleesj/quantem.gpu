"""Display size is quantem.widget's presets: one name, one meaning in both packages."""

import pytest

from quantem.gpu.display._layout import PANEL_WIDTHS, panel_inches


def test_presets_match_quantem_widget_widths():
    assert PANEL_WIDTHS == {"small": 300, "medium": 450, "large": 600, "xlarge": 900}
    assert [panel_inches(name) for name in PANEL_WIDTHS] == [3.0, 4.5, 6.0, 9.0]


@pytest.mark.parametrize("size", [450, 750, 0, True, "huge", None])
def test_a_number_or_unknown_name_is_refused_with_the_presets(size):
    with pytest.raises(ValueError, match="'small', 'medium', 'large' or 'xlarge'"):
        panel_inches(size)
