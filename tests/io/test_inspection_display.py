"""Inspection rendering stays compact and does not expose calibration arrays."""
from quantem.gpu.io.inspect import Inspection


def test_inspection_display_is_compact_and_escaped():
    info = Inspection(True, 'header_complete_payload_unverified',
                      'Load to verify payload.', {'secret': 'do-not-display'}, None,
                      'resident', 4, 4, (2, 2), (8, 8), 'uint16',
                      {'path': '<script>unsafe</script>'})
    html = info._repr_html_()
    assert '<table' in html
    assert '<script>' not in html
    assert 'do-not-display' not in html
    assert 'payload unverified' in html
    assert '2 × 2' in repr(info)
