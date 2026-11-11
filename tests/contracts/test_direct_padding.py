"""One scan-padding policy for CUDA and MPS direct ptychography: which scans pad, to what side, and the one printed line."""

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.padding import pad_scan, padded_side


@pytest.mark.parametrize(("shape", "any_square", "side"), [
    ((256, 256), False, None),
    ((200, 200), False, 256),
    ((200, 200), True, None),
    ((96, 130), True, 256),
    ((37, 53), False, 128),
    ((1024, 1000), False, 1024),
])
def test_padded_side_is_the_next_supported_square(shape, any_square, side) -> None:
    assert padded_side(shape, any_square=any_square) == side


@pytest.mark.parametrize("any_square", [False, True])
def test_scans_above_1024_are_refused_on_both_backends(any_square) -> None:
    with pytest.raises(ValueError, match="at most 1024 x 1024 positions; got 1025 x 2.*never crops"):
        padded_side((1025, 2), any_square=any_square)


def test_padding_keeps_every_count_and_announces_the_padded_fraction(capsys) -> None:
    counts = np.random.default_rng(3).integers(0, 65536, (37, 53, 4, 5), dtype=np.uint16)
    padded = pad_scan(counts, any_square=True)
    # the integer mean: the exact total divided in float64, truncated to the counts' dtype
    pattern = (counts.sum(axis=(0, 1), dtype=np.int64) / (37 * 53)).astype(np.uint16)
    expected = np.broadcast_to(pattern, (128, 128, 4, 5)).copy()
    expected[45:82, 37:90] = counts
    np.testing.assert_array_equal(padded, expected)
    assert capsys.readouterr().out == (
        "Direct ptycho: 37 x 53 scan padded to 128 x 128 with its mean pattern (88% of positions are padding); "
        "crop to a supported side (128, 256, 512, 1024) to fit measured positions only.\n"
    )
    square = counts[:, :37]
    assert pad_scan(square, any_square=True) is square
    assert capsys.readouterr().out == ""
