"""Removed format and representation names fail before any file is touched."""

import numpy as np
import pytest

from quantem.gpu import io


def test_removed_names_are_rejected_before_saving_or_loading(tmp_path):
    counts = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    for name in ("arina-h5", "arina_h5", "h5"):
        with pytest.raises(ValueError, match="use format='arina' or 'quantem'"):
            io.save(tmp_path / "unused", counts, format=name, backend="cpu")
    with pytest.raises(ValueError, match="representation must be one of"):
        io.load(tmp_path / "unused", representation="lossless_packed")
    assert not (tmp_path / "unused").exists()
