"""FP8 type tags are explicit; byte storage alone never implies quantization."""

import h5py
import ml_dtypes
import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats.float8 import FLOAT8_DTYPES


@pytest.mark.parametrize("tag,dtype", (("float8", "u1"), ("float8_e4m3fn", "u2")))
def test_invalid_fp8_tag_is_rejected_before_loading(tmp_path, tag, dtype):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", shape=(2, 2, 4, 4), dtype=dtype)
        data.attrs["quantem_dtype"] = tag
    with pytest.raises(ValueError, match="FP8"):
        io.inspect(path)


def test_untagged_bytes_remain_integer_counts(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", shape=(2, 2, 4, 4), dtype="u1")
    assert io.inspect(path).dtype == "uint8"


def test_fp8_selection_is_refused_and_the_default_fill_has_nothing_to_change():
    """An FP8 array carries no detector mask, so io.load's default median fill leaves every bit; selection is refused.

    A file whose mask flags pixels in a non-count dtype is refused by the fill itself (``io.arrays``).
    """
    data = np.zeros((2, 2, 4, 4), np.uint8).view(FLOAT8_DTYPES["float8_e4m3fn"])
    with pytest.raises(ValueError, match="without conversion"):
        io.load(data, scan_region=(0, 1, 0, 1), backend="cuda")


def test_unknown_array_variant_is_rejected_before_upload():
    values = np.zeros((2, 2, 4, 4), ml_dtypes.float8_e4m3)
    with pytest.raises(NotImplementedError, match="supported variants"):
        io.load(values, backend="cuda")
