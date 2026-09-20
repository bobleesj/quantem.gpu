"""Adversarial bounds checks for the exact IEEE bit-lane QEM codec."""

import copy
import io

import numpy as np
import pytest

from quantem.gpu.io._qem_reference import read_envelope, save_array
from quantem.gpu.io.qem_validation import _validate_float_ans_layout


def test_float_ans_rejects_invalid_streams_before_gpu_upload(tmp_path):
    path = tmp_path / "zeros.qem"
    save_array(path, np.zeros((1, 1, 128, 128), dtype=np.float32))
    header, start = read_envelope(path)
    blob = path.read_bytes()
    _validate_float_ans_layout(io.BytesIO(blob), header, start)
    chunk = header["chunks"][0]
    mutations = [
        (chunk["model_offset"], b"\x40"),  # reserved model
        (chunk["offset_offset"], (1).to_bytes(4, "little")),  # nonzero initial offset
        (chunk["offset_offset"] + 4, (0xFFFFFFFF).to_bytes(4, "little")),
    ]
    for offset, values in mutations:
        damaged = bytearray(blob)
        damaged[start + offset:start + offset + len(values)] = values
        with pytest.raises(ValueError, match="ANS"):
            _validate_float_ans_layout(io.BytesIO(damaged), header, start)
    for field, value in (("scans", 513), ("first", 1), ("offset_bytes", 4)):
        changed = copy.deepcopy(header)
        changed["chunks"][0][field] = value
        with pytest.raises(ValueError, match="ANS"):
            _validate_float_ans_layout(io.BytesIO(blob), changed, start)
