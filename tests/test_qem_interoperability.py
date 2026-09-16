"""Native-written references -> selected GPU -> Python save -> moved-file reopen."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.io.qem_validation import validate_qem
from quantem.gpu.io._qem_metadata import microscopy_metadata

REFERENCES = Path(__file__).parent / "data" / "qem-v1"
BACKEND = os.environ.get("QEM_TEST_BACKEND")
pytestmark = pytest.mark.skipif(
    BACKEND is None, reason="Set QEM_TEST_BACKEND=mps or cuda for real GPU checks"
)


def _counts(source):
    result = source.decode_scan_range_device(0, int(np.prod(source.shape[:2])))
    if BACKEND == "cuda":
        return result.get().reshape(source.shape)
    try:
        return result.to_numpy().reshape(source.shape)
    finally:
        result.release()


@pytest.mark.parametrize("name", ["uint8", "uint16"])
def test_counts_and_metadata_across_writers(name, tmp_path):
    assert BACKEND in ("mps", "cuda"), "Choose an explicit hardware backend"
    expected = np.load(REFERENCES / f"{name}.npy")
    manifest = json.loads((REFERENCES / "manifest.json").read_text())
    entry = next(item for item in manifest["entries"] if item["name"] == name)
    original = io.load(REFERENCES / f"{name}.qem", backend=BACKEND, verbose=False)
    try:
        assert original.metadata["backend"] == BACKEND
        np.testing.assert_array_equal(_counts(original.data), expected)
        assert original.metadata["scientific_metadata"] == entry["scientific_metadata"]
        saved = tmp_path / f"{name}-python.qem"
        io.save(saved, original, backend=BACKEND)
    finally:
        original.close()
    moved = tmp_path / "moved"
    moved.mkdir()
    saved = saved.rename(moved / saved.name)
    assert validate_qem(saved)["integrity"] == "verified"
    restored = io.load(saved, backend=BACKEND, verbose=False)
    try:
        np.testing.assert_array_equal(_counts(restored.data), expected)
        assert restored.metadata["scientific_metadata"] == microscopy_metadata(entry["scientific_metadata"])
        if name == "uint16":
            assert restored.metadata["scan_sampling_A"] == pytest.approx([0.4, 0.6])
            assert restored.metadata["voltage_kV"] == 200
    finally:
        restored.close()
