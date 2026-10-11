"""Frozen schema-2 references -> selected GPU -> Python save -> moved-file reopen."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats.qem.validation import validate_qem

REFERENCES = Path(__file__).parent / "data" / "qem-v2"
# What the frozen files record; a reader takes these from scientific_metadata, never from the
# detector_sampling / detector_sampling_unit copy the same files still hold in their metadata block.
CALIBRATION = dict(scan_sampling_A=[0.4, 0.6], detector_sampling_inv_A=[0.02, 0.03], voltage_kV=300.0)
BACKEND = os.environ.get("QEM_TEST_BACKEND")
pytestmark = pytest.mark.skipif(
    BACKEND is None, reason="Set QEM_TEST_BACKEND=mps or cuda for real GPU checks"
)


def _counts(source):
    result = source.decode_scan_range_device(0, int(np.prod(source.shape[:2])))
    if BACKEND == "cuda":
        return result.get().reshape(source.shape)
    try:
        return result.get().reshape(source.shape)
    finally:
        result.release()


def _calibration(loaded):
    """The flat calibration keys and axis calibration of a loaded acquisition."""
    flat = ("scan_sampling_A", "detector_sampling_mrad", "detector_sampling_inv_A", "detector_sampling",
            "detector_sampling_unit", "voltage_kV", "semiangle_mrad", "dwell_time_us", "camera_length_mm")
    return {key: loaded.metadata[key] for key in flat if key in loaded.metadata}, loaded.sampling, loaded.units


@pytest.mark.parametrize("name", ["u8-interval-boundary", "u16-multiple-chunks"])
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
        assert _calibration(original) == (CALIBRATION, (0.4, 0.6, 0.02, 0.03), ("angstrom", "angstrom", "1/angstrom", "1/angstrom"))
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
        assert restored.metadata["scientific_metadata"] == entry["scientific_metadata"]
        assert _calibration(restored) == (CALIBRATION, (0.4, 0.6, 0.02, 0.03), ("angstrom", "angstrom", "1/angstrom", "1/angstrom"))
    finally:
        restored.close()
