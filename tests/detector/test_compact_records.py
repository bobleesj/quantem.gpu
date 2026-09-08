"""Read prepared scientific records independently of accelerator allocation."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu._compact.layout import read_record, record_views


def test_exact_components_survive_a_record_read(tmp_path):
    """Counts and bit-packed words survive a bounded reusable host slot."""
    counts = np.array([0, 1, 65535, 123456789, 2**32 - 1], np.uint32)
    widths = np.array([0, 1, 17, 32], np.uint8)
    row = {
        "file_offset": 4096,
        "record_bytes": 4096,
        "components": [
            {"name": "total", "shape": [5], "dtype": "<u4", "offset": 0, "nbytes": 20},
            {
                "name": "coarse_widths",
                "shape": [4],
                "dtype": "|u1",
                "offset": 64,
                "nbytes": 4,
            },
        ],
    }
    payload = bytearray(8192)
    payload[4096:4116] = counts.tobytes()
    payload[4160:4164] = widths.tobytes()
    path = tmp_path / "data-0.bin"
    path.write_bytes(payload)
    slot = np.empty(4096, np.uint8)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        read_record(descriptor, row, slot)
    finally:
        os.close(descriptor)
    products = record_views(slot, row)
    np.testing.assert_array_equal(products["total"], counts)
    np.testing.assert_array_equal(products["coarse_widths"], widths)
    assert np.shares_memory(products["total"], slot)


def test_prepared_real_record_matches_published_hash():
    """A real prepared record and its metadata match the completed manifest."""
    selected = os.environ.get("QUANTEM_PREPARED_SERIES")
    if not selected:
        pytest.skip("Set QUANTEM_PREPARED_SERIES to run the real prepared-file check.")
    from quantem.gpu._compact.metadata import read_metadata

    path = Path(selected)
    manifest, metadata = read_metadata(path)
    row = manifest["layout"]["chunks"][0]
    slot = np.empty(row["record_bytes"], np.uint8)
    descriptor = os.open(
        path / manifest["layout"]["files"][row["shard"]]["name"], os.O_RDONLY
    )
    try:
        read_record(descriptor, row, slot)
    finally:
        os.close(descriptor)
    assert hashlib.sha256(slot).hexdigest() == row["sha256"]
    assert np.array_equal(metadata["planner__valid"], ~metadata["planner__hardware"])


def test_existing_prepared_metadata_survives_canonical_label_migration(tmp_path):
    """An archived source opens with old or canonical labels and identical metadata."""
    selected = os.environ.get("QUANTEM_PREPARED_SERIES")
    if not selected:
        pytest.skip("Set QUANTEM_PREPARED_SERIES to check archived format compatibility.")
    from quantem.gpu._compact import FORMAT, LAYOUT_FORMAT
    from quantem.gpu._compact.metadata import read_metadata

    path = Path(selected)
    manifest, original = read_metadata(path)
    manifest["format"] = FORMAT
    manifest["layout"]["format"] = LAYOUT_FORMAT
    state_file = manifest["global_state_file"]
    (tmp_path / state_file).symlink_to(path / state_file)
    (tmp_path / "checkpoint.json").write_text(json.dumps(manifest))
    _, reopened = read_metadata(tmp_path)
    assert reopened.keys() == original.keys()
    for name in original:
        np.testing.assert_array_equal(reopened[name], original[name])
