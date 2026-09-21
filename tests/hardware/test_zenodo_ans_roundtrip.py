"""Full published acquisitions stay exact through bounded ANS/QEM workflows.

Set QEM_ZENODO_ROOT to the locally retained, checksum-verified collection.
Original acquisitions are never modified; each generated copy is disposable.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io.qem_validation import validate_qem

ACQUISITIONS = (
    [
        f"experiment/{folder}/scan_x256_y256.raw"
        for folder in ("Fig2-disloc", "extFig-disloc")
    ]
    + [
        f"simulation/Fig4_icom/nv80_ny180_nx180_df{defocus}.0_5pA1ms.npy"
        for defocus in range(-10, 171, 10)
    ]
    + ["simulation/Fig4_ptycho/nv128_ny120_nx120_5pA1ms.npy"]
)


@pytest.fixture(scope="module")
def collection():
    location = os.environ.get("QEM_ZENODO_ROOT")
    backend = os.environ.get("QEM_TEST_BACKEND")
    if not location or backend not in {"cuda", "mps"}:
        pytest.skip(
            "Set QEM_ZENODO_ROOT and QEM_TEST_BACKEND to verify the full collection."
        )
    root = Path(location)
    receipt = json.loads((root / "local-download-receipt.json").read_text())
    assert receipt["archive_md5"] == "fd1306838e19236a19aef478a718a957"
    assert receipt["archive_checksum_verified"] is True
    entries = {item["path"]: item for item in receipt["files"]}
    assert set(ACQUISITIONS).issubset(entries)
    return root, entries, backend


def _exact(actual, expected):
    if expected.dtype.kind == "f":
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("relative", ACQUISITIONS)
def test_original_ans_qem_full_value_parity(
    collection, relative, tmp_path, record_property
):
    """Save and reopen every detector value without uncompressed GPU residency."""
    root, entries, backend = collection
    path = root / relative
    entry = entries[relative]
    assert path.stat().st_size == entry["bytes"]
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == entry["sha256"]
    before = path.stat()
    mapped = (
        np.memmap(path, dtype="<f4", mode="r", shape=(256, 256, 130, 128))
        if path.suffix == ".raw"
        else np.load(path, mmap_mode="r", allow_pickle=False)
    )
    values = mapped[:, :, :128] if path.suffix == ".raw" else mapped
    rows, columns = values.shape[:2]
    try:
        with tempfile.TemporaryDirectory(
            dir=tmp_path, prefix="qem-roundtrip-"
        ) as directory:
            saved = Path(directory) / "copy.qem"
            started = time.perf_counter()
            with io.load(path, backend=backend, verbose=False) as loaded:
                record_property("load_seconds", time.perf_counter() - started)
                assert loaded.representation.value == "encoded"
                assert loaded.metadata["backend"] == backend
                assert "rans" in loaded.metadata["resident_profile"]
                assert loaded.metadata["load_timings"]["peak_ingest_bytes"] <= 32 << 20
                record_property("resident_profile", loaded.metadata["resident_profile"])
                record_property(
                    "resident_bytes", loaded.metadata["physical_resident_bytes"]
                )
                session = detector.prepare(loaded)
                points = [(0, 0), (rows // 2, columns // 2), (rows - 1, columns - 1)]
                for row, column in points:
                    _exact(session.frame(row * columns + column), values[row, column])
                mean = session.mean_dp()
                assert mean.shape == values.shape[2:] and np.isfinite(mean).all()
                mask = np.zeros(values.shape[2:], bool)
                mask[::2, ::2] = True
                image = session.masked_sum(mask)
                for row, column in points:
                    np.testing.assert_allclose(
                        image[row, column],
                        values[row, column][mask].sum(dtype=np.float64),
                        rtol=1e-5,
                        atol=1e-4,
                    )
                del session, mean, image
                para = (path.parent / "para.txt").read_text()
                metadata = dict(loaded.metadata)
                original = dict(metadata.get("source_metadata", {}))
                original.update(
                    {
                        "published_companion/para.txt": para,
                        "dataset_doi": "10.5281/zenodo.7464234",
                    }
                )
                metadata["source_metadata"] = original
                required = 2 * loaded.metadata["physical_resident_bytes"] + (5 << 30)
                assert (
                    shutil.disk_usage(directory).free > required
                ), "Need space for one temporary export plus a 5 GiB reserve."
                io.save(saved, loaded, metadata=metadata)
            validation = validate_qem(saved)
            assert validation["integrity"] == validation["codec_layout"] == "verified"
            record_property("qem_bytes", saved.stat().st_size)
            with io.load(saved, backend=backend, verbose=False) as restored:
                assert restored.representation.value == "encoded"
                assert restored.metadata["backend"] == backend
                source_metadata = restored.metadata["scientific_metadata"][
                    "source_metadata"
                ]
                assert source_metadata["published_companion/para.txt"] == para
                if path.suffix == ".raw":
                    (xml,) = path.parent.glob("*.xml")
                    assert source_metadata["empad_xml"] == xml.read_text()
                for row in range(rows):
                    block = restored.read(scan_region=(row, row + 1, 0, columns))
                    assert block.numel() * block.element_size() <= 32 << 20
                    _exact(block.cpu().numpy(), values[row : row + 1])
                    del block
            record_property("values_compared", int(values.size))
            record_property("differing_values", 0)
            record_property("source_sha256", entry["sha256"])
        assert not saved.exists()
        assert (before.st_size, before.st_mtime_ns) == (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
    finally:
        del values
        mapped._mmap.close()
        gc.collect()
