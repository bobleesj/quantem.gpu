"""Verify published prepared stacks, companions and simulation measurements."""

import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io.qem_validation import validate_qem


def _paths():
    root = os.environ.get("QEM_TCMEP_ROOT")
    if not root:
        return [None]
    collection = Path(root) / "TCMEP"
    paths = sorted(collection.rglob("data_roi*_dp.hdf5")) + sorted(
        (collection / "20241120_TCMEPsimul").glob("*.mat")
    )
    if len(paths) != 21:
        raise ValueError(
            f"Expected all 21 published acquisitions, found {len(paths)}; restore the complete collection."
        )
    return paths


@pytest.mark.parametrize(
    "path", _paths(), ids=lambda path: str(path) if path else "missing-data"
)
def test_published_acquisition_roundtrip(path, tmp_path, record_property):
    """Load native measurements, inspect DPs and export every value losslessly."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if path is None or backend not in {"mps", "cuda"}:
        pytest.skip(
            "Set QEM_TCMEP_ROOT and QEM_TEST_BACKEND on a physical accelerator."
        )
    root = Path(os.environ["QEM_TCMEP_ROOT"])
    receipt = json.loads((root / "local-download-receipt.json").read_text())
    assert receipt["archive_md5"] == "cd4b86347cd16c38aa4570ad87f02222"
    entry = next(
        item
        for item in receipt["files"]
        if item["path"] == path.relative_to(root).as_posix()
    )
    with path.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == entry["sha256"]
    info = io.inspect(path)
    assert info.ready
    rows, columns = info.scan_shape
    shape = (*info.scan_shape, *info.detector_shape)
    with (
        h5py.File(path, "r") as original,
        tempfile.TemporaryDirectory(dir=tmp_path) as scratch,
    ):
        data = original[info.metadata["dataset_path"]]

        def expected_row(row):
            return (
                data[row : row + 1]
                if data.ndim == 4
                else data[row * columns : (row + 1) * columns][None]
            )

        saved = Path(scratch) / "copy.qem"
        started = time.perf_counter()
        with io.load(path, backend=backend, verbose=False) as loaded:
            record_property("load_seconds", time.perf_counter() - started)
            assert loaded.representation.value == "encoded"
            assert loaded.metadata["load_timings"]["peak_ingest_bytes"] <= 32 << 20
            session = detector.prepare(loaded)
            for row in (0, rows // 2, rows - 1):
                actual = session.frame(row * columns).astype(data.dtype)
                np.testing.assert_array_equal(
                    actual.view(f"u{data.dtype.itemsize}"),
                    expected_row(row)[0, 0].view(f"u{data.dtype.itemsize}"),
                )
            mean = session.mean_dp()
            assert mean.shape == shape[2:] and np.isfinite(mean).all()
            mask = np.zeros(shape[2:], bool)
            mask[::2, ::2] = True
            image = session.masked_sum(mask)
            for row in (0, rows // 2, rows - 1):
                np.testing.assert_allclose(
                    image[row, 0],
                    expected_row(row)[0, 0][mask].sum(dtype=np.float64),
                    rtol=1e-5,
                    atol=1e-5,
                )
            io.save(saved, loaded)
            del session, mean, image
        assert validate_qem(saved)["integrity"] == "verified"
        with io.load(saved, backend=backend, verbose=False) as restored:
            for row in range(rows):
                actual = (
                    restored.read(scan_region=(row, row + 1, 0, columns))
                    .cpu()
                    .numpy()
                    .astype(data.dtype)
                )
                np.testing.assert_array_equal(
                    actual.view(f"u{data.dtype.itemsize}"),
                    expected_row(row).view(f"u{data.dtype.itemsize}"),
                )
            if path.suffix == ".hdf5":
                metadata = restored.metadata["scientific_metadata"]
                assert metadata["source_metadata"]["prepared_stack/params_backup"][
                    "voltage"
                ] == [300]
                assert (
                    metadata["electron_microscope"][
                        "electron_source/accelerating_voltage"
                    ]["value"]
                    == 300
                )
                assert (
                    metadata["electron_microscope"][
                        "illumination_system/semi_convergence_angle"
                    ]["value"]
                    == 25
                )
        record_property("values_compared", int(np.prod(shape)))
        record_property("differing_values", 0)
        record_property("qem_bytes", saved.stat().st_size)
    assert not saved.exists()
    gc.collect()
