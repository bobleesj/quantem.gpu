"""acquisition_identity is a portable content signature: it survives a move and tells acquisitions apart."""

import shutil
from pathlib import Path

import h5py
import numpy as np

from quantem.gpu.ssb.persistence import acquisition_identity, portable_source


def _master(folder: Path, counts: np.ndarray) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    master = folder / "scan_master.h5"
    with h5py.File(folder / "scan_data_000001.h5", "w") as handle:
        handle["entry/data/data"] = counts
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "/entry/data/data")
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = np.uint32(len(counts))
    return master


def test_identity_holds_content_not_location_and_survives_a_move(tmp_path):
    counts = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
    master = _master(tmp_path / "a", counts)
    record = {"method": "median", "pixel_count": 4, "applied": True, "largest_window_px": 3, "backend": "mps"}
    identity = acquisition_identity(master, scan_shape=(2, 3), hot_pixel_correction=record)
    assert identity["file_name"] == "scan_master.h5"
    assert [(item["file_name"], item["size"]) for item in identity["linked_files"]] == [
        ("scan_data_000001.h5", (tmp_path / "a" / "scan_data_000001.h5").stat().st_size)]
    assert (identity["scan_shape"], identity["detector_shape"]) == ([2, 3], [4, 4])
    # the backend that applied the fill does not change a count
    assert identity["hot_pixel_correction"] == {"method": "median", "pixel_count": 4, "applied": True,
                                                "largest_window_px": 3}
    text = repr(identity)
    assert str(tmp_path) not in text and str(Path.home()) not in text

    shutil.move(tmp_path / "a", tmp_path / "moved")
    assert acquisition_identity(tmp_path / "moved" / "scan_master.h5", scan_shape=(2, 3),
                                hot_pixel_correction=record) == identity

    # equal metadata and sizes: only the data file's first bytes tell the two scans apart
    other = _master(tmp_path / "b", counts + 1)
    assert acquisition_identity(other, scan_shape=(2, 3), hot_pixel_correction=record) != identity
    assert acquisition_identity(tmp_path / "moved" / "scan_master.h5", scan_shape=(2, 3),
                                hot_pixel_correction={**record, "method": "none", "applied": False}) != identity


def test_portable_source_is_relative_only_under_the_record_folder(tmp_path):
    (tmp_path / "data").mkdir()
    source = tmp_path / "data" / "scan_master.h5"
    source.write_bytes(b"x")
    assert portable_source(source, tmp_path) == {"file_name": "scan_master.h5", "relative_path": "data/scan_master.h5"}
    assert portable_source(source, tmp_path / "elsewhere") == {"file_name": "scan_master.h5", "relative_path": None}
    assert portable_source(None, tmp_path) is None
