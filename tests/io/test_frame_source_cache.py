"""Reopening a changed detector file must not reuse obsolete chunk offsets."""

import importlib

import h5py
import numpy as np


def test_reopen_changed_detector_chunk_refreshes_index(tmp_path, monkeypatch):
    loader = importlib.import_module("quantem.gpu.io.load")
    monkeypatch.setenv(loader._FRAME_SOURCE_CACHE_ENV, "")
    master = tmp_path / "scan_master.h5"
    chunk = tmp_path / "detector.h5"
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(chunk.name, "/entry/data/data")

    def write(values):
        with h5py.File(chunk, "w") as handle:
            handle.create_dataset("entry/data/data", data=values,
                                  chunks=(1, 8, 8), compression="gzip")

    write(np.zeros((4, 8, 8), np.uint16))
    first = loader._get_master_frame_sources(str(master), ["data_000001"], apply_mask=False)
    assert loader._get_master_frame_sources(str(master), ["data_000001"], apply_mask=False) is first
    write(np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8))
    second = loader._get_master_frame_sources(str(master), ["data_000001"], apply_mask=False)
    assert second is not first and second[0][0]["n_frames"] == 6
