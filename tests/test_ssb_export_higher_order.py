"""Portable screening results retain the full fitted model on local volumes."""

import errno
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import h5py
import numpy as np
import pytest

from quantem.gpu.formats import publish
from quantem.gpu.formats.ssb_phase import _publish, export_result, read_result
from quantem.gpu.direct_ptycho.persistence import FIT_SCHEMA
from quantem.gpu.direct_ptycho.units import HIGHER_ORDER


def _deny_hardlinks(*args):
    raise OSError(errno.EPERM, "filesystem does not permit hardlinks")


@pytest.mark.parametrize("separate_tilt", [False, True])
@pytest.mark.parametrize("hardlinks", [True, False])
def test_fifth_order_screening_export_roundtrip(tmp_path, monkeypatch, separate_tilt, hardlinks):
    """Export, reopen and repeat a fifth-order tilt fit without losing coefficients or phase bits."""
    if not hardlinks:
        if sys.platform not in {"darwin", "linux"}:
            pytest.skip("native exclusive rename requires macOS or Linux")
        monkeypatch.setattr(publish.os, "link", _deny_hardlinks)
    master = tmp_path / "scan_master.h5"
    with h5py.File(tmp_path / "scan_data.h5", "w") as handle:
        handle["data"] = np.arange(32, dtype=np.uint16).reshape(2, 4, 4)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data.h5", "/data")
    primary = {"C10_nm": -8.25, "C12_nm": 1.75, "phi12_deg": 21.5}
    aberrations = {**primary, **{key: float(index + 1) for index, key in enumerate(HIGHER_ORDER)}}
    fit = {
        "schema": FIT_SCHEMA, "state": "complete", "voltage_kV": 300.0,
        "semiangle_mrad": 25.0, "scan_sampling_A": [0.4, 0.4],
        "detector_sampling_mrad": [0.5, 0.5], "bright_field_center_px": [2.0, 2.0],
        "bright_field_radius_px": 1.5, "scan_detector_rotation_deg": 191.0,
        "scan_detector_rotation_flipped": False, "aberrations": aberrations,
        "sample_tilt_mrad": [-0.75, 1.25], "depth_spread_nm": 16.5,
    }
    run = tmp_path / "run"
    run.mkdir()
    phase = np.array([[-0.0, -1.25], [3.5, 1e-20]], dtype=np.float32)
    np.save(run / "ssb_phase.npy", phase if not separate_tilt else np.zeros_like(phase))
    computed = {"ssb": fit}
    if separate_tilt:
        computed = {"ssb": {**fit, "aberrations": primary}, "ssb_tilt": fit}
        np.save(run / "ssb_phase_tilt.npy", phase)
    (run / "config.json").write_text(json.dumps({"computed": computed}))
    destination = tmp_path / "results" / "ssb.json"

    exported = export_result(run, master, destination)
    record, restored = read_result(exported)
    assert record["aberrations"] == aberrations
    assert record["sample_tilt_mrad"] == fit["sample_tilt_mrad"]
    assert record["depth_spread_nm"] == fit["depth_spread_nm"]
    assert restored.tobytes() == phase.tobytes()
    saved = {path.name: path.read_bytes() for path in destination.parent.iterdir()}
    assert export_result(run, master, destination) == exported
    assert {path.name: path.read_bytes() for path in destination.parent.iterdir()} == saved

    # A new calibration may share phase bytes but cannot replace an earlier manifest.
    fit["aberrations"] = {**aberrations, "C56_nm": aberrations["C56_nm"] + 1.0}
    (run / "config.json").write_text(json.dumps({"computed": computed}))
    with pytest.raises(ValueError, match="Choose a new name"):
        export_result(run, master, destination)
    assert {path.name: path.read_bytes() for path in destination.parent.iterdir()} == saved
    new = export_result(run, master, destination.with_name("next.json"))
    assert read_result(new)[0]["aberrations"] == fit["aberrations"]


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="native exclusive rename")
def test_concurrent_exports_keep_one_complete_result_without_hardlinks(tmp_path, monkeypatch):
    """Two writers cannot replace each other's complete output on a hardlink-free volume."""
    monkeypatch.setattr(publish.os, "link", _deny_hardlinks)
    destination = tmp_path / "phase.npy"
    ready = Barrier(2)
    contents = (b"first phase" * 1000, b"second phase" * 1000)

    def save(payload):
        ready.wait()
        try:
            _publish(destination, payload)
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as writers:
        accepted = list(writers.map(save, contents))
    assert sum(accepted) == 1
    assert destination.read_bytes() == contents[accepted.index(True)]
    assert list(tmp_path.iterdir()) == [destination]
