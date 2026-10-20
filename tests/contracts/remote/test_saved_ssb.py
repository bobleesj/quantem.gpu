"""Saved phase transfer preserves source binding and the exact float payload."""

import base64
import hashlib
import json

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from quantem.gpu.formats.ssb_phase import SCHEMA, acquisition_identity
from quantem.gpu.remote.saved_ssb import SavedSSBResults
from quantem.gpu.remote import create_app


def test_saved_phase_transfer_reuse_rename_and_source_change(tmp_path, monkeypatch):
    master = tmp_path / "sample_master.h5"
    shard = tmp_path / "counts.h5"
    with h5py.File(shard, "w") as handle:
        handle["counts"] = np.arange(16, dtype=np.uint16).reshape(4, 2, 2)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(shard.name, "counts")
    identity = acquisition_identity(master)
    result_folder = tmp_path / "live/screen/renamed-result"
    result_folder.mkdir(parents=True)
    phase = np.array([[-0.0, -1.25], [3.5, 1e-20]], dtype="<f4")
    np.save(result_folder / "ssb.npy", phase, allow_pickle=False)
    record = dict(
        schema=SCHEMA,
        phase_file="ssb.npy",
        shape=[2, 2],
        phase_encoding="float32-le-row-major",
        phase_unit="rad",
        source_identity=identity,
        phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest(),
        calibration=dict(voltage_kV=1.0, semiangle_mrad=1.0, scan_sampling_A=[1.0, 1.0],
                         detector_sampling_mrad=[1.0, 1.0], bright_field_center_px=[1.0, 1.0]),
        aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
        scan_detector_rotation_deg=0.0,
        scan_detector_rotation_flipped=False,
        provenance={"producer": "test fixture"},
        run_metadata={"notes": "Exact phase"},
    )
    (result_folder / "ssb.json").write_text(json.dumps(record))
    app = create_app(tmp_path)
    monkeypatch.setattr(app.state.browse_service.catalog, "resolve_master", lambda session, file: master)
    client = TestClient(app)
    response = client.get(
        "/api/ssb/saved-results", params={"session": ".", "file": master.name}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert base64.b64decode(result["phase"]) == phase.tobytes()
    assert json.loads(base64.b64decode(result["run_metadata"])) == record["run_metadata"]
    assert response.json()["schema"] == "quantem.ssb-phase-results/2"
    saved = SavedSSBResults(tmp_path)
    assert saved.read(master)["source_identity"] == identity
    monkeypatch.setattr(
        "quantem.gpu.remote.saved_ssb.acquisition_identity",
        lambda path: pytest.fail("unchanged acquisition was hashed again"),
    )
    assert saved.read(master)["results"][0]["phase_sha256"] == record["phase_sha256"]
    monkeypatch.undo()
    moved = master.with_name("renamed_master.h5")
    master.rename(moved)
    assert saved.read(moved)["source_identity"] == identity
    with h5py.File(shard, "r+") as handle:
        handle["counts"][0, 0, 0] = 99
    assert saved.read(moved)["results"] == []
