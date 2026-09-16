"""Saved phase transfer preserves source binding and the exact float payload."""

import base64
import hashlib
import json

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from quantem.gpu.io.ssb_result import acquisition_identity
from quantem.gpu.remote.saved_ssb import SavedSSBResults
from quantem.gpu.remote.server import BrowseService, create_app


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
        format="live.ssb",
        schemaVersion=1,
        phaseFile="ssb.npy",
        rows=2,
        columns=2,
        phaseEncoding="float32-le-row-major",
        phaseUnits="rad",
        sourceIdentity=identity,
        phaseSHA256=hashlib.sha256(phase.tobytes()).hexdigest(),
        calibration={
            key: 1.0
            for key in (
                "beamEnergyKeV",
                "semiangleMrad",
                "scanStepRowAngstroms",
                "scanStepColumnAngstroms",
                "detectorStepRowMrad",
                "detectorStepColumnMrad",
                "centerRow",
                "centerColumn",
            )
        },
        c10Nanometers=0,
        c12Nanometers=0,
        phi12Radians=0,
        rotationDegrees=0,
        provenance={"producer": "test fixture"},
        runMetadata={"notes": "Exact phase"},
    )
    (result_folder / "ssb.json").write_text(json.dumps(record))
    service = BrowseService(tmp_path, initialize_cuda=False)
    monkeypatch.setattr(service, "resolve_master", lambda session, file: master)
    client = TestClient(create_app(tmp_path, service=service))
    response = client.get(
        "/api/ssb/saved-results", params={"session": ".", "file": master.name}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert base64.b64decode(result["phase"]) == phase.tobytes()
    assert json.loads(base64.b64decode(result["runMetadata"])) == record["runMetadata"]
    saved = SavedSSBResults(tmp_path)
    assert saved.read(master)["sourceIdentity"] == identity
    monkeypatch.setattr(
        "quantem.gpu.remote.saved_ssb.acquisition_identity",
        lambda path: pytest.fail("unchanged acquisition was hashed again"),
    )
    assert saved.read(master)["results"][0]["phaseSHA256"] == record["phaseSHA256"]
    monkeypatch.undo()
    moved = master.with_name("renamed_master.h5")
    master.rename(moved)
    assert saved.read(moved)["sourceIdentity"] == identity
    with h5py.File(shard, "r+") as handle:
        handle["counts"][0, 0, 0] = 99
    assert saved.read(moved)["results"] == []
