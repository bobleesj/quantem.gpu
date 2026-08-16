from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from quantem.gpu.remote.ssb_api import (
    SSBPayloadNotReady,
    SSBProtocolError,
    SSBProtocolService,
)


def _request(source: dict, *, generation: int = 7) -> dict:
    candidate = {
        "id": "raw-reference-fit",
        "calibration": {
            "scanSamplingRowAngstrom": {"value": 0.264, "unit": "angstrom", "origin": "validated_preset"},
            "scanSamplingColumnAngstrom": {"value": 0.264, "unit": "angstrom", "origin": "validated_preset"},
            "accelerationVoltageKilovolts": {"value": 300, "unit": "kV", "origin": "validated_preset"},
            "convergenceSemiangleMilliradians": {"value": 30, "unit": "mrad", "origin": "validated_preset"},
            "scanRotationDegrees": {"value": 158.8827, "unit": "degree", "origin": "validated_preset"},
            "c10Nanometers": {"value": 73.1336, "unit": "nm", "origin": "validated_preset"},
            "c12Nanometers": {"value": 14.1409, "unit": "nm", "origin": "validated_preset"},
            "phi12Radians": {"value": 0.474155, "unit": "radian", "origin": "validated_preset"},
        },
        "evidenceSHA256": "a" * 64,
        "objective": "exact_full_bf_phase_variance",
        "loss": 0.044,
    }
    return {
        "contractVersion": "live4dstem.ssb/v0.1",
        "algorithmVersion": "quantem.gpu.SSB/v0.1",
        "jobID": "5fce107b-c5fa-45fe-a6db-2096171049bb",
        "datasetGeneration": generation,
        "source": {"datasetID": "BTO_18", "datasetSchema": "test", **source},
        "scanShape": {"rows": 2, "columns": 2},
        "detectorShape": {"rows": 2, "columns": 2},
        "precision": {"sourceDType": "uint8", "realDType": "float32", "complexDType": "complex64"},
        "calibration": {
            "schemaVersion": "live4dstem.dataset/v0.1",
            "sourceIdentitySHA256": source["sourceIdentitySHA256"],
            "resolution": {"state": "valid", "calibration": candidate},
        },
        "selection": {
            "policy": "full_active",
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            "detectorBin": 1,
            "scanCrop": None,
        },
        "backend": {"kind": "remote_cuda", "profile_id": "mjgoat", "gpu_index": 0},
        "computeLoss": True,
        "measureWarm": False,
    }


def _service(tmp_path: Path) -> tuple[SSBProtocolService, dict]:
    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    holder: dict = {}

    def runner(_source, gpu, request):
        holder.update(gpu=gpu, request=request)
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            "implementationRevision": "test",
            "timings": {"firstReconstructSeconds": 0.1},
        }

    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: "Test CUDA",
        runner=runner,
    )
    return service, holder


def test_request_result_and_generation_bound_payload(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)

    result = service.reconstruct(request)
    payload, descriptor = service.payload(request["jobID"], 7)

    assert holder["gpu"] == 0
    assert result["executedDevice"]["backend"] == "cuda"
    assert result["executedBrightFieldCounts"] == {"logical": 4, "active": 3}
    assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert descriptor["byteCount"] == 16
    assert result["phasePayload"]["datasetGeneration"] == 7
    with pytest.raises(SSBProtocolError, match="No validated"):
        service.payload(request["jobID"], 8)


def test_source_identity_matches_live4dstem_dataset_v0_1_digest(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    expected = hashlib.sha256()
    expected.update(b"live4dstem.dataset/v0.1\0")
    expected.update(identity["masterSHA256"].encode())
    for value in identity["orderedMemberSHA256"]:
        expected.update(b"\0")
        expected.update(value.encode())

    assert identity["sourceIdentitySHA256"] == expected.hexdigest()


def test_unresolved_calibration_and_mps_selection_are_rejected(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)
    request["calibration"]["resolution"] = {"state": "unresolved", "candidates": []}
    with pytest.raises(SSBProtocolError, match="not resolved"):
        service.reconstruct(request)

    request = _request(identity)
    request["backend"] = {"kind": "local_mps"}
    with pytest.raises(SSBProtocolError, match="remote_cuda"):
        service.reconstruct(request)


def test_explicit_local_mps_service_never_accepts_cuda_or_reports_fallback(tmp_path):
    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    calls = []

    def runner(_source, gpu, request):
        calls.append((gpu, request["backend"]["kind"]))
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            "implementationRevision": "test",
            "timings": {"firstReconstructSeconds": 0.1},
        }

    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: pytest.fail("local MPS queried CUDA devices"),
        device_name=lambda _gpu: "Apple MPS",
        runner=runner,
        backend_kind="local_mps",
    )
    identity = service.source_identity(str(master))
    local_request = _request(identity)
    local_request["backend"] = {"kind": "local_mps"}

    result = service.reconstruct(local_request)

    assert calls == [(None, "local_mps")]
    assert result["requestedBackend"] == {"kind": "local_mps"}
    assert result["executedDevice"]["backend"] == "mps"
    capability = service.advertised_capability("local_mps")
    assert capability["backendKind"] == "local_mps"
    assert capability["implicitFallback"] is False

    with pytest.raises(SSBProtocolError, match="local_mps"):
        service.reconstruct(_request(identity))
    assert calls == [(None, "local_mps")]


def test_source_hash_mismatch_is_rejected_before_runner(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)
    request["source"]["masterSHA256"] = "0" * 64

    with pytest.raises(SSBProtocolError, match="masterSHA256"):
        service.reconstruct(request)
    assert holder == {}


def test_logical_and_active_brightfield_counts_are_validated_separately(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))

    wrong_logical = _request(identity)
    wrong_logical["selection"]["logicalBrightFieldCount"] = 5
    with pytest.raises(SSBProtocolError, match="logical BF count changed"):
        service.reconstruct(wrong_logical)

    wrong_active = _request(identity)
    wrong_active["selection"]["activeBrightFieldCount"] = 2
    with pytest.raises(SSBProtocolError, match="active BF count changed"):
        service.reconstruct(wrong_active)


def _wait_for_terminal(service, job_id, generation=7):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = service.job_snapshot(job_id, generation)
        if snapshot["state"] in {"completed", "cancelled", "failed"}:
            return snapshot
        time.sleep(0.005)
    raise AssertionError("SSB job did not reach a terminal state")


def test_async_submit_is_idempotent_and_reconnects_to_same_result(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)

    accepted = service.submit(request)
    duplicate = service.submit(request)
    completed = _wait_for_terminal(service, request["jobID"])

    assert accepted["state"] == "accepted"
    assert duplicate["requestSHA256"] == accepted["requestSHA256"]
    assert completed["state"] == "completed"
    assert completed["sequence"] > accepted["sequence"]
    assert completed["result"]["phase"]["sha256"]
    assert service.job_snapshot(request["jobID"], 7) == completed

    conflicting = _request(identity)
    conflicting["jobID"] = request["jobID"]
    conflicting["computeLoss"] = False
    with pytest.raises(SSBProtocolError, match="different request digest"):
        service.submit(conflicting)


def test_cancel_during_runner_waits_for_stage_boundary_and_discards_payload(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def runner(_source, _gpu, _request):
        entered.set()
        assert release.wait(timeout=2)
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            "implementationRevision": "test",
            "timings": {"firstReconstructSeconds": 0.1},
        }

    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: "Test CUDA",
        runner=runner,
    )
    request = _request(service.source_identity(str(master)))

    service.submit(request)
    assert entered.wait(timeout=2)
    requested = service.cancel_job(request["jobID"], 7)
    assert requested["state"] == "cancel_requested"
    release.set()
    cancelled = _wait_for_terminal(service, request["jobID"])

    assert cancelled["state"] == "cancelled"
    with pytest.raises(SSBPayloadNotReady):
        service.payload(request["jobID"], 7)


def test_async_http_status_cancel_and_payload_gate(tmp_path):
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import create_app

    entered = threading.Event()
    release = threading.Event()

    def runner(_source, _gpu, _request):
        entered.set()
        assert release.wait(timeout=2)
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            "implementationRevision": "test",
            "timings": {"firstReconstructSeconds": 0.1},
        }

    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: "Test CUDA",
        runner=runner,
    )
    request = _request(service.source_identity(str(master)))
    client = TestClient(create_app(tmp_path, ssb_service=service))

    accepted = client.post("/api/ssb/jobs", json=request)
    assert accepted.status_code == 202
    assert entered.wait(timeout=2)
    snapshot = client.get(
        f"/api/ssb/jobs/{request['jobID']}", params={"generation": 7}
    )
    assert snapshot.status_code == 200
    assert snapshot.json()["state"] == "reconstructing_first"
    assert client.get(
        f"/api/ssb/jobs/{request['jobID']}/phase", params={"generation": 7}
    ).status_code == 425

    cancelled = client.delete(
        f"/api/ssb/jobs/{request['jobID']}", params={"generation": 7}
    )
    assert cancelled.status_code == 202
    assert cancelled.json()["state"] == "cancel_requested"
    release.set()
    assert _wait_for_terminal(service, request["jobID"])["state"] == "cancelled"
