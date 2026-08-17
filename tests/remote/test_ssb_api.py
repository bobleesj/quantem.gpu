from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from quantem.gpu.remote.ssb_api import (
    ALGORITHM_VERSION,
    PREPARE_CONTRACT_VERSION,
    SSBPayloadUnavailable,
    SSBProtocolError,
    SSBProtocolService,
    count_audit_sha256,
    selection_descriptor_sha256,
)


def _inspection(*_args, **_kwargs):
    return SimpleNamespace(
        ready=True,
        reason="",
        action="",
        dtype="uint16",
        scan_shape=(2, 2),
        detector_shape=(2, 2),
    )


def _execution_evidence(request: dict) -> dict:
    return {
        "detectorSamplingMilliradians": (1.090909, 1.090909),
        "precision": request["precision"],
    }


def _bind_prepared_selection(
    request: dict, *, implementation_revision: str = "test"
) -> dict:
    candidate = request["calibration"]["resolution"]["calibration"]
    source = request["source"]
    descriptor = {
        "contractVersion": PREPARE_CONTRACT_VERSION,
        "productStatus": "beta",
        "workflow": "direct_ptychography",
        "algorithmVersion": ALGORITHM_VERSION,
        "implementationRevision": implementation_revision,
        "source": {
            key: source[key]
            for key in (
                "masterPath",
                "masterSHA256",
                "orderedMemberSHA256",
                "sourceIdentitySHA256",
            )
        },
        "scanShape": request["scanShape"],
        "detectorShape": request["detectorShape"],
        "precision": request["precision"],
        "calibrationBinding": {
            "sourceIdentitySHA256": source["sourceIdentitySHA256"],
            "candidateID": candidate["id"],
            "evidenceSHA256": candidate["evidenceSHA256"],
            "calibrationSHA256": hashlib.sha256(
                json.dumps(candidate, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
        },
        "detectorSamplingMilliradians": {
            "row": candidate["calibration"]["detectorSamplingRowMilliradiansPerPixel"][
                "value"
            ],
            "column": candidate["calibration"][
                "detectorSamplingColumnMilliradiansPerPixel"
            ]["value"],
        },
        "selection": request["selection"],
    }
    descriptor["selectionSHA256"] = selection_descriptor_sha256(descriptor)
    request["preparedSelection"] = descriptor
    return request


def _request(
    source: dict, *, generation: int = 7, implementation_revision: str = "test"
) -> dict:
    audited_element_count = 2 * 2 * 2 * 2
    audit_evidence = count_audit_sha256(
        source["sourceIdentitySHA256"],
        "uint16",
        audited_element_count,
        53,
        0,
    )
    candidate = {
        "id": "raw-reference-fit",
        "calibration": {
            "scanSamplingRowAngstrom": {
                "value": 0.264,
                "unit": "angstrom",
                "origin": "validated_preset",
            },
            "scanSamplingColumnAngstrom": {
                "value": 0.264,
                "unit": "angstrom",
                "origin": "validated_preset",
            },
            "detectorSamplingRowMilliradiansPerPixel": {
                "value": 1.090909,
                "unit": "mrad/pixel",
                "origin": "validated_preset",
            },
            "detectorSamplingColumnMilliradiansPerPixel": {
                "value": 1.090909,
                "unit": "mrad/pixel",
                "origin": "validated_preset",
            },
            "accelerationVoltageKilovolts": {
                "value": 300,
                "unit": "kV",
                "origin": "validated_preset",
            },
            "convergenceSemiangleMilliradians": {
                "value": 30,
                "unit": "mrad",
                "origin": "validated_preset",
            },
            "scanRotationDegrees": {
                "value": 158.8827,
                "unit": "degree",
                "origin": "validated_preset",
            },
            "c10Nanometers": {
                "value": 73.1336,
                "unit": "nm",
                "origin": "validated_preset",
            },
            "c12Nanometers": {
                "value": 14.1409,
                "unit": "nm",
                "origin": "validated_preset",
            },
            "phi12Radians": {
                "value": 0.474155,
                "unit": "radian",
                "origin": "validated_preset",
            },
        },
        "evidenceSHA256": "a" * 64,
        "objective": "exact_full_bf_phase_variance",
        "loss": 0.044,
    }
    request = {
        "contractVersion": "live4dstem.ssb/v0.2",
        "algorithmVersion": "quantem.gpu.SSB/v0.1",
        "jobID": "5fce107b-c5fa-45fe-a6db-2096171049bb",
        "datasetGeneration": generation,
        "source": {"datasetID": "BTO_18", "datasetSchema": "test", **source},
        "scanShape": {"rows": 2, "columns": 2},
        "detectorShape": {"rows": 2, "columns": 2},
        "precision": {
            "nativeSourceDType": "uint16",
            "workingSourceDType": "uint8",
            "losslessWorkingDTypeAudit": {
                "scope": "complete_native_detector_source",
                "sourceIdentitySHA256": source["sourceIdentitySHA256"],
                "auditedElementCount": audited_element_count,
                "maximumCount": 53,
                "workingMaximum": 255,
                "countsAboveWorkingMaximum": 0,
                "evidenceSHA256": audit_evidence,
            },
            "realDType": "float32",
            "complexDType": "complex64",
        },
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
    return _bind_prepared_selection(
        request, implementation_revision=implementation_revision
    )


def _prepare_request(source: dict, *, backend: dict | None = None) -> dict:
    reconstruction = _request(source)
    return {
        "contractVersion": PREPARE_CONTRACT_VERSION,
        "masterPath": source["masterPath"],
        "expectedSourceIdentitySHA256": source["sourceIdentitySHA256"],
        "calibration": reconstruction["calibration"],
        "selection": {
            "policy": "full_active",
            "detectorBin": 1,
            "scanCrop": None,
        },
        "precision": reconstruction["precision"],
        "backend": backend
        or {"kind": "remote_cuda", "profile_id": "mjgoat", "gpu_index": 0},
    }


def _service(
    tmp_path: Path,
    *,
    prepared_sampling: tuple[float, float] = (1.090909, 1.090909),
) -> tuple[SSBProtocolService, dict]:
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
            "detectorSamplingMilliradians": (1.090909, 1.090909),
            "precision": request["precision"],
            "implementationRevision": "test",
            "timings": {"firstReconstructSeconds": 0.1},
        }

    def preparer(_source, gpu, request):
        holder.setdefault("prepareCalls", []).append(
            {"gpu": gpu, "backend": request["backend"]["kind"]}
        )
        return {
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            **_execution_evidence(request),
            "detectorSamplingMilliradians": prepared_sampling,
        }

    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: "Test CUDA",
        runner=runner,
        preparer=preparer,
        source_inspector=_inspection,
        implementation_revision="test",
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
    assert result["executedDevice"]["implementationRevision"] == "test"
    assert result["executedBrightFieldCounts"] == {"logical": 4, "active": 3}
    assert result["executedDetectorSamplingMilliradians"] == {
        "row": 1.090909,
        "column": 1.090909,
    }
    assert result["executedPrecision"] == request["precision"]
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


def test_prepare_returns_source_bound_selection_and_roundtrips(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))

    descriptor = service.prepare(_prepare_request(identity))

    assert descriptor["contractVersion"] == PREPARE_CONTRACT_VERSION
    assert descriptor["productStatus"] == "beta"
    assert descriptor["workflow"] == "direct_ptychography"
    assert descriptor["source"] == identity
    assert descriptor["scanShape"] == {"rows": 2, "columns": 2}
    assert descriptor["detectorShape"] == {"rows": 2, "columns": 2}
    assert descriptor["precision"]["nativeSourceDType"] == "uint16"
    assert descriptor["precision"]["workingSourceDType"] == "uint8"
    assert (
        descriptor["precision"]["losslessWorkingDTypeAudit"]["auditedElementCount"]
        == 16
    )
    assert descriptor["detectorSamplingMilliradians"] == {
        "row": 1.090909,
        "column": 1.090909,
    }
    assert descriptor["selection"] == {
        "policy": "full_active",
        "logicalBrightFieldCount": 4,
        "activeBrightFieldCount": 3,
        "detectorBin": 1,
        "scanCrop": None,
    }
    assert descriptor["selectionSHA256"] == selection_descriptor_sha256(descriptor)
    assert holder["prepareCalls"] == [{"gpu": 0, "backend": "remote_cuda"}]

    reconstruction = _request(identity)
    reconstruction["preparedSelection"] = descriptor
    result = service.reconstruct(reconstruction)
    assert result["selection"] == descriptor["selection"]


def test_prepare_descriptor_keeps_exact_calibration_after_tolerated_backend_rounding(
    tmp_path,
):
    requested = 1.090909
    service, _ = _service(
        tmp_path,
        prepared_sampling=(requested + 5e-10, requested - 5e-10),
    )
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))

    descriptor = service.prepare(_prepare_request(identity))

    assert descriptor["detectorSamplingMilliradians"] == {
        "row": requested,
        "column": requested,
    }
    reconstruction = _request(identity)
    reconstruction["preparedSelection"] = descriptor
    service.reconstruct(reconstruction)


def test_prepare_rejects_detector_sampling_outside_tolerance(tmp_path):
    requested = 1.090909
    service, _ = _service(
        tmp_path,
        prepared_sampling=(requested + 2e-9, requested),
    )
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))

    with pytest.raises(SSBProtocolError, match="prepared detector sampling differs"):
        service.prepare(_prepare_request(identity))


def test_prepare_http_capability_and_unresolved_calibration_are_explicit(tmp_path):
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import BrowseService, create_app

    service, holder = _service(tmp_path)
    browse = BrowseService(tmp_path, initialize_cuda=False)
    client = TestClient(create_app(tmp_path, service=browse, ssb_service=service))
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    prepared = client.post("/api/ssb/prepare", json=_prepare_request(identity))
    request = _prepare_request(identity)
    request["calibration"]["resolution"] = {"state": "unresolved", "candidates": []}

    response = client.post("/api/ssb/prepare", json=request)
    capability = client.get("/api/browse/capabilities").json()["features"]["ssb"]

    assert prepared.status_code == 200
    assert prepared.json()["selection"]["activeBrightFieldCount"] == 3
    assert response.status_code == 409
    assert "not resolved" in response.json()["detail"]
    assert holder["prepareCalls"] == [{"gpu": 0, "backend": "remote_cuda"}]
    assert capability["preparation"] == {
        "contractVersion": PREPARE_CONTRACT_VERSION,
        "endpoint": "/api/ssb/prepare",
        "method": "POST",
        "productStatus": "beta",
        "workflow": "direct_ptychography",
        "selectionPolicy": "full_active",
        "detectorBin": 1,
        "scanCropSupported": False,
        "workingPrecision": "lossless_uint8_complete_source_audit",
        "requiresExplicitBackend": True,
        "implicitFallback": False,
    }
    assert capability["fixedReconstructionControls"] == {
        "aberrations": [
            {"name": "C10", "unit": "nm"},
            {"name": "C12", "unit": "nm"},
            {"name": "phi12", "unit": "radian"},
        ],
        "scanRotation": {"name": "scanRotation", "unit": "degree"},
        "higherOrderAberrations": False,
        "preparedSessionWarmReconstruct": False,
        "unavailableReason": (
            "The service does not retain a source-bound SSB session between requests; "
            "measureWarm is a within-job benchmark only."
        ),
    }


def test_prepare_descriptor_rejects_tampering_and_stale_source_or_calibration(
    tmp_path,
):
    service, _ = _service(tmp_path)
    master = tmp_path / "BTO_18_master.h5"
    identity = service.source_identity(str(master))
    descriptor = service.prepare(_prepare_request(identity))

    missing = _request(identity)
    missing.pop("preparedSelection")
    with pytest.raises(SSBProtocolError, match="requires a server-prepared"):
        service.reconstruct(missing)

    tampered = _request(identity)
    tampered["preparedSelection"] = json.loads(json.dumps(descriptor))
    tampered["preparedSelection"]["selection"]["activeBrightFieldCount"] = 2
    with pytest.raises(SSBProtocolError, match="digest is invalid"):
        service.reconstruct(tampered)

    changed_calibration = _request(identity)
    changed_calibration["preparedSelection"] = descriptor
    changed_calibration["calibration"]["resolution"]["calibration"]["calibration"][
        "c10Nanometers"
    ]["value"] = 74.0
    with pytest.raises(SSBProtocolError, match="calibration changed"):
        service.reconstruct(changed_calibration)

    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"changed shard")
    stale_source = _request(identity)
    stale_source["preparedSelection"] = descriptor
    with pytest.raises(SSBProtocolError, match="source identity mismatch"):
        service.reconstruct(stale_source)


def test_prepare_never_falls_back_to_an_alternate_backend(tmp_path):
    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    calls = []
    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: pytest.fail("local MPS queried CUDA devices"),
        device_name=lambda _gpu: "Apple MPS",
        preparer=lambda *_args: calls.append("mps"),
        source_inspector=_inspection,
        backend_kind="local_mps",
        implementation_revision="test",
    )
    identity = service.source_identity(str(master))

    with pytest.raises(SSBProtocolError, match="local_mps"):
        service.prepare(_prepare_request(identity))
    assert calls == []


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
            **_execution_evidence(request),
            "implementationRevision": "test",
            "timings": {"firstReconstructSeconds": 0.1},
        }

    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: pytest.fail("local MPS queried CUDA devices"),
        device_name=lambda _gpu: "Apple MPS",
        runner=runner,
        source_inspector=_inspection,
        backend_kind="local_mps",
        implementation_revision="test",
    )
    identity = service.source_identity(str(master))
    local_request = _request(identity)
    local_request["backend"] = {"kind": "local_mps"}

    result = service.reconstruct(local_request)

    assert calls == [(None, "local_mps")]
    assert result["requestedBackend"] == {"kind": "local_mps"}
    assert result["executedDevice"]["backend"] == "mps"
    capability = service.capability()
    assert capability["backendKind"] == "local_mps"
    assert capability["implicitFallback"] is False
    assert capability["ready"] is True
    assert capability["implementationRevision"] == "test"
    assert capability["device"] == {
        "backend": "mps",
        "deviceName": "Apple MPS",
        "gpuIndex": None,
    }

    with pytest.raises(SSBProtocolError, match="local_mps"):
        service.reconstruct(_request(identity))
    assert calls == [(None, "local_mps")]


def test_local_mps_async_result_payload_and_capability_are_explicit(tmp_path):
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import BrowseService, create_app

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
            **_execution_evidence(request),
            "timings": {
                "firstReconstructSeconds": 0.1,
                "transferSeconds": None,
            },
        }

    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: pytest.fail("local MPS queried CUDA devices"),
        device_name=lambda _gpu: "Apple M5 Max; MLX test",
        runner=runner,
        source_inspector=_inspection,
        backend_kind="local_mps",
        implementation_revision="test-revision",
    )
    browse = BrowseService(tmp_path, initialize_cuda=False)
    client = TestClient(create_app(tmp_path, service=browse, ssb_service=service))
    capability_response = client.get("/api/browse/capabilities")
    capability = capability_response.json()

    assert capability_response.status_code == 200
    assert capability["backend"] == "mps"
    assert capability["device_error"] is None
    assert capability["features"] == {"ssb": service.capability()}
    assert capability["features"]["ssb"]["implementationRevision"] == "test-revision"
    assert capability["features"]["ssb"]["device"] == {
        "backend": "mps",
        "deviceName": "Apple M5 Max; MLX test",
        "gpuIndex": None,
    }

    request = _request(
        service.source_identity(str(master)), implementation_revision="test-revision"
    )
    request["backend"] = {"kind": "local_mps"}
    accepted = client.post("/api/ssb/jobs", json=request)
    completed = _wait_for_terminal(service, request["jobID"])
    payload = client.get(
        f"/api/ssb/jobs/{request['jobID']}/phase", params={"generation": 7}
    )

    assert accepted.status_code == 202
    assert completed["state"] == "completed"
    assert completed["result"]["requestedBackend"] == {"kind": "local_mps"}
    assert completed["result"]["executedDevice"]["backend"] == "mps"
    assert (
        completed["result"]["executedDevice"]["implementationRevision"]
        == "test-revision"
    )
    assert completed["result"]["timings"]["transferSeconds"] is None
    assert completed["result"]["executedBrightFieldCounts"] == {
        "logical": 4,
        "active": 3,
    }
    assert payload.status_code == 200
    assert calls == [(None, "local_mps")]


def test_unrecorded_or_unavailable_device_is_not_ready(tmp_path):
    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: (_ for _ in ()).throw(RuntimeError("device offline")),
        runner=lambda *_args: pytest.fail("unready service invoked runner"),
    )

    capability = service.capability()

    assert capability["ready"] is False
    assert capability["implementationRevision"] == "unrecorded"
    assert capability["unavailableReason"] == (
        "The exact quantem.gpu implementation revision is not recorded."
    )
    request = _request(
        service.source_identity(str(master)), implementation_revision="test-revision"
    )
    with pytest.raises(
        SSBProtocolError, match="implementation revision is not recorded"
    ):
        service.reconstruct(request)


def test_mps_session_closes_and_server_does_not_claim_loopback_transfer(
    tmp_path, monkeypatch
):
    import quantem.gpu as quantem_gpu

    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    closed = []
    opened = []

    class FakeSession:
        source_dtype = "uint8"
        source_load_seconds = 0.02

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(True)

        def reconstruct(self, _aberrations, *, compute_loss, force=False):
            assert compute_loss is True
            assert force is False
            return SimpleNamespace(
                phase=np.arange(4, dtype=np.float32).reshape(2, 2),
                loss=0.25,
            )

        def browser_state(self):
            return SimpleNamespace(
                num_bf=4,
                active_num_bf=3,
                angular_sampling_rad=(0.001090909, 0.001090909),
                bf_source_dtype=None,
                bf_source_max_value=None,
            )

    def fake_open(*_args, **kwargs):
        opened.append(kwargs)
        return FakeSession()

    monkeypatch.setattr(quantem_gpu, "SSB", SimpleNamespace(open=fake_open))
    monkeypatch.setattr(
        "quantem.gpu.remote.ssb_api.version",
        lambda package: "test-mlx" if package == "mlx" else pytest.fail(package),
    )
    service = SSBProtocolService(
        tmp_path,
        available_gpus=list,
        device_name=lambda _gpu: "Apple M5 Max",
        source_inspector=_inspection,
        backend_kind="local_mps",
        implementation_revision="test-revision",
    )
    identity = service.source_identity(str(master))
    prepared = service.prepare(
        _prepare_request(identity, backend={"kind": "local_mps"})
    )
    request = _request(identity, implementation_revision="test-revision")
    request["backend"] = {"kind": "local_mps"}
    request["preparedSelection"] = prepared

    result = service.reconstruct(request)

    assert closed == [True, True]
    assert opened[0]["dtype"] == "uint8"
    assert opened[0]["det_sampling"] == (1.090909, 1.090909)
    assert result["executedPrecision"] == request["precision"]
    assert result["timings"]["transferSeconds"] is None
    assert result["executedDevice"]["implementationRevision"] == "test-revision"

    class FailingSession(FakeSession):
        def reconstruct(self, _aberrations, *, compute_loss, force=False):
            raise RuntimeError("opaque MPS stage failed")

    monkeypatch.setattr(
        quantem_gpu,
        "SSB",
        SimpleNamespace(open=lambda *_args, **_kwargs: FailingSession()),
    )
    failed_request = _request(identity, implementation_revision="test-revision")
    failed_request["jobID"] = str(uuid4())
    failed_request["backend"] = {"kind": "local_mps"}

    with pytest.raises(RuntimeError, match="opaque MPS stage failed"):
        service.reconstruct(failed_request)
    assert closed == [True, True, True]


def test_source_hash_mismatch_is_rejected_before_runner(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)
    request["source"]["masterSHA256"] = "0" * 64

    with pytest.raises(SSBProtocolError, match="masterSHA256"):
        service.reconstruct(request)
    assert holder == {}


def test_detector_sampling_is_required_and_changes_request_identity(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)
    original_digest = service.request_sha256(request)
    values = request["calibration"]["resolution"]["calibration"]["calibration"]
    values.pop("detectorSamplingRowMilliradiansPerPixel")

    with pytest.raises(SSBProtocolError, match="detector sampling"):
        service.reconstruct(request)
    assert holder == {}

    changed = _request(identity)
    changed_values = changed["calibration"]["resolution"]["calibration"]["calibration"]
    changed_values["detectorSamplingRowMilliradiansPerPixel"]["value"] = 1.1
    assert service.request_sha256(changed) != original_digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda audit: audit.update(maximumCount=256), "not proven lossless"),
        (
            lambda audit: audit.update(countsAboveWorkingMaximum=1),
            "not proven lossless",
        ),
        (lambda audit: audit.update(auditedElementCount=15), "complete native"),
        (lambda audit: audit.update(sourceIdentitySHA256="0" * 64), "different source"),
        (lambda audit: audit.update(evidenceSHA256="0" * 64), "evidence digest"),
    ],
)
def test_invalid_or_incomplete_uint8_audit_is_rejected_before_runner(
    tmp_path, mutation, message
):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)
    mutation(request["precision"]["losslessWorkingDTypeAudit"])

    with pytest.raises(SSBProtocolError, match=message):
        service.reconstruct(request)
    assert holder == {}


def test_header_and_executed_precision_mismatches_are_rejected(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _request(identity)
    service._source_inspector = lambda *_args, **_kwargs: SimpleNamespace(
        ready=True,
        reason="",
        action="",
        dtype="uint32",
        detector_shape=(2, 2),
    )
    with pytest.raises(SSBProtocolError, match="native source dtype mismatch"):
        service.reconstruct(request)

    def wrong_runner(_source, _gpu, request):
        outcome = {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            **_execution_evidence(request),
            "timings": {"firstReconstructSeconds": 0.1},
        }
        outcome["precision"] = {**request["precision"], "workingSourceDType": "uint16"}
        return outcome

    service._source_inspector = _inspection
    service._runner = wrong_runner
    with pytest.raises(SSBProtocolError, match="executed precision"):
        service.reconstruct(request)


def test_logical_and_active_brightfield_counts_are_validated_separately(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))

    wrong_logical = _request(identity)
    wrong_logical["selection"]["logicalBrightFieldCount"] = 5
    _bind_prepared_selection(wrong_logical)
    with pytest.raises(SSBProtocolError, match="logical BF count changed"):
        service.reconstruct(wrong_logical)

    wrong_active = _request(identity)
    wrong_active["selection"]["activeBrightFieldCount"] = 2
    _bind_prepared_selection(wrong_active)
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


def test_async_request_is_validated_once_and_cannot_collide_with_sync(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def runner(_source, _gpu, request):
        entered.set()
        assert release.wait(timeout=2)
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            **_execution_evidence(request),
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
        source_inspector=_inspection,
        implementation_revision="test",
    )
    request = _request(service.source_identity(str(master)))
    original_validate = service._validate_request
    validations = 0

    def counted_validate(value):
        nonlocal validations
        validations += 1
        return original_validate(value)

    service._validate_request = counted_validate
    service.submit(request)
    assert entered.wait(timeout=2)
    with pytest.raises(SSBProtocolError, match="lifecycle endpoint"):
        service.reconstruct(request)
    release.set()
    assert _wait_for_terminal(service, request["jobID"])["state"] == "completed"
    assert validations == 1


def test_distinct_jobs_on_one_gpu_never_overlap(tmp_path):
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    counter_lock = threading.Lock()
    calls = 0
    active = 0
    maximum_active = 0

    def runner(_source, _gpu, request):
        nonlocal calls, active, maximum_active
        with counter_lock:
            calls += 1
            call = calls
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                second_entered.set()
            return {
                "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
                "logicalBrightFieldCount": 4,
                "activeBrightFieldCount": 3,
                **_execution_evidence(request),
                "implementationRevision": "test",
                "timings": {"firstReconstructSeconds": 0.1},
            }
        finally:
            with counter_lock:
                active -= 1

    master = tmp_path / "BTO_18_master.h5"
    master.write_bytes(b"master")
    (tmp_path / "BTO_18_data_000001.h5").write_bytes(b"shard")
    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: "Test CUDA",
        runner=runner,
        source_inspector=_inspection,
        implementation_revision="test",
    )
    identity = service.source_identity(str(master))
    first = _request(identity)
    second = _request(identity)
    second["jobID"] = str(uuid4())

    service.submit(first)
    assert first_entered.wait(timeout=2)
    service.submit(second)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    assert _wait_for_terminal(service, first["jobID"])["state"] == "completed"
    assert _wait_for_terminal(service, second["jobID"])["state"] == "completed"
    assert second_entered.is_set()
    assert maximum_active == 1


def test_cancel_during_runner_waits_for_stage_boundary_and_discards_payload(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def runner(_source, _gpu, request):
        entered.set()
        assert release.wait(timeout=2)
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            **_execution_evidence(request),
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
        source_inspector=_inspection,
        implementation_revision="test",
    )
    request = _request(service.source_identity(str(master)))

    service.submit(request)
    assert entered.wait(timeout=2)
    requested = service.cancel_job(request["jobID"], 7)
    assert requested["state"] == "cancel_requested"
    release.set()
    cancelled = _wait_for_terminal(service, request["jobID"])

    assert cancelled["state"] == "cancelled"
    with pytest.raises(SSBPayloadUnavailable):
        service.payload(request["jobID"], 7)


def test_cancel_acknowledged_before_stage_entry_never_starts_runner(tmp_path):
    stage_entry_pending = threading.Event()
    release_stage_entry = threading.Event()
    runner_entered = threading.Event()

    def runner(_source, _gpu, request):
        runner_entered.set()
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            **_execution_evidence(request),
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
        source_inspector=_inspection,
        implementation_revision="test",
    )
    request = _request(service.source_identity(str(master)))
    original_begin_stage = service._begin_stage

    def blocked_begin_stage(key, state):
        if state == "reconstructing_first":
            stage_entry_pending.set()
            assert release_stage_entry.wait(timeout=2)
        return original_begin_stage(key, state)

    service._begin_stage = blocked_begin_stage
    service.submit(request)
    assert stage_entry_pending.wait(timeout=2)

    acknowledged = service.cancel_job(request["jobID"], 7)
    assert acknowledged["state"] == "cancel_requested"
    release_stage_entry.set()
    terminal = _wait_for_terminal(service, request["jobID"])

    assert terminal["state"] == "cancelled"
    assert terminal["sequence"] > acknowledged["sequence"]
    assert runner_entered.is_set() is False


def test_async_http_status_cancel_and_payload_gate(tmp_path):
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import create_app

    entered = threading.Event()
    release = threading.Event()

    def runner(_source, _gpu, request):
        entered.set()
        assert release.wait(timeout=2)
        return {
            "phase": np.arange(4, dtype=np.float32).reshape(2, 2),
            "logicalBrightFieldCount": 4,
            "activeBrightFieldCount": 3,
            **_execution_evidence(request),
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
        source_inspector=_inspection,
        implementation_revision="test",
    )
    request = _request(service.source_identity(str(master)))
    client = TestClient(create_app(tmp_path, ssb_service=service))

    accepted = client.post("/api/ssb/jobs", json=request)
    assert accepted.status_code == 202
    assert entered.wait(timeout=2)
    snapshot = client.get(f"/api/ssb/jobs/{request['jobID']}", params={"generation": 7})
    assert snapshot.status_code == 200
    assert snapshot.json()["state"] == "reconstructing_first"
    assert (
        client.get(
            f"/api/ssb/jobs/{request['jobID']}/phase", params={"generation": 7}
        ).status_code
        == 425
    )

    cancelled = client.delete(
        f"/api/ssb/jobs/{request['jobID']}", params={"generation": 7}
    )
    assert cancelled.status_code == 202
    assert cancelled.json()["state"] == "cancel_requested"
    release.set()
    assert _wait_for_terminal(service, request["jobID"])["state"] == "cancelled"
    assert (
        client.get(
            f"/api/ssb/jobs/{request['jobID']}/phase", params={"generation": 7}
        ).status_code
        == 409
    )
