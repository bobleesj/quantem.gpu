from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from quantem.gpu.remote.ssb_api import (
    ALGORITHM_VERSION,
    INTERACTIVE_CONTRACT_VERSION,
    PREPARE_CONTRACT_V0_1,
    PREPARE_CONTRACT_VERSION,
    SSBPayloadNotReady,
    SSBPayloadUnavailable,
    SSBProtocolError,
    SSBProtocolService,
    _fit_history_counts,
    _source_identity_sha256,
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
    selected_backend = backend or {
        "kind": "remote_cuda",
        "profile_id": "mjgoat",
        "gpu_index": 0,
    }
    expected_identity = {
        key: source[key]
        for key in (
            "datasetSchema",
            "masterSHA256",
            "orderedMemberSHA256",
            "sourceIdentitySHA256",
        )
    }
    source_locator = {
        "kind": "configured_root_identity",
        "expectedIdentity": expected_identity,
    }
    if selected_backend["kind"] == "local_mps":
        source_locator = {
            "kind": "local_path",
            "masterPath": source["masterPath"],
            "expectedIdentity": expected_identity,
        }
    return {
        "contractVersion": PREPARE_CONTRACT_VERSION,
        "sourceLocator": source_locator,
        "calibration": reconstruction["calibration"],
        "selection": {
            "policy": "full_active",
            "detectorBin": 1,
            "scanCrop": None,
        },
        "precision": reconstruction["precision"],
        "backend": selected_backend,
    }


def _service(
    tmp_path: Path,
    *,
    prepared_sampling: tuple[float, float] = (1.090909, 1.090909),
    clock=lambda: time.time(),
    session_lease_seconds: float = 300.0,
    backend_kind: str = "remote_cuda",
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

    class FakeSession:
        source_dtype = "uint8"

        def fit(
            self,
            *,
            trials,
            refinement,
            search_ranges,
            seed,
            force,
            verbose,
        ):
            holder.setdefault("sessionFits", []).append(
                {
                    "trials": trials,
                    "refinement": refinement,
                    "searchRanges": search_ranges,
                    "seed": seed,
                    "force": force,
                    "verbose": verbose,
                }
            )
            gate = holder.get("fitGate")
            if gate is not None:
                holder["fitStarted"].set()
                gate.wait(timeout=5)
            fitted = {"C10": 73.0, "C12": 14.0, "phi12": 0.47}
            history = [{"params": fitted, "loss": 0.04}] * trials
            if backend_kind == "local_mps":
                history.insert(0, {"params": fitted, "loss": 0.05})
            return SimpleNamespace(
                phase=np.full((2, 2), 73.0, dtype=np.float32),
                aberrations=fitted,
                loss=0.04,
                timings={"optuna_seconds": 0.2},
                n_trials=trials,
                optuna_trials=history,
            )

        def set_rotation(self, value):
            holder.setdefault("rotations", []).append(float(value))

        def reconstruct(self, aberrations, *, compute_loss, force=False):
            holder.setdefault("sessionReconstructs", []).append(
                {
                    "aberrations": dict(aberrations),
                    "computeLoss": compute_loss,
                    "force": force,
                }
            )
            gate = holder.get("warmGate")
            if gate is not None and len(holder["sessionReconstructs"]) > 1:
                holder["warmStarted"].set()
                gate.wait(timeout=5)
            if (
                holder.get("warmBaseException")
                and len(holder["sessionReconstructs"]) > 1
            ):
                raise SystemExit("simulated worker termination")
            value = float(aberrations["C10"])
            return SimpleNamespace(
                phase=np.full((2, 2), value, dtype=np.float32),
                loss=value if compute_loss else None,
            )

        def browser_state(self):
            return SimpleNamespace(
                num_bf=4,
                active_num_bf=3,
                angular_sampling_rad=(0.001090909, 0.001090909),
            )

    class FakeContext:
        def __init__(self):
            self.session = FakeSession()

        def __enter__(self):
            return self.session

        def __exit__(self, *_args):
            holder["sessionCloses"] = holder.get("sessionCloses", 0) + 1

    def session_opener(_source, gpu, request):
        holder.setdefault("sessionOpens", []).append(
            {"gpu": gpu, "backend": request["backend"]["kind"]}
        )
        context = FakeContext()
        return context, context.__enter__()

    service = SSBProtocolService(
        tmp_path,
        available_gpus=lambda: [0],
        device_name=lambda _gpu: "Test CUDA",
        runner=runner,
        preparer=preparer,
        session_opener=session_opener,
        session_device_context=lambda _gpu: nullcontext(),
        runtime_diagnostics=lambda: {
            "processID": 123,
            "processPeakRSSBytes": 456,
            "mlxActiveBytes": None,
            "mlxPeakBytes": None,
            "mlxCacheBytes": None,
        },
        source_inspector=_inspection,
        clock=clock,
        session_lease_seconds=session_lease_seconds,
        implementation_revision="test",
        backend_kind=backend_kind,
    )
    return service, holder


def _open_interactive(
    service, identity, *, backend=None, contract_version=INTERACTIVE_CONTRACT_VERSION
):
    backend = backend or {"kind": "remote_cuda", "profile_id": "mjgoat", "gpu_index": 0}
    initial = _request(identity)
    initial["jobID"] = str(uuid4())
    initial["backend"] = backend
    initial["preparedSelection"] = service.prepare(
        _prepare_request(identity, backend=backend)
    )
    return service.open_interactive_session(
        {
            "contractVersion": contract_version,
            "sessionID": str(uuid4()),
            "initialRequest": initial,
        }
    )


def _interactive_request(opened, *, control_generation=1, job_id=None):
    session = opened["session"]
    binding = session["binding"]
    return {
        "contractVersion": INTERACTIVE_CONTRACT_VERSION,
        "sessionID": session["sessionID"],
        "jobID": job_id or str(uuid4()),
        "datasetGeneration": opened["initialResult"]["datasetGeneration"],
        "controlGeneration": control_generation,
        "sessionBindingSHA256": binding["sessionBindingSHA256"],
        "sourceIdentitySHA256": binding["sourceIdentitySHA256"],
        "selectionSHA256": binding["selectionSHA256"],
        "backend": binding["backend"],
        "controls": {
            "C10": 74.0 + control_generation,
            "C12": 14.0,
            "phi12": 0.47,
            "scanRotation": 158.9,
        },
        "evaluation": "exact_full_bf_object_phase",
        "settlesControlGeneration": None,
    }


def _fit_request(opened, *, control_generation=1, job_id=None):
    session = opened["session"]
    binding = session["binding"]
    return {
        "contractVersion": "live4dstem.ssb.fit/v0.1",
        "sessionID": session["sessionID"],
        "jobID": job_id or str(uuid4()),
        "datasetGeneration": opened["initialResult"]["datasetGeneration"],
        "controlGeneration": control_generation,
        "sessionBindingSHA256": binding["sessionBindingSHA256"],
        "sourceIdentitySHA256": binding["sourceIdentitySHA256"],
        "selectionSHA256": binding["selectionSHA256"],
        "backend": binding["backend"],
        "optimizerTrials": 200,
        "seed": 42,
        "objective": "exact_full_active_bf_phase_variance_float32",
        "searchRanges": {
            "C10Nanometers": {"minimum": -400.0, "maximum": 400.0},
            "C12Nanometers": {"minimum": 0.0, "maximum": 100.0},
            "phi12Radians": {
                "minimum": -math.pi / 2.0,
                "maximum": math.pi / 2.0,
            },
        },
        "fixedScanRotationDegrees": 158.8827,
        "refinement": None,
    }


def _wait_interactive(service, request):
    for _ in range(200):
        snapshot = service.interactive_job_snapshot(
            request["jobID"], request["datasetGeneration"]
        )
        if snapshot["state"] in {
            "completed",
            "cancelled",
            "failed",
            "expired",
            "superseded",
        }:
            return snapshot
        time.sleep(0.005)
    raise AssertionError("interactive SSB job did not finish")


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


def test_remote_prepare_resolves_one_complete_identity_without_client_path(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _prepare_request(identity)

    assert "masterPath" not in request
    assert "masterPath" not in request["sourceLocator"]
    descriptor = service.prepare(request)

    assert descriptor["source"] == identity
    assert descriptor["contractVersion"] == PREPARE_CONTRACT_VERSION


def test_remote_prepare_rejects_missing_and_ambiguous_complete_identity(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    missing = json.loads(json.dumps(identity))
    missing["masterSHA256"] = "0" * 64
    missing["sourceIdentitySHA256"] = _source_identity_sha256(
        missing["masterSHA256"], missing["orderedMemberSHA256"]
    )
    request = _prepare_request(missing)
    with pytest.raises(SSBProtocolError, match="No configured-root"):
        service.prepare(request)

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "BTO_18_master.h5").write_bytes(b"master")
    (duplicate / "BTO_18_data_000001.h5").write_bytes(b"shard")
    with pytest.raises(SSBProtocolError, match="ambiguous"):
        service.prepare(_prepare_request(identity))


def test_remote_prepare_rejects_client_path_and_inconsistent_identity(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _prepare_request(identity)
    request["sourceLocator"]["masterPath"] = "/Users/phil/BTO_18_master.h5"
    with pytest.raises(SSBProtocolError, match="rejects client paths"):
        service.prepare(request)

    request = _prepare_request(identity)
    request["sourceLocator"]["expectedIdentity"]["orderedMemberSHA256"] = ["f" * 64]
    with pytest.raises(SSBProtocolError, match="digest is inconsistent"):
        service.prepare(request)


def test_remote_prepare_never_resolves_a_symlink_outside_configured_root(tmp_path):
    root = tmp_path / "served"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    outside_master = outside / "BTO_18_master.h5"
    outside_master.write_bytes(b"master")
    (outside / "BTO_18_data_000001.h5").write_bytes(b"shard")
    service, _ = _service(root)
    identity = service.source_identity(str(root / "BTO_18_master.h5"))
    (root / "linked_master.h5").symlink_to(outside_master)
    (root / "BTO_18_master.h5").unlink()
    (root / "BTO_18_data_000001.h5").unlink()

    with pytest.raises(SSBProtocolError, match="No configured-root"):
        service.prepare(_prepare_request(identity))


def test_prepare_v0_1_remains_explicitly_compatible(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    request = _prepare_request(identity)
    request.update(
        contractVersion=PREPARE_CONTRACT_V0_1,
        masterPath=identity["masterPath"],
        expectedSourceIdentitySHA256=identity["sourceIdentitySHA256"],
    )
    request.pop("sourceLocator")

    descriptor = service.prepare(request)

    assert descriptor["contractVersion"] == PREPARE_CONTRACT_V0_1
    assert "datasetSchema" not in descriptor["source"]
    reconstruction = _request(identity)
    reconstruction["preparedSelection"] = descriptor
    service.reconstruct(reconstruction)


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
        "compatibleContractVersions": [PREPARE_CONTRACT_V0_1],
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
    assert capability["sourceResolution"] == {
        "contractVersion": "live4dstem.ssb.source-resolution/v0.1",
        "mode": "configured_root_exact_identity",
        "remoteClientPathAccepted": False,
        "missingIsError": True,
        "ambiguousIsError": True,
    }
    assert capability["fixedReconstructionControls"] == {
        "aberrations": [
            {"name": "C10", "unit": "nm"},
            {"name": "C12", "unit": "nm"},
            {"name": "phi12", "unit": "radian"},
        ],
        "scanRotation": {"name": "scanRotation", "unit": "degree"},
        "higherOrderAberrations": False,
        "preparedSessionWarmReconstruct": True,
        "contractVersion": INTERACTIVE_CONTRACT_VERSION,
        "compatibleRequestVersions": ["live4dstem.ssb.interactive/v0.1"],
        "sessionEndpoint": "/api/ssb/interactive/sessions",
        "jobEndpoint": "/api/ssb/interactive/jobs",
        "evaluations": [
            "exact_full_bf_object_phase",
            "exact_full_bf_object_phase_and_loss",
        ],
        "settleBinding": "exact_prior_preview_control_generation",
        "saveEvidenceRequiresLossState": "settled",
        "sessionLeaseSeconds": 300,
        "cancellationMode": "stage_boundary",
        "reconnectScope": "same_server_process",
        "serverRestartResume": False,
    }


def test_interactive_session_reuses_one_open_and_publishes_bound_result(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))

    opened = _open_interactive(service, identity)
    request = _interactive_request(opened)
    service.submit_interactive(request)
    completed = _wait_interactive(service, request)
    payload, descriptor = service.interactive_payload(
        request["jobID"], request["datasetGeneration"]
    )

    assert holder["sessionOpens"] == [{"gpu": 0, "backend": "remote_cuda"}]
    assert len(holder["sessionReconstructs"]) == 2
    assert all(call["force"] for call in holder["sessionReconstructs"])
    assert completed["state"] == "completed"
    result = completed["result"]
    assert result["controlGeneration"] == 1
    assert result["interactiveControls"] == request["controls"]
    assert result["source"] == opened["initialResult"]["source"]
    assert result["selection"] == opened["initialResult"]["selection"]
    assert result["executedBrightFieldCounts"] == {"logical": 4, "active": 3}
    assert result["executedPrecision"] == opened["initialResult"]["executedPrecision"]
    assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["phase"]["sha256"] == descriptor["sha256"]
    assert result["runtimeMemory"]["processID"] > 0
    assert result["runtimeMemory"]["processPeakRSSBytes"] > 0
    assert opened["session"]["binding"]["device"] == {
        "backend": "cuda",
        "deviceName": "Test CUDA",
        "gpuIndex": 0,
    }
    canonical_result = json.loads(json.dumps(result))
    provenance = canonical_result.pop("provenanceSHA256")
    assert (
        provenance
        == hashlib.sha256(
            json.dumps(canonical_result, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )


def test_interactive_worker_logs_base_exception_and_publishes_failure(tmp_path, caplog):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _interactive_request(opened)
    holder["warmBaseException"] = True

    with caplog.at_level("ERROR", logger="quantem.gpu.remote.ssb"):
        service.submit_interactive(request)
        completed = _wait_interactive(service, request)

    assert completed["state"] == "failed"
    assert completed["error"] == {"message": "simulated worker termination"}
    assert "Retained SSB reconstruction failed" in caplog.text


def test_interactive_http_roundtrip_and_shutdown_close(tmp_path):
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import BrowseService, create_app

    service, holder = _service(tmp_path)
    browse = BrowseService(tmp_path, initialize_cuda=False)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    initial = _request(identity)
    initial["jobID"] = str(uuid4())
    initial["preparedSelection"] = service.prepare(_prepare_request(identity))
    app = create_app(tmp_path, service=browse, ssb_service=service)

    with TestClient(app) as client:

        def assert_phase_payload(result):
            descriptor = result["phase"]
            payload = client.get(result["phasePayload"]["path"])
            assert payload.status_code == 200
            assert payload.headers["X-Width"] == str(descriptor["shape"]["columns"])
            assert payload.headers["X-Height"] == str(descriptor["shape"]["rows"])
            assert payload.headers["X-Dtype"] == descriptor["dtype"]
            assert payload.headers["X-Byte-Count"] == str(descriptor["byteCount"])
            assert payload.headers["X-SHA256"] == descriptor["sha256"]
            assert len(payload.content) == descriptor["byteCount"]
            assert hashlib.sha256(payload.content).hexdigest() == descriptor["sha256"]

        open_request = {
            "contractVersion": INTERACTIVE_CONTRACT_VERSION,
            "sessionID": str(uuid4()),
            "initialRequest": initial,
        }
        opened_response = client.post(
            "/api/ssb/interactive/sessions", json=open_request
        )
        assert opened_response.status_code == 201
        opened = opened_response.json()
        assert_phase_payload(opened["initialResult"])
        repeated = client.post("/api/ssb/interactive/sessions", json=open_request)
        assert repeated.status_code == 201
        assert repeated.json() == opened
        assert len(holder["sessionOpens"]) == 1
        reconnected = client.get(
            f"/api/ssb/interactive/sessions/{opened['session']['sessionID']}"
        )
        assert reconnected.status_code == 200
        assert reconnected.json() == opened["session"]
        request = _interactive_request(opened)
        submitted = client.post("/api/ssb/interactive/jobs", json=request)
        assert submitted.status_code == 202
        for _ in range(200):
            snapshot = client.get(
                f"/api/ssb/interactive/jobs/{request['jobID']}",
                params={"generation": request["datasetGeneration"]},
            ).json()
            if snapshot["state"] == "completed":
                break
            time.sleep(0.005)
        assert snapshot["state"] == "completed"
        assert_phase_payload(snapshot["result"])
        fit_request = _fit_request(opened, control_generation=2)
        fit_accepted = client.post("/api/ssb/interactive/fits", json=fit_request)
        assert fit_accepted.status_code == 202
        for _ in range(200):
            fit_snapshot = client.get(
                f"/api/ssb/interactive/jobs/{fit_request['jobID']}",
                params={"generation": fit_request["datasetGeneration"]},
            ).json()
            if fit_snapshot["state"] == "completed":
                break
            time.sleep(0.005)
        assert fit_snapshot["state"] == "completed"
        assert_phase_payload(fit_snapshot["result"])
        assert (
            fit_snapshot["result"]["initialAberrationFit"][
                "persistedAsDatasetCalibration"
            ]
            is False
        )

    assert holder["sessionCloses"] == 1


def test_interactive_warm_same_controls_has_exact_phase_parity_without_loss(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _interactive_request(opened)
    request["controls"] = opened["initialResult"]["interactiveControls"]

    service.submit_interactive(request)
    completed = _wait_interactive(service, request)

    assert completed["state"] == "completed"
    assert (
        completed["result"]["phase"]["sha256"]
        == opened["initialResult"]["phase"]["sha256"]
    )
    assert completed["result"]["loss"] is None
    assert completed["result"]["evaluation"] == "exact_full_bf_object_phase"
    assert completed["result"]["lossState"] == "pending_exact_phase_variance"
    assert completed["result"]["saveEvidenceEligible"] is False
    assert completed["result"]["timings"]["sessionOpenSeconds"] is None
    assert completed["result"]["timings"]["sourceReadSeconds"] is None
    assert completed["result"]["timings"]["warmReconstructSeconds"] is not None


def test_interactive_v02_preview_then_exact_settle_uses_same_controls(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    preview = _interactive_request(opened, control_generation=1)

    accepted = service.submit_interactive(preview)
    assert accepted["evaluation"] == "exact_full_bf_object_phase"
    assert accepted["settlesControlGeneration"] is None
    assert accepted["lossState"] == "pending_exact_phase_variance"
    preview_done = _wait_interactive(service, preview)

    settle = _interactive_request(opened, control_generation=2)
    settle["controls"] = preview["controls"]
    settle["evaluation"] = "exact_full_bf_object_phase_and_loss"
    settle["settlesControlGeneration"] = 1
    service.submit_interactive(settle)
    settled = _wait_interactive(service, settle)

    assert [call["computeLoss"] for call in holder["sessionReconstructs"]] == [
        True,
        False,
        True,
    ]
    assert preview_done["result"]["loss"] is None
    assert settled["result"]["loss"] == preview["controls"]["C10"]
    assert (
        settled["result"]["phase"]["sha256"]
        == preview_done["result"]["phase"]["sha256"]
    )
    assert settled["result"]["evaluation"] == "exact_full_bf_object_phase_and_loss"
    assert settled["result"]["settlesControlGeneration"] == 1
    assert settled["result"]["lossState"] == "settled"
    assert settled["result"]["saveEvidenceEligible"] is True


def test_interactive_v02_rejects_stale_or_changed_preview_settle(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    first = _interactive_request(opened, control_generation=1)
    service.submit_interactive(first)
    _wait_interactive(service, first)

    changed = _interactive_request(opened, control_generation=2)
    changed["evaluation"] = "exact_full_bf_object_phase_and_loss"
    changed["settlesControlGeneration"] = 1
    with pytest.raises(SSBProtocolError, match="exact completed preview controls"):
        service.submit_interactive(changed)

    second = _interactive_request(opened, control_generation=2)
    service.submit_interactive(second)
    _wait_interactive(service, second)
    stale = _interactive_request(opened, control_generation=3)
    stale["controls"] = first["controls"]
    stale["evaluation"] = "exact_full_bf_object_phase_and_loss"
    stale["settlesControlGeneration"] = 1
    with pytest.raises(SSBProtocolError, match="stale preview"):
        service.submit_interactive(stale)
    assert len(holder["sessionReconstructs"]) == 3


def test_interactive_v01_compute_loss_compatibility_is_explicit(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(
        service,
        identity,
        contract_version="live4dstem.ssb.interactive/v0.1",
    )
    assert opened["contractVersion"] == INTERACTIVE_CONTRACT_VERSION
    request = _interactive_request(opened)
    request["contractVersion"] = "live4dstem.ssb.interactive/v0.1"
    request.pop("evaluation")
    request.pop("settlesControlGeneration")
    request["computeLoss"] = False

    accepted = service.submit_interactive(request)
    completed = _wait_interactive(service, request)

    assert accepted["contractVersion"] == "live4dstem.ssb.interactive/v0.1"
    assert accepted["evaluation"] == "exact_full_bf_object_phase"
    assert holder["sessionReconstructs"][-1]["computeLoss"] is False
    assert completed["result"]["loss"] is None
    assert completed["result"]["phasePayload"]["path"].endswith(
        f"/phase?generation={request['datasetGeneration']}"
    )

    legacy_loss = _interactive_request(opened, control_generation=2)
    legacy_loss["contractVersion"] = "live4dstem.ssb.interactive/v0.1"
    legacy_loss.pop("evaluation")
    legacy_loss.pop("settlesControlGeneration")
    legacy_loss["computeLoss"] = True
    service.submit_interactive(legacy_loss)
    loss_completed = _wait_interactive(service, legacy_loss)
    assert holder["sessionReconstructs"][-1]["computeLoss"] is True
    assert loss_completed["result"]["lossState"] == "settled"
    assert loss_completed["result"]["saveEvidenceEligible"] is True


def test_interactive_v02_payload_paths_are_canonical_path_only(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    assert "?" not in opened["initialResult"]["phasePayload"]["path"]

    request = _interactive_request(opened)
    service.submit_interactive(request)
    completed = _wait_interactive(service, request)

    assert "?" not in completed["result"]["phasePayload"]["path"]


def test_initial_fit_uses_exact_common_200_trial_contract_and_slider_seed(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _fit_request(opened)

    accepted = service.submit_interactive_fit(request)
    completed = _wait_interactive(service, request)
    payload, descriptor = service.interactive_payload(
        request["jobID"], request["datasetGeneration"]
    )

    assert accepted["state"] == "accepted"
    assert holder["sessionOpens"] == [{"gpu": 0, "backend": "remote_cuda"}]
    assert holder["sessionFits"] == [
        {
            "trials": 200,
            "refinement": None,
            "searchRanges": {
                "C10_nm": (-400.0, 400.0),
                "C12_nm": (0.0, 100.0),
                "phi12_deg": (-90.0, 90.0),
            },
            "seed": 42,
            "force": True,
            "verbose": False,
        }
    ]
    assert completed["state"] == "completed"
    evidence = completed["result"]["initialAberrationFit"]
    assert evidence["evidenceVersion"] == "live4dstem.ssb.fit.evidence/v0.2"
    assert evidence["specification"] == accepted["fitSpecification"]
    assert evidence["optimizerTrialsCompleted"] == 200
    assert evidence["optimizerTrialHistoryCount"] == 200
    assert evidence["baselineHistoryCount"] == 0
    assert evidence["recordedHistoryCount"] == 200
    assert evidence["totalObjectiveEvaluationCount"] is None
    assert evidence["fittedControls"] == {
        "C10": 73.0,
        "C12": 14.0,
        "phi12": 0.47,
        "scanRotation": 158.8827,
    }
    assert evidence["sliderSeed"] == evidence["fittedControls"]
    assert evidence["operatorAcceptanceRequired"] is True
    assert evidence["persistedAsDatasetCalibration"] is False
    assert completed["result"]["requestedBackend"] == request["backend"]
    assert completed["result"]["executedDevice"]["backend"] == "cuda"
    assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()


def test_initial_fit_history_distinguishes_mps_baseline_from_optimizer_trials():
    assert _fit_history_counts(
        backend_kind="remote_cuda", optimizer_trials=200, recorded_history_count=200
    ) == (200, 0, 200)
    assert _fit_history_counts(
        backend_kind="local_mps", optimizer_trials=200, recorded_history_count=201
    ) == (200, 1, 201)
    with pytest.raises(SSBProtocolError, match="baseline=1, recorded=200"):
        _fit_history_counts(
            backend_kind="local_mps", optimizer_trials=200, recorded_history_count=200
        )


def test_local_mps_fit_result_labels_one_recorded_baseline_and_200_optimizer_trials(
    tmp_path,
):
    service, holder = _service(tmp_path, backend_kind="local_mps")
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity, backend={"kind": "local_mps"})
    request = _fit_request(opened)

    service.submit_interactive_fit(request)
    completed = _wait_interactive(service, request)

    assert completed["state"] == "completed"
    evidence = completed["result"]["initialAberrationFit"]
    assert holder["sessionFits"][0]["trials"] == 200
    assert evidence["optimizerTrialsCompleted"] == 200
    assert evidence["optimizerTrialHistoryCount"] == 200
    assert evidence["baselineHistoryCount"] == 1
    assert evidence["recordedHistoryCount"] == 201
    assert evidence["totalObjectiveEvaluationCount"] is None
    assert completed["result"]["executedDevice"]["backend"] == "mps"


def test_local_mps_fit_http_reports_split_history_and_payload_headers(tmp_path):
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import BrowseService, create_app

    service, _ = _service(tmp_path, backend_kind="local_mps")
    browse = BrowseService(tmp_path, initialize_cuda=False)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    backend = {"kind": "local_mps"}
    initial = _request(identity)
    initial["jobID"] = str(uuid4())
    initial["backend"] = backend
    initial["preparedSelection"] = service.prepare(
        _prepare_request(identity, backend=backend)
    )
    app = create_app(tmp_path, service=browse, ssb_service=service)

    with TestClient(app) as client:
        opened_response = client.post(
            "/api/ssb/interactive/sessions",
            json={
                "contractVersion": INTERACTIVE_CONTRACT_VERSION,
                "sessionID": str(uuid4()),
                "initialRequest": initial,
            },
        )
        assert opened_response.status_code == 201
        opened = opened_response.json()
        request = _fit_request(opened)
        accepted = client.post("/api/ssb/interactive/fits", json=request)
        assert accepted.status_code == 202
        for _ in range(200):
            snapshot = client.get(
                f"/api/ssb/interactive/jobs/{request['jobID']}",
                params={"generation": request["datasetGeneration"]},
            ).json()
            if snapshot["state"] == "completed":
                break
            time.sleep(0.005)
        assert snapshot["state"] == "completed"
        evidence = snapshot["result"]["initialAberrationFit"]
        assert evidence["optimizerTrialHistoryCount"] == 200
        assert evidence["baselineHistoryCount"] == 1
        assert evidence["recordedHistoryCount"] == 201
        payload = client.get(snapshot["result"]["phasePayload"]["path"])
        assert payload.status_code == 200
        assert payload.headers["X-Dtype"] == "float32"
        assert int(payload.headers["X-Byte-Count"]) == len(payload.content)
        assert (
            payload.headers["X-SHA256"] == hashlib.sha256(payload.content).hexdigest()
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda request: request.update(optimizerTrials=199), "exactly 200"),
        (
            lambda request: request["searchRanges"].update(
                C21Nanometers={"minimum": -1.0, "maximum": 1.0}
            ),
            "higher-order",
        ),
        (
            lambda request: request.update(fixedScanRotationDegrees=159.0),
            "stay fixed",
        ),
        (
            lambda request: request.update(backend={"kind": "local_mps"}),
            "backend or device",
        ),
    ],
)
def test_initial_fit_rejects_budget_higher_order_rotation_and_fallback(
    tmp_path, mutation, message
):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _fit_request(opened)
    mutation(request)

    with pytest.raises(SSBProtocolError, match=message):
        service.submit_interactive_fit(request)
    assert holder.get("sessionFits") is None


def test_initial_fit_latest_wins_and_stage_boundary_cancel(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    holder["fitGate"] = threading.Event()
    holder["fitStarted"] = threading.Event()
    first = _fit_request(opened, control_generation=1)
    second = _fit_request(opened, control_generation=2)

    service.submit_interactive_fit(first)
    assert holder["fitStarted"].wait(timeout=2)
    service.submit_interactive_fit(second)
    service.cancel_interactive_job(second["jobID"], second["datasetGeneration"])
    holder["fitGate"].set()

    first_done = _wait_interactive(service, first)
    second_done = _wait_interactive(service, second)
    assert first_done["state"] == "superseded"
    assert second_done["state"] == "cancelled"
    assert len(holder["sessionFits"]) == 1
    with pytest.raises(SSBPayloadUnavailable):
        service.interactive_payload(first["jobID"], first["datasetGeneration"])
    with pytest.raises(SSBPayloadUnavailable):
        service.interactive_payload(second["jobID"], second["datasetGeneration"])


def test_initial_fit_rejects_stale_dataset_generation(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _fit_request(opened)
    request["datasetGeneration"] += 1

    with pytest.raises(SSBProtocolError, match="dataset generation"):
        service.submit_interactive_fit(request)
    assert holder.get("sessionFits") is None


def test_initial_fit_capability_is_backend_honest() -> None:
    cuda = SSBProtocolService.advertised_capability(
        backend_kind="remote_cuda",
        implementation_revision="revision",
        device_name="CUDA Device",
        gpu_index=0,
    )["initialAberrationFit"]
    mps = SSBProtocolService.advertised_capability(
        backend_kind="local_mps",
        implementation_revision="revision",
        device_name="Apple GPU",
    )["initialAberrationFit"]

    assert cuda["optimizerTrials"] == mps["optimizerTrials"] == 200
    assert cuda["objective"] == mps["objective"]
    assert cuda["searchRanges"] == mps["searchRanges"]
    assert cuda["scanRotation"] == mps["scanRotation"] == "fixed_to_retained_session"
    assert cuda["candidateBatchSize"] == 4
    assert mps["candidateBatchSize"] == 2
    assert (
        cuda["evidenceVersion"]
        == mps["evidenceVersion"]
        == ("live4dstem.ssb.fit.evidence/v0.2")
    )
    assert cuda["optimizerTrialHistoryCount"] == 200
    assert mps["optimizerTrialHistoryCount"] == 200
    assert cuda["baselineHistoryCount"] == 0
    assert mps["baselineHistoryCount"] == 1
    assert cuda["recordedHistoryCount"] == 200
    assert mps["recordedHistoryCount"] == 201
    assert cuda["totalObjectiveEvaluations"] == "not_exposed_by_public_backend"
    assert mps["totalObjectiveEvaluations"] == "not_exposed_by_public_backend"
    assert cuda["retainedSessionBehavior"] == "reuses_prepared_accelerator"
    assert mps["retainedSessionBehavior"] == "reprepares_from_retained_source_object"
    assert cuda["sourceReopen"] is mps["sourceReopen"] is False
    assert cuda["implicitFallback"] is mps["implicitFallback"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sessionBindingSHA256", "0" * 64, "session binding"),
        ("sourceIdentitySHA256", "0" * 64, "source or selection"),
        ("selectionSHA256", "0" * 64, "source or selection"),
        ("backend", {"kind": "local_mps"}, "backend or device"),
    ],
)
def test_interactive_session_rejects_tampered_binding(tmp_path, field, value, message):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _interactive_request(opened)
    request[field] = value

    with pytest.raises(SSBProtocolError, match=message):
        service.submit_interactive(request)


def test_interactive_session_rejects_higher_orders_and_stale_generation(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _interactive_request(opened)
    request["controls"]["C21"] = 1.0
    with pytest.raises(SSBProtocolError, match="exactly C10"):
        service.submit_interactive(request)

    request = _interactive_request(opened)
    request["datasetGeneration"] += 1
    with pytest.raises(SSBProtocolError, match="dataset generation"):
        service.submit_interactive(request)

    accepted = _interactive_request(opened, control_generation=1)
    service.submit_interactive(accepted)
    _wait_interactive(service, accepted)
    repeated = _interactive_request(opened, control_generation=1)
    with pytest.raises(SSBProtocolError, match="increase monotonically"):
        service.submit_interactive(repeated)


def test_interactive_session_is_not_resumable_after_service_restart(tmp_path):
    service, _ = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    request = _interactive_request(opened)
    restarted, _ = _service(tmp_path)

    with pytest.raises(SSBProtocolError, match="missing or expired"):
        restarted.submit_interactive(request)


def test_interactive_session_expires_closes_and_releases_device(tmp_path):
    now = [100.0]
    service, holder = _service(
        tmp_path, clock=lambda: now[0], session_lease_seconds=5.0
    )
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    with pytest.raises(SSBProtocolError, match="already has"):
        _open_interactive(service, identity)

    now[0] = 106.0
    reopened = _open_interactive(service, identity)

    assert holder["sessionCloses"] == 1
    assert reopened["session"]["sessionID"] != opened["session"]["sessionID"]
    closed = service.close_interactive_session(reopened["session"]["sessionID"])
    assert closed["state"] == "closed"
    assert holder["sessionCloses"] == 2


def test_interactive_latest_wins_serializes_and_discards_stale_publish(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    holder["warmGate"] = threading.Event()
    holder["warmStarted"] = threading.Event()
    first = _interactive_request(opened, control_generation=1)
    service.submit_interactive(first)
    assert holder["warmStarted"].wait(timeout=2)
    second = _interactive_request(opened, control_generation=2)
    service.submit_interactive(second)
    time.sleep(0.02)
    assert len(holder["sessionReconstructs"]) == 2
    holder["warmGate"].set()

    first_done = _wait_interactive(service, first)
    second_done = _wait_interactive(service, second)

    assert first_done["state"] == "superseded"
    assert second_done["state"] == "completed"
    with pytest.raises(SSBPayloadUnavailable, match="superseded"):
        service.interactive_payload(first["jobID"], first["datasetGeneration"])


def test_interactive_cancel_waits_for_opaque_stage_and_never_publishes(tmp_path):
    service, holder = _service(tmp_path)
    identity = service.source_identity(str(tmp_path / "BTO_18_master.h5"))
    opened = _open_interactive(service, identity)
    holder["warmGate"] = threading.Event()
    holder["warmStarted"] = threading.Event()
    request = _interactive_request(opened)
    service.submit_interactive(request)
    assert holder["warmStarted"].wait(timeout=2)

    cancelled = service.cancel_interactive_job(
        request["jobID"], request["datasetGeneration"]
    )
    assert cancelled["state"] == "cancel_requested"
    with pytest.raises(SSBPayloadNotReady):
        service.interactive_payload(request["jobID"], request["datasetGeneration"])
    holder["warmGate"].set()
    done = _wait_interactive(service, request)

    assert done["state"] == "cancelled"
    with pytest.raises(SSBPayloadUnavailable, match="cancelled"):
        service.interactive_payload(request["jobID"], request["datasetGeneration"])


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
