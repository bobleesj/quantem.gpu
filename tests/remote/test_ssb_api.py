from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.remote.ssb_api import SSBProtocolError, SSBProtocolService


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
