from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from uuid import uuid4

import numpy as np
import pytest

from quantem.gpu.remote.maped_api import (
    ALGORITHM_VERSION,
    CACHE_VERSION,
    CONTRACT_VERSION,
    MAPEDProtocolError,
    MAPEDProtocolService,
)


_TILTS = [
    (-17.0, 0.0),
    (-8.5, -14.72),
    (-8.5, 14.72),
    (0.0, 0.0),
    (17.0, 0.0),
    (8.5, -14.72),
    (8.5, 14.72),
]


def _parameters() -> dict[str, object]:
    return {
        "batchSize": 16,
        "detectorBin": 1,
        "scanBin": 1,
        "originSigmaPixels": 1,
        "diffractionEdgeBlendPixels": 2,
        "realSpaceIterations": 20,
        "hanningFilter": True,
        "realSpacePaddingPixels": 2,
        "realSpaceEdgeBlendPixels": 5,
        "realSpacePadValue": "median",
        "shiftMethod": "bilinear",
        "accumulatorPolicy": "float32_accumulator",
        "outputDtype": "float32",
    }


def _fixture_folder(root: Path, count: int = 2) -> Path:
    folder = root / "samsung" / "maped"
    folder.mkdir(parents=True)
    for index, (x_degrees, y_degrees) in enumerate(_TILTS[:count]):
        stem = f"pos_38_tilt{index}_{x_degrees}x_{y_degrees}y"
        (folder / f"{stem}_master.h5").write_bytes(f"master-{index}".encode())
        (folder / f"{stem}_data_000001.h5").write_bytes(
            f"detector-member-{index}".encode()
        )
    return folder


def _inspector(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ready=True,
        reason=None,
        scan_shape=(4, 5),
        detector_shape=(3, 2),
        dtype="uint16",
        metadata={},
    )


def _service(root: Path, **kwargs) -> MAPEDProtocolService:
    return MAPEDProtocolService(
        root,
        available_gpus=lambda: [0, 1],
        device_name=lambda index: f"Test CUDA {index}",
        inspector=_inspector,
        implementation_revision="gpu-test-revision",
        **kwargs,
    )


def _inventory(service: MAPEDProtocolService, folder: Path) -> dict[str, object]:
    return service.inventory(
        {"contractVersion": CONTRACT_VERSION, "folderPath": str(folder)}
    )


def _run_request(
    inventory: dict[str, object],
    *,
    validation: dict[str, object] | None = None,
) -> dict[str, object]:
    tilts = inventory["tilts"]
    assert isinstance(tilts, list)
    return {
        "contractVersion": CONTRACT_VERSION,
        "algorithmVersion": ALGORITHM_VERSION,
        "runID": str(uuid4()),
        "collectionIdentitySHA256": inventory["collectionIdentitySHA256"],
        "mode": "automaticAlignment",
        "parameters": _parameters(),
        "backend": {
            "kind": "remote_cuda",
            "profile_id": "test-profile",
            "gpu_index": 0,
        },
        "implementationRevision": "gpu-test-revision",
        "orderedCalibrations": [item["calibration"] for item in tilts],
        "validation": validation or {"kind": "integrity_only"},
    }


def _alignment(request: dict[str, object]) -> dict[str, object]:
    coordinates = request["cacheIdentity"]["orderedTilts"]
    return {
        "realSpaceShifts": [
            {"tilt": tilt, "rowPixels": float(index), "columnPixels": -0.5}
            for index, tilt in enumerate(coordinates)
        ],
        "diffractionShifts": [
            {"tilt": tilt, "rowPixels": 0.25, "columnPixels": float(-index)}
            for index, tilt in enumerate(coordinates)
        ],
    }


def _outcome(
    request: dict[str, object],
    working_directory: Path,
    *,
    validation: dict[str, object] | None = None,
    reference_identity: dict[str, str] | None = None,
) -> dict[str, object]:
    output_path = working_directory / "merged.npy"
    np.save(output_path, np.arange(120, dtype=np.float32).reshape(4, 5, 3, 2))
    products_path = working_directory / "products.npz"
    np.savez(
        products_path,
        bright_field=np.ones((4, 5), dtype=np.float32),
        mean_diffraction=np.ones((3, 2), dtype=np.float32),
    )
    return {
        "outputFile": output_path.name,
        "outputSHA256": sha256(output_path.read_bytes()).hexdigest(),
        "outputShape": [4, 5, 3, 2],
        "outputDtype": "float32",
        "products": {
            "path": products_path.name,
            "sha256": sha256(products_path.read_bytes()).hexdigest(),
            "byteCount": products_path.stat().st_size,
        },
        "automaticAlignment": _alignment(request),
        "validation": validation or {"kind": "integrity_only", "passed": True},
        "referenceIdentity": reference_identity,
        "executedDevices": [
            {
                "backend": "cuda",
                "deviceName": "Test CUDA 0",
                "gpuIndex": 0,
                "driverVersion": "test",
                "runtimeVersion": "test",
                "implementationRevision": "gpu-test-revision",
            }
        ],
        "backendEnvironment": {
            "python": "test",
            "quantemRevision": "58eb7dad",
            "quantemGPURevision": "gpu-test-revision",
        },
        "stageTimings": [{"stage": "merge", "seconds": 0.01}],
        "algorithmTimings": {"mergeSeconds": 0.01},
        "coldRunSeconds": 0.02,
    }


def test_inventory_orders_sources_and_detects_additive_folder_changes(tmp_path):
    folder = _fixture_folder(tmp_path, count=2)
    service = _service(tmp_path)

    inventory = _inventory(service, folder)

    assert inventory["isRunnable"] is True
    assert [item["tilt"] for item in inventory["tilts"]] == [
        {"xDegrees": -17.0, "yDegrees": 0.0},
        {"xDegrees": -8.5, "yDegrees": -14.72},
    ]
    first = inventory["tilts"][0]
    assert first["sourceDtype"] == "uint16"
    assert len(first["sourceIdentity"]["masterSHA256"]) == 64
    assert len(first["sourceIdentity"]["orderedMemberSHA256"]) == 1
    expected_source = sha256()
    expected_source.update(b"live4dstem.dataset/v0.1\0")
    expected_source.update(first["sourceIdentity"]["masterSHA256"].encode())
    expected_source.update(b"\0")
    expected_source.update(
        first["sourceIdentity"]["orderedMemberSHA256"][0].encode()
    )
    assert (
        first["sourceIdentity"]["sourceIdentitySHA256"]
        == expected_source.hexdigest()
    )
    assert first["calibration"]["resolution"]["state"] == "missing"
    before = inventory["snapshot"]["snapshotToken"]
    assert inventory["snapshot"]["members"][0]["url"].endswith("_master.h5")

    stem = "pos_38_tilt2_-8.5x_14.72y"
    (folder / f"{stem}_master.h5").write_bytes(b"new-master")
    (folder / f"{stem}_data_000001.h5").write_bytes(b"new-member")

    after = service.folder_snapshot(folder)
    assert after["snapshotToken"] != before
    assert len(after["members"]) == 3


def test_typed_preview_and_latest_selected_diffraction_descriptors(tmp_path):
    folder = _fixture_folder(tmp_path, count=2)
    reads = []

    def previewer(_path: Path, gpu: int):
        assert gpu == 0
        return np.arange(20).reshape(4, 5), np.arange(6).reshape(3, 2)

    def diffraction_reader(path, row_start, row_stop, column_start, column_stop, gpu):
        reads.append((path.name, row_start, row_stop, column_start, column_stop, gpu))
        return np.full((3, 2), 7.5, dtype=np.float32)

    service = _service(
        tmp_path,
        previewer=previewer,
        diffraction_reader=diffraction_reader,
    )
    inventory = _inventory(service, folder)
    first = inventory["tilts"][0]
    base = {
        "contractVersion": CONTRACT_VERSION,
        "collectionIdentitySHA256": inventory["collectionIdentitySHA256"],
        "backend": {
            "kind": "remote_cuda",
            "profile_id": "test-profile",
            "gpu_index": 0,
        },
    }

    previews = service.previews({**base, "datasetIDs": [first["datasetID"]]})
    descriptor = previews["items"][0]["brightField"]
    payload, stored = service.payload(descriptor["payloadID"])
    assert descriptor["dtype"] == "float32"
    assert descriptor["shape"] == {"rows": 4, "columns": 5}
    assert descriptor["byteCount"] == 4 * 5 * 4
    assert descriptor["sha256"] == sha256(payload).hexdigest()
    assert descriptor["generation"] == stored["generation"]
    assert descriptor["sourceIdentity"] == first["sourceIdentity"]
    assert not payload.startswith(b"\x89PNG")

    selected = service.selected_diffraction(
        {
            **base,
            "clientID": "mac-window-1",
            "requestID": 4,
            "datasetID": first["datasetID"],
            "scan": {"row": 0, "column": 2},
            "averaging": {"width": 3, "aggregation": "mean"},
        }
    )
    assert selected["sampleBounds"] == {
        "rowStart": 0,
        "rowStop": 2,
        "columnStart": 1,
        "columnStop": 4,
    }
    assert selected["diffraction"]["generation"] > descriptor["generation"]
    assert reads[-1][1:] == (0, 2, 1, 4, 0)
    with pytest.raises(MAPEDProtocolError, match="superseded"):
        service.selected_diffraction(
            {
                **base,
                "clientID": "mac-window-1",
                "requestID": 3,
                "datasetID": first["datasetID"],
                "scan": {"row": 1, "column": 1},
                "averaging": {"width": 1, "aggregation": "mean"},
            }
        )


def test_cache_identity_binds_nullable_ordered_calibrations(tmp_path):
    folder = _fixture_folder(tmp_path, count=2)
    service = _service(tmp_path)
    inventory = _inventory(service, folder)
    request = _run_request(inventory)
    request["orderedCalibrations"][0] = None

    identity = service.cache_identity(request)
    first_hash = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert set(identity) == {
        "schemaVersion",
        "algorithmVersion",
        "orderedSources",
        "orderedCalibrations",
        "orderedTilts",
        "parameters",
        "mode",
        "backend",
        "implementationRevision",
    }
    assert identity["schemaVersion"] == CACHE_VERSION
    assert identity["orderedCalibrations"][0] is None
    changed = json.loads(json.dumps(request))
    changed["orderedCalibrations"][1]["resolution"]["reason"] = "User cleared it."
    second_identity = service.cache_identity(changed)
    second_hash = sha256(
        json.dumps(second_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first_hash != second_hash


def test_run_persists_alignment_and_distinguishes_cold_from_cached(tmp_path):
    folder = _fixture_folder(tmp_path, count=2)
    calls = []

    def runner(request, progress, _cancel_requested, working_directory):
        calls.append(request["runID"])
        progress(
            {
                "stage": "merge",
                "completedUnits": 1,
                "totalUnits": 1,
                "detail": "Merged test tilts",
                "elapsedSeconds": 0.01,
            }
        )
        return _outcome(request, working_directory)

    service = _service(tmp_path, runner=runner)
    inventory = _inventory(service, folder)
    request = _run_request(inventory)

    accepted = service.start_run(request)
    events = list(service.run_events(accepted["runID"]))
    completed = events[-1]

    assert completed["type"] == "completed"
    assert any(
        event.get("type") == "progress" and event.get("stage") == "ready"
        for event in events
    )
    assert completed["receipt"]["coldRunSeconds"] == pytest.approx(0.02)
    assert completed["receipt"]["cachedOpenSeconds"] is None
    cache = completed["receipt"]["cache"]
    assert cache["validation"] == {"kind": "integrity_only", "passed": True}
    assert "parity" not in cache["validation"]
    result_bf = completed["receipt"]["resultProducts"]["brightField"]
    assert result_bf["dtype"] == "float32"
    assert result_bf["shape"] == {"rows": 4, "columns": 5}
    assert result_bf["sourceIdentity"]["kind"] == "maped_result"
    assert [
        value["tilt"] for value in cache["automaticAlignment"]["realSpaceShifts"]
    ] == cache["identity"]["orderedTilts"]
    assert not list(folder.glob("maped-results/.*.incomplete"))

    cached_request = {**request, "runID": str(uuid4())}
    cached = service.start_run(cached_request)
    cached_events = list(service.run_events(cached["runID"]))
    cached_receipt = cached_events[-1]["receipt"]
    assert cached_events[-1]["type"] == "completed"
    assert cached_events[-2]["stage"] == "ready"
    assert cached_receipt["cachedOpenSeconds"] >= 0
    assert cached_receipt["coldRunSeconds"] == pytest.approx(0.02)
    assert len(calls) == 1
    assert (
        cached_receipt["resultProducts"]["brightField"]["generation"]
        > result_bf["generation"]
    )


def test_reference_parity_is_explicit_and_never_inferred(tmp_path):
    folder = _fixture_folder(tmp_path, count=2)
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"validated-control")
    reference_hash = sha256(reference.read_bytes()).hexdigest()
    parity = {
        "sampleCount": 16,
        "sampleMeanAbsoluteError": 0.0,
        "sampleMaximumAbsoluteError": 0.0,
        "brightFieldMeanAbsoluteError": 0.0,
        "brightFieldMaximumAbsoluteError": 0.0,
        "meanDiffractionMeanAbsoluteError": 0.0,
        "meanDiffractionMaximumAbsoluteError": 0.0,
        "passed": True,
    }

    def runner(request, _progress, _cancel_requested, working_directory):
        return _outcome(
            request,
            working_directory,
            validation={"kind": "reference_parity", "passed": True, "parity": parity},
            reference_identity={
                "path": str(reference.resolve()),
                "sha256": reference_hash,
            },
        )

    service = _service(tmp_path, runner=runner)
    inventory = _inventory(service, folder)
    with pytest.raises(MAPEDProtocolError, match="cannot carry"):
        service.cache_identity(
            _run_request(
                inventory,
                validation={
                    "kind": "integrity_only",
                    "referenceSHA256": reference_hash,
                },
            )
        )
    with pytest.raises(MAPEDProtocolError, match="exact reference"):
        service.cache_identity(
            _run_request(
                inventory,
                validation={
                    "kind": "reference_parity",
                    "referencePath": str(reference),
                },
            )
        )

    request = _run_request(
        inventory,
        validation={
            "kind": "reference_parity",
            "referencePath": str(reference),
            "referenceSHA256": reference_hash,
        },
    )
    accepted = service.start_run(request)
    receipt = list(service.run_events(accepted["runID"]))[-1]["receipt"]
    assert receipt["cache"]["validation"] == {
        "kind": "reference_parity",
        "passed": True,
        "parity": parity,
    }

    different_reference = {**request["validation"], "referenceSHA256": "f" * 64}
    identity = service.cache_identity(request)
    miss = service.validate_cache(
        {
            "cacheIdentity": identity,
            "cacheIdentitySHA256": sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "validation": different_reference,
        }
    )
    assert miss["state"] == "miss"
    assert "requested reference" in miss["reason"]


def test_cancel_terminal_waits_for_runner_stop_and_cleanup(tmp_path):
    folder = _fixture_folder(tmp_path, count=2)
    entered = threading.Event()
    cancel_seen = threading.Event()
    allow_stop = threading.Event()

    def runner(request, _progress, cancel_requested, working_directory):
        (working_directory / "partial.bin").write_bytes(b"incomplete")
        entered.set()
        assert cancel_requested.wait(timeout=2)
        cancel_seen.set()
        assert allow_stop.wait(timeout=2)
        return _outcome(request, working_directory)

    service = _service(tmp_path, runner=runner)
    inventory = _inventory(service, folder)
    accepted = service.start_run(_run_request(inventory))
    assert entered.wait(timeout=2)
    with pytest.raises(MAPEDProtocolError, match="already active"):
        service.start_run(_run_request(inventory))

    response = service.cancel_run(accepted["runID"])
    assert response["state"] == "cancellation_requested"
    assert cancel_seen.wait(timeout=2)
    assert not any(
        event["type"] in {"cancelled", "completed", "failed"}
        for event in service.run_event_snapshot(accepted["runID"])
    )
    allow_stop.set()

    events = list(service.run_events(accepted["runID"]))
    assert events[-1]["type"] == "cancelled"
    assert "worker stopped" in events[-1]["detail"]
    assert not list(folder.glob("maped-results/.*.incomplete"))


def test_http_endpoints_advertise_maped_and_serve_typed_payload(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from quantem.gpu.remote.server import BrowseService, create_app

    folder = _fixture_folder(tmp_path, count=2)
    maped = _service(
        tmp_path,
        previewer=lambda _path, _gpu: (
            np.ones((4, 5), dtype=np.float32),
            np.ones((3, 2), dtype=np.float32),
        ),
    )
    browse = BrowseService(tmp_path, gpu=0, initialize_cuda=False)
    browse.backend = "cuda"
    browse.gpus = [0]
    browse.device_name = "Test CUDA 0"
    app = create_app(
        tmp_path,
        service=browse,
        maped_service=maped,
        implementation_revision="gpu-test-revision",
    )
    client = TestClient(app)

    assert app.state.ssb_service.implementation_revision == "gpu-test-revision"
    assert app.state.maped_service.implementation_revision == "gpu-test-revision"

    capabilities = client.get("/api/browse/capabilities").json()
    assert capabilities["features"]["maped"]["contractVersion"] == CONTRACT_VERSION
    assert capabilities["features"]["maped"]["previewPayload"] == (
        "scientific_float32_array"
    )

    inventory = client.post(
        "/api/maped/inventory",
        json={"contractVersion": CONTRACT_VERSION, "folderPath": str(folder)},
    ).json()
    preview = client.post(
        "/api/maped/previews",
        json={
            "contractVersion": CONTRACT_VERSION,
            "collectionIdentitySHA256": inventory["collectionIdentitySHA256"],
            "datasetIDs": [inventory["tilts"][0]["datasetID"]],
            "backend": {
                "kind": "remote_cuda",
                "profile_id": "test-profile",
                "gpu_index": 0,
            },
        },
    ).json()
    descriptor = preview["items"][0]["brightField"]
    payload = client.get(descriptor["path"])

    assert payload.status_code == 200
    assert payload.headers["x-dtype"] == "float32"
    assert payload.headers["x-byte-count"] == str(descriptor["byteCount"])
    assert payload.headers["x-sha256"] == descriptor["sha256"]
    assert payload.headers["x-generation"] == str(descriptor["generation"])
    assert payload.content == np.ones((4, 5), dtype="<f4").tobytes()
