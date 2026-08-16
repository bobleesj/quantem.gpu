from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from quantem.gpu import detector
from quantem.gpu.cli import main
from quantem.gpu.remote.server import (
    BrowseService,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    _wire_image,
    create_app,
)


def _master(
    root: Path,
    name: str = "sample_00_master.h5",
    *,
    dtype: type[np.unsignedinteger] = np.uint16,
    session: str = "arina/20260815_session",
) -> Path:
    h5py = pytest.importorskip("h5py")
    session_path = root / session
    session_path.mkdir(parents=True, exist_ok=True)
    path = session_path / name
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.arange(4 * 4 * 4, dtype=dtype).reshape(4, 4, 4),
        )
        specific = handle.create_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=4)
        specific.create_dataset("nimages", data=1)
        specific.create_dataset("x_pixels_in_detector", data=4)
        specific.create_dataset("y_pixels_in_detector", data=4)
    return path


def _service(root: Path) -> BrowseService:
    service = BrowseService(root, gpu=2, initialize_cuda=False)
    service.backend = "cuda"
    service.device_name = "Test CUDA"
    service.cache_budget_bytes = 1 << 30
    return service


def _externally_linked_master(root: Path) -> tuple[Path, Path]:
    h5py = pytest.importorskip("h5py")
    session = root / "arina" / "20260815_linked"
    session.mkdir(parents=True, exist_ok=True)
    shard = session / "detector_payload.h5"
    with h5py.File(shard, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4),
        )
    master = session / "sample_master.h5"
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry/data")
        data["data_000001"] = h5py.ExternalLink(shard.name, "/entry/data/data")
        specific = handle.create_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=4)
        specific.create_dataset("nimages", data=1)
        specific.create_dataset("x_pixels_in_detector", data=4)
        specific.create_dataset("y_pixels_in_detector", data=4)
    return master, shard


def _multi_service(root: Path) -> BrowseService:
    service = BrowseService(root, gpus=[0, 1], initialize_cuda=False)
    service.backend = "cuda"
    service.gpus = [0, 1]
    service.gpu = 0
    service._devices = {0: _RecordingDevice(), 1: _RecordingDevice()}
    service._device_names = {0: "Test CUDA 0", 1: "Test CUDA 1"}
    service._cache_budgets = {0: 100, 1: 100}
    service.device = service._devices[0]
    service.device_name = service._device_names[0]
    service.cache_budget_bytes = 100
    service.aggregate_cache_budget_bytes = 200
    return service


class _RecordingDevice:
    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_capabilities_identify_quantem_gpu_protocol(tmp_path):
    service = _service(tmp_path)
    client = TestClient(create_app(tmp_path, service=service))

    payload = client.get("/api/browse/capabilities").json()

    assert payload["protocol"] == PROTOCOL_NAME
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["backend"] == "cuda"
    assert payload["browse_gpu"] == 2
    assert payload["browse_gpus"] == [2]
    assert payload["data_folders"] == [str(tmp_path)]
    assert payload["features"]["exact_integer_images"] is True
    assert payload["features"]["multi_gpu_residency"] is False


def test_multi_gpu_cache_places_whole_datasets_and_reuses_hits(tmp_path, monkeypatch):
    first = _master(tmp_path, "sample_00_master.h5")
    second = _master(tmp_path, "sample_01_master.h5")
    service = _multi_service(tmp_path)
    service.refresh_catalog()
    loads: list[tuple[str, int]] = []

    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))

    def fake_load(path, _det_bin, _scan_bin, _scan_region, gpu):
        loads.append((path.name, gpu))
        return {"gpu": gpu, "data": np.zeros(10, dtype=np.uint32)}

    monkeypatch.setattr(service, "_load_entry", fake_load)
    session = "arina/20260815_session"

    first_key, first_entry = service.entry(
        session, first.name, det_bin=1, scan_bin=1, scan_region=None
    )
    second_key, second_entry = service.entry(
        session, second.name, det_bin=1, scan_bin=1, scan_region=None
    )
    hit_key, hit_entry = service.entry(
        session, first.name, det_bin=1, scan_bin=1, scan_region=None
    )

    assert first_entry["gpu"] == 0
    assert second_entry["gpu"] == 1
    assert hit_key == first_key
    assert hit_entry is first_entry
    assert first_key != second_key
    assert loads == [(first.name, 0), (second.name, 1)]
    devices = service.capabilities()["devices"]
    assert [(device["index"], device["resident_entries"]) for device in devices] == [
        (0, 1),
        (1, 1),
    ]


def test_full_scan_region_uses_the_uncropped_loader_path(tmp_path, monkeypatch):
    master = _master(tmp_path)
    service = _multi_service(tmp_path)
    service.refresh_catalog()
    loaded_regions = []
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))

    def fake_load(_path, _det_bin, _scan_bin, scan_region, gpu):
        loaded_regions.append(scan_region)
        return {"gpu": gpu, "data": np.zeros(10, dtype=np.uint32)}

    monkeypatch.setattr(service, "_load_entry", fake_load)

    key, _ = service.entry(
        "arina/20260815_session",
        master.name,
        det_bin=1,
        scan_bin=1,
        scan_region=(0, 2, 0, 2),
    )

    assert key[-1] is None
    assert loaded_regions == [None]


def test_multi_gpu_lru_evicts_only_the_selected_device(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(3)]
    service = _multi_service(tmp_path)
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda _path, _det_bin, _scan_bin, _scan_region, gpu: {
            "gpu": gpu,
            "data": np.zeros(20, dtype=np.uint32),
        },
    )
    session = "arina/20260815_session"

    entries = []
    placements = []
    for path in files:
        loaded = service.entry(session, path.name, det_bin=1, scan_bin=1, scan_region=None)
        entries.append(loaded)
        placements.append(loaded[1]["gpu"])

    assert placements == [0, 1, 0]
    assert entries[0][0] not in service._master_cache
    assert entries[1][0] in service._master_cache
    assert entries[2][0] in service._master_cache


def test_multi_gpu_cache_preserves_the_active_dataset(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(3)]
    service = _multi_service(tmp_path)
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda _path, _det_bin, _scan_bin, _scan_region, gpu: {
            "gpu": gpu,
            "data": np.zeros(20, dtype=np.uint32),
        },
    )
    session = "arina/20260815_session"
    first = service.entry(
        session, files[0].name, det_bin=1, scan_bin=1, scan_region=None
    )
    service._active_master_key = first[0]
    second = service.entry(
        session, files[1].name, det_bin=1, scan_bin=1, scan_region=None
    )
    third = service.entry(
        session, files[2].name, det_bin=1, scan_bin=1, scan_region=None
    )

    assert first[0] in service._master_cache
    assert second[0] not in service._master_cache
    assert third[0] in service._master_cache
    assert service._master_cache[first[0]]["gpu"] == 0
    assert service._master_cache[third[0]]["gpu"] == 1


def test_catalog_and_acquisition_status_use_quantem_gpu_inspection(tmp_path):
    master = _master(tmp_path)
    service = _service(tmp_path)
    client = TestClient(create_app(tmp_path, service=service))

    catalog = client.get("/api/browse/sessions").json()
    status = client.get("/api/browse/acquisitions").json()

    assert catalog["complete"] is True
    assert catalog["sessions"][0]["source"] == "arina"
    assert catalog["sessions"][0]["date"] == "20260815_session"
    assert catalog["sessions"][0]["path"] == "arina/20260815_session"
    assert catalog["sessions"][0]["files"][0]["name"] == master.name
    assert catalog["sessions"][0]["files"][0]["shape"] == [2, 2, 4, 4]
    assert catalog["sessions"][0]["files"][0]["loadable"] is True
    assert status["pending"] == []
    assert status["history"][0]["path"] == str(master)
    assert status["ready_token"]


def test_catalog_preserves_nested_paths_that_share_the_same_leaf_names(tmp_path):
    first = _master(
        tmp_path,
        "first_master.h5",
        session="collaborator-a/project/shared-session",
    )
    second = _master(
        tmp_path,
        "second_master.h5",
        session="collaborator-b/project/shared-session",
    )
    service = _service(tmp_path)

    catalog = service.refresh_catalog()

    assert [item["path"] for item in catalog["sessions"]] == [
        "collaborator-a/project/shared-session",
        "collaborator-b/project/shared-session",
    ]
    assert service.resolve_master(catalog["sessions"][0]["path"], first.name) == first
    assert service.resolve_master(catalog["sessions"][1]["path"], second.name) == second
    with pytest.raises(HTTPException) as error:
        service.resolve_master("project/shared-session", first.name)
    assert error.value.status_code == 404


def test_catalog_follows_nonstandard_external_shard_names(tmp_path):
    master, shard = _externally_linked_master(tmp_path)
    service = _service(tmp_path)

    catalog = service.refresh_catalog()
    item = catalog["sessions"][0]["files"][0]

    assert item["loadable"] is True
    assert item["size_bytes"] == master.stat().st_size + shard.stat().st_size


def test_exact_virtual_image_and_selected_diffraction_share_resident_plan(tmp_path):
    master = _master(tmp_path)
    service = _service(tmp_path)
    device = _RecordingDevice()
    service.device = device
    service.refresh_catalog()
    data = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    compute = detector.prepare(data)
    key = service._plan_key(master, 1, 1, None)
    service._master_cache[key] = {
        "data": data,
        "compute": compute,
        "scan_shape": (2, 2),
        "detector_shape": (4, 4),
        "mean_dp": compute.mean_dp(),
        "bf_geometry": None,
        "com_row": None,
        "com_column": None,
    }
    client = TestClient(create_app(tmp_path, service=service))
    common = {
        "session": "arina/20260815_session",
        "file": master.name,
        "det_bin": 1,
        "scan_bin": 1,
    }

    bright_field = client.get(
        "/api/browse/realspace",
        params={**common, "mode": "BF", "inner": 0, "outer": 1},
    )
    diffraction = client.get(
        "/api/browse/cbed",
        params={**common, "sx": 1, "sy": 0},
    )

    assert bright_field.status_code == 200
    assert bright_field.headers["x-dtype"] == "<u4"
    assert np.frombuffer(bright_field.content, dtype="<u4").size == 4
    assert diffraction.status_code == 200
    assert diffraction.headers["x-dtype"] == "<u4"
    np.testing.assert_array_equal(
        np.frombuffer(diffraction.content, dtype="<u4").reshape(4, 4),
        data[1, 0],
    )
    assert device.entries >= 2


def test_custom_annulus_returns_exact_counts(tmp_path):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    data = np.ones((2, 2, 4, 4), dtype=np.uint16)
    compute = detector.prepare(data)
    key = service._plan_key(master, 1, 1, None)
    service._master_cache[key] = {
        "data": data,
        "compute": compute,
        "scan_shape": (2, 2),
        "detector_shape": (4, 4),
        "mean_dp": compute.mean_dp(),
        "bf_geometry": None,
        "com_row": None,
        "com_column": None,
    }
    client = TestClient(create_app(tmp_path, service=service))

    response = client.get(
        "/api/browse/realspace-shape",
        params={
            "session": "arina/20260815_session",
            "file": master.name,
            "shape": "annulus",
            "cx": 1.5,
            "cy": 1.5,
            "inner": 0,
            "outer": 1.6,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-dtype"] == "<u4"
    values = np.frombuffer(response.content, dtype="<u4")
    assert values.tolist() == [12, 12, 12, 12]


def test_oversized_plan_is_rejected_before_loading(tmp_path, monkeypatch):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    service.cache_budget_bytes = 1
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda *_args, **_kwargs: pytest.fail("oversized plan reached CUDA loading"),
    )
    client = TestClient(create_app(tmp_path, service=service))

    response = client.get(
        "/api/browse/realspace",
        params={
            "session": "arina/20260815_session",
            "file": master.name,
            "mode": "BF",
        },
    )

    assert response.status_code == 413
    assert "Crop the scan region" in response.text


def test_uint32_plan_reserves_exact_native_counts(tmp_path):
    master = _master(tmp_path, dtype=np.uint32)
    service = _service(tmp_path)
    service.refresh_catalog()

    resident_bytes, peak_bytes = service._plan_bytes(master, 1, 1, None)

    assert resident_bytes == 2 * 2 * 4 * 4 * np.dtype(np.uint32).itemsize
    assert peak_bytes == resident_bytes
    assert service.sessions()["sessions"][0]["files"][0]["dtype"] == "uint32"


def test_wire_image_preserves_counts_above_float32_precision():
    source = np.asarray([[16_777_217, 40_000_001]], dtype=np.uint64)

    payload, dtype = _wire_image(source)

    assert dtype == "<u4"
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<u4"), source.ravel())


def test_server_source_has_no_quantem_live_dependency():
    source = Path(__file__).parents[2] / "src/quantem/gpu/remote/server.py"
    assert "quantem.live" not in source.read_text()


def test_cli_rejects_negative_gpu_before_launching():
    with pytest.raises(SystemExit, match="--gpu must be zero or greater"):
        main(["serve", "/data", "--gpu", "-1"])


def test_cli_rejects_invalid_gpu_pool_before_launching():
    with pytest.raises(SystemExit, match="comma-separated CUDA indices"):
        main(["serve", "/data", "--gpus", "0,nope"])
