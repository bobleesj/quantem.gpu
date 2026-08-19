from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from quantem.gpu import detector
from quantem.gpu.cli import main
from quantem.gpu.remote.server import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BrowseService,
    _scan_roi_indices,
    _wire_image,
    create_app,
)


def _master(
    root: Path,
    name: str = "sample_00_master.h5",
    *,
    dtype: type[np.unsignedinteger] = np.uint16,
    session: str = "detector/20260101_session",
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
    session = root / "detector" / "20260101_linked"
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
    assert payload["cache_fraction"] == pytest.approx(0.80)
    assert payload["data_folders"] == [str(tmp_path)]
    assert payload["features"]["exact_integer_images"] is True
    assert payload["features"]["multi_gpu_residency"] is False


def test_capabilities_report_live_admission_bytes(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service.cache_budget_bytes = 100
    active_key = ("active",)
    service._master_cache[active_key] = {
        "gpu": 2,
        "data": np.zeros(30, dtype=np.uint8),
    }
    service._master_cache[("evictable",)] = {
        "gpu": 2,
        "data": np.zeros(20, dtype=np.uint8),
    }
    service.mark_active(active_key)
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: 10)

    device = service.capabilities()["devices"][0]

    assert device["resident_bytes"] == 50
    assert device["active_resident_bytes"] == 30
    assert device["evictable_bytes"] == 20
    assert device["available_peak_bytes"] == 30
    assert device["available_resident_bytes"] == 70


def test_capabilities_use_budget_when_free_memory_is_unknown(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service.cache_budget_bytes = 100
    active_key = ("active",)
    service._master_cache[active_key] = {
        "gpu": 2,
        "data": np.zeros(30, dtype=np.uint8),
    }
    service.mark_active(active_key)
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: None)

    device = service.capabilities()["devices"][0]

    assert device["available_peak_bytes"] == 100
    assert device["available_resident_bytes"] == 70


def test_advertised_capacity_matches_admission_boundaries(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service.cache_budget_bytes = 100
    active_key = ("active",)
    service._master_cache[active_key] = {
        "gpu": 2,
        "data": np.zeros(30, dtype=np.uint8),
    }
    service._master_cache[("evictable",)] = {
        "gpu": 2,
        "data": np.zeros(20, dtype=np.uint8),
    }
    service.mark_active(active_key)
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: 10)
    device = service.capabilities()["devices"][0]

    assert service._candidate_gpus(
        device["available_resident_bytes"], device["available_peak_bytes"]
    ) == [2]
    assert service._candidate_gpus(
        device["available_resident_bytes"] + 1, device["available_peak_bytes"]
    ) == []
    assert service._candidate_gpus(
        device["available_resident_bytes"], device["available_peak_bytes"] + 1
    ) == []


def test_active_dataset_on_one_gpu_preserves_capacity_on_another(tmp_path, monkeypatch):
    service = _multi_service(tmp_path)
    active_key = ("active",)
    service._master_cache[active_key] = {
        "gpu": 0,
        "data": np.zeros(60, dtype=np.uint8),
    }
    service.mark_active(active_key)
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: 100)

    devices = service.capabilities()["devices"]

    assert devices[0]["available_resident_bytes"] == 40
    assert devices[1]["available_resident_bytes"] == 100
    assert service._candidate_gpus(resident_bytes=50, peak_bytes=70) == [1]


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
    session = "detector/20260101_session"

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


def test_serialized_cuda_loads_reuse_one_host_worker(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(2)]
    service = _service(tmp_path)
    service.refresh_catalog()
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: None)
    load_threads: list[tuple[int, str]] = []

    def fake_load(_path, _det_bin, _scan_bin, _scan_region, gpu):
        load_threads.append((threading.get_ident(), threading.current_thread().name))
        return {"gpu": gpu, "data": np.zeros(10, dtype=np.uint32)}

    monkeypatch.setattr(service, "_load_entry", fake_load)
    try:
        for path in files:
            service.entry(
                "detector/20260101_session",
                path.name,
                det_bin=1,
                scan_bin=1,
                scan_region=None,
            )
    finally:
        service._load_executor.shutdown(wait=True, cancel_futures=True)

    assert len({thread_id for thread_id, _ in load_threads}) == 1
    assert all(name.startswith("quantem-cuda-load") for _, name in load_threads)


def test_app_shutdown_closes_service_resources(tmp_path, monkeypatch):
    class Compute:
        closed = False

        def close(self) -> None:
            self.closed = True

    service = _service(tmp_path)
    compute = Compute()
    service._master_cache[("fixture",)] = {"compute": compute, "gpu": 2}
    original_close = service.close
    closed = False

    def close() -> None:
        nonlocal closed
        original_close()
        closed = True

    monkeypatch.setattr(service, "close", close)

    with TestClient(create_app(tmp_path, service=service)) as client:
        assert not closed
        assert client.get("/api/browse/capabilities").status_code == 200
        loader_thread = service._load_executor.submit(threading.current_thread).result()
        assert loader_thread.is_alive()

    assert closed
    assert not loader_thread.is_alive()
    assert compute.closed
    assert not service._master_cache


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
        "detector/20260101_session",
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
    session = "detector/20260101_session"

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
    session = "detector/20260101_session"
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


def test_cache_evicts_active_dataset_when_it_is_the_only_valid_transition(
    tmp_path,
    monkeypatch,
):
    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path)
    service.cache_budget_bytes = 100
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
    session = "detector/20260101_session"

    first_key, _ = service.entry(
        session, first.name, det_bin=1, scan_bin=1, scan_region=None
    )
    service._active_master_key = first_key
    second_key, _ = service.entry(
        session, second.name, det_bin=1, scan_bin=1, scan_region=None
    )

    assert first_key not in service._master_cache
    assert second_key in service._master_cache
    assert service._active_master_key is None


def test_reserved_entry_stays_resident_until_request_finishes(tmp_path, monkeypatch):
    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path)
    service.cache_budget_bytes = 100
    service.refresh_catalog()
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: None)
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda _path, _det_bin, _scan_bin, _scan_region, gpu: {
            "gpu": gpu,
            "data": np.zeros(20, dtype=np.uint32),
        },
    )
    session = "detector/20260101_session"
    first_key, first_entry = service.entry(
        session,
        first.name,
        det_bin=1,
        scan_bin=1,
        scan_region=None,
        reserve=True,
    )
    second_complete = threading.Event()

    def load_second():
        try:
            return service.entry(
                session,
                second.name,
                det_bin=1,
                scan_bin=1,
                scan_region=None,
            )
        finally:
            second_complete.set()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(load_second)
            try:
                assert second_complete.wait(0.1) is False
                assert first_key in service._master_cache
            finally:
                service._release_entry(first_entry)
            second_key, _ = future.result(timeout=2)
    finally:
        service._load_executor.shutdown(wait=True, cancel_futures=True)

    assert first_key not in service._master_cache
    assert second_key in service._master_cache


def test_same_file_plan_change_waits_for_reserved_interaction(tmp_path, monkeypatch):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: None)
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda _path, _det_bin, _scan_bin, _scan_region, gpu: {
            "gpu": gpu,
            "data": np.zeros(10, dtype=np.uint32),
        },
    )
    session = "detector/20260101_session"
    first_key, first_entry = service.entry(
        session,
        master.name,
        det_bin=1,
        scan_bin=1,
        scan_region=None,
        reserve=True,
    )
    replacement_complete = threading.Event()

    def replace_plan():
        try:
            return service.entry(
                session,
                master.name,
                det_bin=2,
                scan_bin=1,
                scan_region=None,
            )
        finally:
            replacement_complete.set()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(replace_plan)
            try:
                assert replacement_complete.wait(0.1) is False
                assert first_key in service._master_cache
            finally:
                service._release_entry(first_entry)
            second_key, _ = future.result(timeout=2)
    finally:
        service._load_executor.shutdown(wait=True, cancel_futures=True)

    assert first_key not in service._master_cache
    assert second_key in service._master_cache


def test_interactive_compute_pins_entry_until_concurrent_eviction_finishes(
    tmp_path,
    monkeypatch,
):
    class Compute:
        def __init__(self) -> None:
            self.closed = False

        def masked_sum_exact(self, _mask):
            assert self.closed is False
            return np.arange(4, dtype=np.uint64)

        def close(self) -> None:
            self.closed = True

    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path)
    service.cache_budget_bytes = 100
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (80, 90))
    first_compute = Compute()

    def fake_load(path, _det_bin, _scan_bin, _scan_region, gpu):
        compute = first_compute if path == first else Compute()
        return {
            "gpu": gpu,
            "data": np.zeros(20, dtype=np.uint32),
            "compute": compute,
            "scan_shape": (2, 2),
            "detector_shape": (2, 2),
            "mean_dp": np.zeros((2, 2), dtype=np.float32),
            "bf_geometry": (0.5, 0.5, 1.0),
        }

    monkeypatch.setattr(service, "_load_entry", fake_load)
    geometry_started = threading.Event()
    release_geometry = threading.Event()

    def blocking_geometry(_entry):
        geometry_started.set()
        assert release_geometry.wait(2)
        return (0.5, 0.5, 1.0)

    monkeypatch.setattr(service, "_bf_geometry", blocking_geometry)
    session = "detector/20260101_session"
    first_key, first_entry = service.entry(
        session, first.name, det_bin=1, scan_bin=1, scan_region=None
    )
    second_complete = threading.Event()

    def load_second():
        try:
            return service.entry(
                session, second.name, det_bin=1, scan_bin=1, scan_region=None
            )
        finally:
            second_complete.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        image_future = pool.submit(
            service.virtual_image,
            first_key,
            first_entry,
            mode="BF",
            inner=0.0,
            outer=1.0,
        )
        assert geometry_started.wait(2)
        load_future = pool.submit(load_second)

        assert second_complete.wait(0.1) is False
        assert first_compute.closed is False
        release_geometry.set()
        image_result = image_future.result(timeout=2)
        load_future.result(timeout=2)

    assert np.array_equal(image_result, np.arange(4, dtype=np.uint64).reshape(2, 2))
    assert first_compute.closed is True
    assert first_key not in service._master_cache


def test_busy_entry_does_not_block_another_resident_dataset(tmp_path, monkeypatch):
    class Compute:
        def masked_sum_exact(self, _mask):
            return np.arange(4, dtype=np.uint64)

    service = _service(tmp_path)
    first_key = ("first", 1, 1, None)
    second_key = ("second", 1, 1, None)

    def resident(key):
        return {
            "cache_key": key,
            "gpu": 2,
            "data": np.zeros(4, dtype=np.uint16),
            "compute": Compute(),
            "scan_shape": (2, 2),
            "detector_shape": (2, 2),
            "mean_dp": np.zeros((2, 2), dtype=np.float32),
            "bf_geometry": (0.5, 0.5, 1.0),
        }

    first_entry = resident(first_key)
    second_entry = resident(second_key)
    service._master_cache[first_key] = first_entry
    service._master_cache[second_key] = second_entry
    geometry_started = threading.Event()
    release_geometry = threading.Event()

    def blocking_geometry(entry):
        if entry is first_entry:
            geometry_started.set()
            assert release_geometry.wait(2)
        return (0.5, 0.5, 1.0)

    monkeypatch.setattr(service, "_bf_geometry", blocking_geometry)
    original_entry_lock = service._entry_lock
    first_lock_calls = 0
    second_reader_started = threading.Event()

    def recording_entry_lock(key):
        nonlocal first_lock_calls
        if key == first_key:
            first_lock_calls += 1
            if first_lock_calls == 2:
                second_reader_started.set()
        return original_entry_lock(key)

    monkeypatch.setattr(service, "_entry_lock", recording_entry_lock)

    def image(key, entry):
        return service.virtual_image(
            key,
            entry,
            mode="BF",
            inner=0.0,
            outer=1.0,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(image, first_key, first_entry)
        assert geometry_started.wait(2)
        queued = pool.submit(image, first_key, first_entry)
        assert second_reader_started.wait(2)

        independent = pool.submit(image, second_key, second_entry)
        np.testing.assert_array_equal(
            independent.result(timeout=0.5),
            np.arange(4, dtype=np.uint64).reshape(2, 2),
        )
        release_geometry.set()
        first.result(timeout=2)
        queued.result(timeout=2)


def test_free_memory_headroom_evicts_inactive_entry_before_load(
    tmp_path,
    monkeypatch,
):
    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path)
    service.cache_budget_bytes = 1_000
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))
    free_bytes = [70]
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: free_bytes[0])
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda _path, _det_bin, _scan_bin, _scan_region, gpu: {
            "gpu": gpu,
            "data": np.zeros(10, dtype=np.uint32),
        },
    )
    session = "detector/20260101_session"

    first_key, _ = service.entry(
        session, first.name, det_bin=1, scan_bin=1, scan_region=None
    )
    free_bytes[0] = 25
    second_key, _ = service.entry(
        session, second.name, det_bin=1, scan_bin=1, scan_region=None
    )

    assert first_key not in service._master_cache
    assert second_key in service._master_cache


def test_out_of_memory_evicts_another_entry_and_retries_same_gpu(
    tmp_path,
    monkeypatch,
):
    import weakref

    class LoaderTemporary:
        pass

    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path)
    service.cache_budget_bytes = 1_000
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))
    calls: list[str] = []
    temporary_ref: weakref.ReferenceType[LoaderTemporary] | None = None

    def fake_load(path, _det_bin, _scan_bin, _scan_region, gpu):
        nonlocal temporary_ref
        calls.append(path.name)
        if path == second and calls.count(second.name) == 1:
            temporary = LoaderTemporary()
            temporary_ref = weakref.ref(temporary)
            raise MemoryError("simulated CUDA allocation failure")
        if path == second:
            assert temporary_ref is not None
            assert temporary_ref() is None
        return {"gpu": gpu, "data": np.zeros(10, dtype=np.uint32)}

    monkeypatch.setattr(service, "_load_entry", fake_load)
    session = "detector/20260101_session"

    first_key, _ = service.entry(
        session, first.name, det_bin=1, scan_bin=1, scan_region=None
    )
    second_key, _ = service.entry(
        session, second.name, det_bin=1, scan_bin=1, scan_region=None
    )

    assert calls == [first.name, second.name, second.name]
    assert first_key not in service._master_cache
    assert second_key in service._master_cache


def test_out_of_memory_retry_preserves_active_entry(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(3)]
    service = _service(tmp_path)
    service.cache_budget_bytes = 1_000
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))
    target_calls = 0

    def fake_load(path, _det_bin, _scan_bin, _scan_region, gpu):
        nonlocal target_calls
        if path == files[2]:
            target_calls += 1
            if target_calls == 1:
                raise MemoryError("simulated CUDA allocation failure")
        return {"gpu": gpu, "data": np.zeros(10, dtype=np.uint32)}

    monkeypatch.setattr(service, "_load_entry", fake_load)
    session = "detector/20260101_session"
    active_key, _ = service.entry(
        session, files[0].name, det_bin=1, scan_bin=1, scan_region=None
    )
    inactive_key, _ = service.entry(
        session, files[1].name, det_bin=1, scan_bin=1, scan_region=None
    )
    service._active_master_key = active_key

    target_key, _ = service.entry(
        session, files[2].name, det_bin=1, scan_bin=1, scan_region=None
    )

    assert target_calls == 2
    assert active_key in service._master_cache
    assert inactive_key not in service._master_cache
    assert target_key in service._master_cache


def test_out_of_memory_error_does_not_retain_the_loader_exception(
    tmp_path,
    monkeypatch,
):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    monkeypatch.setattr(service, "_plan_bytes", lambda *_args: (40, 60))
    monkeypatch.setattr(
        service,
        "_load_entry",
        lambda *_args: (_ for _ in ()).throw(MemoryError("simulated CUDA failure")),
    )

    with pytest.raises(HTTPException) as raised:
        service.entry(
            "detector/20260101_session",
            master.name,
            det_bin=1,
            scan_bin=1,
            scan_region=None,
        )

    assert raised.value.status_code == 413
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_catalog_and_acquisition_status_use_quantem_gpu_inspection(tmp_path):
    master = _master(tmp_path)
    service = _service(tmp_path)
    client = TestClient(create_app(tmp_path, service=service))

    catalog = client.get("/api/browse/sessions").json()
    status = client.get("/api/browse/acquisitions").json()

    assert catalog["complete"] is True
    assert catalog["sessions"][0]["source"] == "detector"
    assert catalog["sessions"][0]["date"] == "20260101_session"
    assert catalog["sessions"][0]["path"] == "detector/20260101_session"
    assert catalog["sessions"][0]["files"][0]["name"] == master.name
    assert catalog["sessions"][0]["files"][0]["shape"] == [2, 2, 4, 4]
    assert catalog["sessions"][0]["files"][0]["loadable"] is True
    assert status["pending"] == []
    assert status["history"][0]["path"] == str(master)
    assert status["ready_token"]


def test_acquisition_poll_reuses_ready_inspection_until_master_changes(
    tmp_path,
    monkeypatch,
):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    original_inspect = service._inspect
    calls = []

    def recording_inspect(path):
        calls.append(path)
        return original_inspect(path)

    monkeypatch.setattr(service, "_inspect", recording_inspect)

    first = service.acquisitions()
    second = service.acquisitions()

    assert first["ready_token"] == second["ready_token"]
    assert calls == []

    master.touch()
    changed = service.acquisitions()

    assert calls == [master]
    assert changed["history"][0]["path"] == str(master)


def test_acquisition_poll_reinspects_pending_master_when_shard_arrives(
    tmp_path,
    monkeypatch,
):
    h5py = pytest.importorskip("h5py")
    master, shard = _externally_linked_master(tmp_path)
    shard.unlink()
    service = _service(tmp_path)
    service.refresh_catalog()
    assert service.acquisitions()["pending"][0]["path"] == str(master)
    original_inspect = service._inspect
    calls = []

    def recording_inspect(path):
        calls.append(path)
        return original_inspect(path)

    monkeypatch.setattr(service, "_inspect", recording_inspect)
    unchanged = service.acquisitions()
    assert unchanged["pending"][0]["path"] == str(master)
    assert calls == []

    with h5py.File(shard, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4),
        )
    completed = service.acquisitions()

    assert calls == [master]
    assert completed["pending"] == []
    assert completed["history"][0]["path"] == str(master)


def test_concurrent_catalog_refreshes_are_coalesced(tmp_path, monkeypatch):
    import concurrent.futures
    import threading

    _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    original_refresh = service._refresh_catalog
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def recording_refresh():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return original_refresh()

    monkeypatch.setattr(service, "_refresh_catalog", recording_refresh)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.refresh_catalog)
        assert entered.wait(timeout=5)
        second = executor.submit(service.refresh_catalog)
        release.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert first_result == second_result
    assert calls == 1


def test_acquisition_discovery_refreshes_master_paths_in_background(
    tmp_path,
    monkeypatch,
):
    import threading

    first = _master(tmp_path, "first_master.h5")
    service = _service(tmp_path)
    service.refresh_catalog()
    second = _master(tmp_path, "second_master.h5")
    scan_finished = threading.Event()
    original_finish = service._finish_background_master_scan

    def recording_finish():
        original_finish()
        scan_finished.set()

    monkeypatch.setattr(service, "_finish_background_master_scan", recording_finish)
    service._last_master_scan_completed = 0

    immediate = service._watched_master_paths()

    assert immediate == [first]
    assert scan_finished.wait(timeout=5)
    assert service._watched_master_paths() == [first, second]


def test_failed_background_discovery_waits_before_retry(tmp_path, monkeypatch):
    _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    service._last_master_scan_completed = 0
    service._master_scan_in_flight = True

    def fail_scan():
        raise OSError("temporary folder failure")

    monkeypatch.setattr(service, "_scan_master_paths", fail_scan)
    service._finish_background_master_scan()

    assert service._master_scan_in_flight is False
    assert service._last_master_scan_completed > 0
    assert service._watched_master_paths()
    assert service._master_scan_in_flight is False


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
        "session": "detector/20260101_session",
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


def test_selected_diffraction_ensure_resident_retries_one_conflict(
    tmp_path,
    monkeypatch,
):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.refresh_catalog()
    monkeypatch.setattr(service, "_free_bytes", lambda _gpu: None)
    data = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    loads = 0

    def fake_load(_path, _det_bin, _scan_bin, _scan_region, gpu):
        nonlocal loads
        loads += 1
        compute = detector.prepare(data.copy())
        return {
            "gpu": gpu,
            "data": data.copy(),
            "compute": compute,
            "scan_shape": (2, 2),
            "detector_shape": (4, 4),
            "mean_dp": compute.mean_dp(),
            "bf_geometry": None,
            "com_row": None,
            "com_column": None,
        }

    monkeypatch.setattr(service, "_load_entry", fake_load)
    original_selected = service.selected_diffraction
    attempts = 0

    def evict_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPException(409, "simulated concurrent eviction")
        return original_selected(*args, **kwargs)

    monkeypatch.setattr(service, "selected_diffraction", evict_once)
    try:
        response = TestClient(create_app(tmp_path, service=service)).get(
            "/api/browse/cbed",
            params={
                "session": "detector/20260101_session",
                "file": master.name,
                "sx": 1,
                "sy": 0,
                "ensure_resident": True,
            },
        )
    finally:
        service._load_executor.shutdown(wait=True, cancel_futures=True)

    assert response.status_code == 200
    assert attempts == 2
    assert loads == 1
    np.testing.assert_array_equal(
        np.frombuffer(response.content, dtype="<u4").reshape(4, 4),
        data[1, 0],
    )


@pytest.mark.parametrize(
    ("shape", "inner", "expected"),
    [("circle", 1.0, 12), ("square", 1.0, 16), ("annulus", 0.0, 12)],
)
def test_custom_detector_shapes_return_exact_counts(tmp_path, shape, inner, expected):
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
            "session": "detector/20260101_session",
            "file": master.name,
            "shape": shape,
            "cx": 1.5,
            "cy": 1.5,
            "inner": inner,
            "outer": 1.6,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-dtype"] == "<u4"
    values = np.frombuffer(response.content, dtype="<u4")
    assert values.tolist() == [expected] * 4


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("cx", "nan"), ("cy", "inf"), ("inner", "nan"), ("outer", "inf")],
)
def test_custom_detector_rejects_nonfinite_geometry(tmp_path, parameter, value):
    client = TestClient(create_app(tmp_path, service=_service(tmp_path)))
    params = {
        "session": "detector/20260101_session",
        "file": "sample_master.h5",
        "shape": "annulus",
        "cx": 1,
        "cy": 1,
        "inner": 1,
        "outer": 2,
        parameter: value,
    }

    response = client.get("/api/browse/realspace-shape", params=params)

    assert response.status_code == 400
    assert "must be finite" in response.text


def test_scan_roi_indices_match_circle_and_square_geometry() -> None:
    circle = _scan_roi_indices(
        (5, 6),
        shape="circle",
        center_row=2,
        center_column=3,
        radius=1.1,
    )
    square = _scan_roi_indices(
        (5, 6),
        shape="square",
        center_row=2,
        center_column=3,
        radius=1,
    )

    assert circle.tolist() == [9, 14, 15, 16, 21]
    assert square.tolist() == [8, 9, 10, 14, 15, 16, 20, 21, 22]


@pytest.mark.parametrize("shape", ["circle", "square"])
@pytest.mark.parametrize("reduce", ["mean", "sum", "max"])
def test_scan_roi_diffraction_returns_exact_sum_and_divisor(
    tmp_path,
    shape,
    reduce,
):
    master = _master(tmp_path)
    service = _service(tmp_path)
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

    response = client.get(
        "/api/browse/cbed-region",
        params={
            "session": "detector/20260101_session",
            "file": master.name,
            "shape": shape,
            "cx": 0.5,
            "cy": 0.5,
            "radius": 1,
            "reduce": reduce,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-dtype"] == "<u4"
    expected = (
        data.max(axis=(0, 1)).astype(np.uint32)
        if reduce == "max"
        else data.sum(axis=(0, 1), dtype=np.uint64).astype(np.uint32)
    )
    np.testing.assert_array_equal(
        np.frombuffer(response.content, dtype="<u4").reshape(4, 4),
        expected,
    )
    assert response.headers["x-value-divisor"] == ("4" if reduce == "mean" else "1")


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
            "session": "detector/20260101_session",
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
        main(
            [
                "serve",
                "/data",
                "--gpu",
                "-1",
                "--implementation-revision",
                "test",
            ]
        )


def test_cli_rejects_invalid_gpu_pool_before_launching():
    with pytest.raises(SystemExit, match="comma-separated CUDA indices"):
        main(
            [
                "serve",
                "/data",
                "--gpus",
                "0,nope",
                "--implementation-revision",
                "test",
            ]
        )


@pytest.mark.parametrize("command", ["serve", "serve-ssb-mps"])
def test_cli_requires_exact_implementation_revision(command):
    with pytest.raises(SystemExit):
        main([command, "/data"])
