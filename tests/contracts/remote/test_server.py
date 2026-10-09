import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from quantem.gpu import __version__, detector
from quantem.gpu.cli import _parser, main
from quantem.gpu.remote import BrowsePlan, BrowseService, ResidentAcquisition, create_app
from quantem.gpu.remote.app import _wire_image
from quantem.gpu.remote.browse import PROTOCOL_NAME, PROTOCOL_VERSION
from quantem.gpu.remote.catalog import Catalog, file_signature

requires_cuda = pytest.mark.skipif(
    os.environ.get("QEM_TEST_BACKEND") != "cuda",
    reason="Set QEM_TEST_BACKEND=cuda: the service holds acquisitions on a physical CUDA GPU.",
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


def _pool(service: BrowseService, *budgets: int) -> BrowseService:
    """Give the service one pool slot per cache budget, all on the visible CUDA device.

    Admission works on slot indices and byte budgets, so slots that share the
    one physical device exercise multi-GPU placement with small, exact numbers.
    """
    gpu = service.residency.gpus[0]
    service.residency.gpus = {
        index: replace(gpu, index=index, cache_budget_bytes=budget)
        for index, budget in enumerate(budgets)
    }
    return service


def _service(root: Path, *budgets: int) -> BrowseService:
    return _pool(BrowseService(root), *(budgets or (1 << 30,)))


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


class _CountingDevice:
    """Wrap the real CUDA device and count how often a calculation selects it."""

    def __init__(self, device) -> None:
        self.device = device
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self.device.__enter__()

    def __exit__(self, *args):
        return self.device.__exit__(*args)


class _Closable:
    """Stands in for the loaded acquisition or detector session in admission tests."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _resident(
    path: Path | str,
    gpu: int,
    resident_bytes: int,
    *,
    session: object | None = None,
) -> ResidentAcquisition:
    """Build a resident entry whose recorded files match ``path`` on disk."""
    return ResidentAcquisition(
        key=str(path),
        gpu=gpu,
        loaded=_Closable(),
        session=_Closable() if session is None else session,
        resident_bytes=resident_bytes,
        source_signature=file_signature(Path(path)),
    )


def _array_resident(service: BrowseService, master: Path, data: np.ndarray) -> ResidentAcquisition:
    """Make ``data`` resident for ``master`` through a real detector session over host counts."""
    entry = _resident(master, 0, data.nbytes, session=detector.prepare(data))
    service.residency._entries[entry.key] = entry
    return entry


def _plan(master_shape=(2, 2), detector_shape=(4, 4)) -> BrowsePlan:
    return BrowsePlan(master_shape, detector_shape)


@requires_cuda
def test_capabilities_identify_quantem_gpu_protocol(tmp_path):
    revision = "d" * 40
    client = TestClient(create_app(tmp_path, implementation_revision=revision))

    payload = client.get("/api/browse/capabilities").json()

    assert payload["protocol"] == PROTOCOL_NAME
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["backend"] == "cuda"
    assert payload["browse_gpu"] == 0
    assert payload["browse_gpus"] == [0]
    assert payload["cache_fraction"] == pytest.approx(0.80)
    assert payload["data_folders"] == [str(tmp_path)]
    assert payload["features"]["exact_integer_images"] is True
    assert payload["features"]["multi_gpu_residency"] is False
    assert payload["implementation_revision"] == revision


@requires_cuda
@requires_cuda
def test_capabilities_report_live_admission_bytes(tmp_path, monkeypatch):
    service = _service(tmp_path, 100)
    active_key = "active"
    service.residency._entries[active_key] = _resident(active_key, 0, 30)
    service.residency._entries["evictable"] = _resident("evictable", 0, 20)
    service.residency.activate(active_key)
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: 10)

    device = service.capabilities()["devices"][0]

    assert device["resident_bytes"] == 50
    assert device["active_resident_bytes"] == 30
    assert device["evictable_bytes"] == 20
    assert device["available_peak_bytes"] == 30
    assert device["available_resident_bytes"] == 70


@requires_cuda
@requires_cuda
def test_capabilities_use_budget_when_free_memory_is_unknown(tmp_path, monkeypatch):
    service = _service(tmp_path, 100)
    active_key = "active"
    service.residency._entries[active_key] = _resident(active_key, 0, 30)
    service.residency.activate(active_key)
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: None)

    device = service.capabilities()["devices"][0]

    assert device["available_peak_bytes"] == 100
    assert device["available_resident_bytes"] == 70


@requires_cuda
@requires_cuda
def test_advertised_capacity_matches_admission_boundaries(tmp_path, monkeypatch):
    service = _service(tmp_path, 100)
    active_key = "active"
    service.residency._entries[active_key] = _resident(active_key, 0, 30)
    service.residency._entries["evictable"] = _resident("evictable", 0, 20)
    service.residency.activate(active_key)
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: 10)
    device = service.capabilities()["devices"][0]

    assert service.residency._candidate_gpus(
        device["available_resident_bytes"], device["available_peak_bytes"]
    ) == [0]
    assert service.residency._candidate_gpus(
        device["available_resident_bytes"] + 1, device["available_peak_bytes"]
    ) == []
    assert service.residency._candidate_gpus(
        device["available_resident_bytes"], device["available_peak_bytes"] + 1
    ) == []


@requires_cuda
@requires_cuda
def test_active_dataset_on_one_gpu_preserves_capacity_on_another(tmp_path, monkeypatch):
    service = _service(tmp_path, 100, 100)
    active_key = "active"
    service.residency._entries[active_key] = _resident(active_key, 0, 60)
    service.residency.activate(active_key)
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: 100)

    devices = service.capabilities()["devices"]

    assert devices[0]["available_resident_bytes"] == 40
    assert devices[1]["available_resident_bytes"] == 100
    assert service.residency._candidate_gpus(resident_bytes=50, peak_bytes=70) == [1]


@requires_cuda
@requires_cuda
def test_multi_gpu_cache_places_whole_datasets_and_reuses_hits(tmp_path, monkeypatch):
    first = _master(tmp_path, "sample_00_master.h5")
    second = _master(tmp_path, "sample_01_master.h5")
    service = _service(tmp_path, 100, 100)
    service.catalog.refresh()
    loads: list[tuple[str, int]] = []

    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (40, 60))

    def fake_load(path, gpu):
        loads.append((path.name, gpu))
        return _resident(path, gpu, 40)

    monkeypatch.setattr(service.residency, "_load", fake_load)

    first_entry = service.residency.entry(first)
    second_entry = service.residency.entry(second)
    hit_entry = service.residency.entry(first)

    assert first_entry.gpu == 0
    assert second_entry.gpu == 1
    assert hit_entry is first_entry
    assert first_entry.key != second_entry.key
    assert loads == [(first.name, 0), (second.name, 1)]
    devices = service.capabilities()["devices"]
    assert [(device["index"], device["resident_entries"]) for device in devices] == [
        (0, 1),
        (1, 1),
    ]


@requires_cuda
@requires_cuda
def test_serialized_cuda_loads_reuse_one_host_worker(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(2)]
    service = _service(tmp_path)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: None)
    load_threads: list[tuple[int, str]] = []

    def fake_load(path, gpu):
        load_threads.append((threading.get_ident(), threading.current_thread().name))
        return _resident(path, gpu, 40)

    monkeypatch.setattr(service.residency, "_load", fake_load)
    try:
        for path in files:
            service.residency.entry(path)
    finally:
        service.residency._load_executor.shutdown(wait=True, cancel_futures=True)

    assert len({thread_id for thread_id, _ in load_threads}) == 1
    assert all(name.startswith("quantem-cuda-load") for _, name in load_threads)


@requires_cuda
@requires_cuda
def test_app_shutdown_closes_service_resources(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    service = app.state.browse_service
    resident = _resident("fixture", 0, 10)
    service.residency._entries[resident.key] = resident
    original_close = service.close
    closed = False

    def close() -> None:
        nonlocal closed
        original_close()
        closed = True

    monkeypatch.setattr(service, "close", close)

    with TestClient(app) as client:
        assert not closed
        assert client.get("/api/browse/capabilities").status_code == 200
        loader_thread = service.residency._load_executor.submit(threading.current_thread).result()
        assert loader_thread.is_alive()

    assert closed
    assert not loader_thread.is_alive()
    assert resident.session.closed
    assert resident.loaded.closed
    assert not service.residency._entries


def test_full_scan_region_is_the_uncropped_plan(tmp_path):
    master = _master(tmp_path)
    service = BrowseService(tmp_path)

    path, plan = service.plan(
        "detector/20260101_session",
        master.name,
        det_bin=1,
        scan_bin=1,
        scan_region=(0, 2, 0, 2),
    )

    assert path == master
    assert plan == BrowsePlan((2, 2), (4, 4))
    assert plan.scan_region is None


@requires_cuda
@requires_cuda
def test_multi_gpu_lru_evicts_only_the_selected_device(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(3)]
    service = _service(tmp_path, 100, 100)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(service.residency, "_load", lambda path, gpu: _resident(path, gpu, 80))

    entries = [service.residency.entry(path) for path in files]

    assert [entry.gpu for entry in entries] == [0, 1, 0]
    assert entries[0].key not in service.residency._entries
    assert entries[1].key in service.residency._entries
    assert entries[2].key in service.residency._entries


@requires_cuda
@requires_cuda
def test_multi_gpu_cache_preserves_the_active_dataset(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(3)]
    service = _service(tmp_path, 100, 100)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(service.residency, "_load", lambda path, gpu: _resident(path, gpu, 80))
    first = service.residency.entry(files[0])
    service.residency._active_key = first.key
    second = service.residency.entry(files[1])
    third = service.residency.entry(files[2])

    assert first.key in service.residency._entries
    assert second.key not in service.residency._entries
    assert third.key in service.residency._entries
    assert service.residency._entries[first.key].gpu == 0
    assert service.residency._entries[third.key].gpu == 1


@requires_cuda
def test_cache_evicts_active_dataset_when_it_is_the_only_valid_transition(
    tmp_path,
    monkeypatch,
):
    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path, 100)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(service.residency, "_load", lambda path, gpu: _resident(path, gpu, 80))

    first_entry = service.residency.entry(first)
    service.residency._active_key = first_entry.key
    second_entry = service.residency.entry(second)

    assert first_entry.key not in service.residency._entries
    assert second_entry.key in service.residency._entries
    assert service.residency._active_key is None


@requires_cuda
def test_reserved_entry_stays_resident_until_request_finishes(tmp_path, monkeypatch):
    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path, 100)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: None)
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (80, 90))
    monkeypatch.setattr(service.residency, "_load", lambda path, gpu: _resident(path, gpu, 80))
    first_entry = service.residency.entry(first, reserve=True)
    second_complete = threading.Event()

    def load_second():
        try:
            return service.residency.entry(second)
        finally:
            second_complete.set()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(load_second)
            try:
                assert second_complete.wait(0.1) is False
                assert first_entry.key in service.residency._entries
            finally:
                service.residency.release(first_entry)
            second_entry = future.result(timeout=2)
    finally:
        service.residency._load_executor.shutdown(wait=True, cancel_futures=True)

    assert first_entry.key not in service.residency._entries
    assert second_entry.key in service.residency._entries


@requires_cuda
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
    service = _service(tmp_path, 100)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (80, 90))
    first_compute = Compute()

    def fake_load(path, gpu):
        compute = first_compute if path == first else Compute()
        return _resident(path, gpu, 80, session=compute)

    monkeypatch.setattr(service.residency, "_load", fake_load)
    geometry_started = threading.Event()
    release_geometry = threading.Event()

    def blocking_geometry(_entry, _plan):
        geometry_started.set()
        assert release_geometry.wait(2)
        return (0.5, 0.5, 1.0)

    monkeypatch.setattr(service, "_bf_geometry", blocking_geometry)
    first_entry = service.residency.entry(first)
    second_complete = threading.Event()

    def load_second():
        try:
            return service.residency.entry(second)
        finally:
            second_complete.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        image_future = pool.submit(
            service.virtual_image,
            first_entry,
            _plan(detector_shape=(2, 2)),
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
    assert first_entry.key not in service.residency._entries


@requires_cuda
def test_busy_entry_does_not_block_another_resident_dataset(tmp_path, monkeypatch):
    class Compute:
        def masked_sum_exact(self, _mask):
            return np.arange(4, dtype=np.uint64)

    service = _service(tmp_path)
    first_entry = _resident("first", 0, 4, session=Compute())
    second_entry = _resident("second", 0, 4, session=Compute())
    service.residency._entries[first_entry.key] = first_entry
    service.residency._entries[second_entry.key] = second_entry
    geometry_started = threading.Event()
    release_geometry = threading.Event()

    def blocking_geometry(entry, _plan):
        if entry is first_entry:
            geometry_started.set()
            assert release_geometry.wait(2)
        return (0.5, 0.5, 1.0)

    monkeypatch.setattr(service, "_bf_geometry", blocking_geometry)
    original_entry_lock = service.residency._entry_lock
    first_lock_calls = 0
    second_reader_started = threading.Event()

    def recording_entry_lock(key):
        nonlocal first_lock_calls
        if key == first_entry.key:
            first_lock_calls += 1
            if first_lock_calls == 2:
                second_reader_started.set()
        return original_entry_lock(key)

    monkeypatch.setattr(service.residency, "_entry_lock", recording_entry_lock)

    def image(entry):
        return service.virtual_image(
            entry,
            _plan(detector_shape=(2, 2)),
            mode="BF",
            inner=0.0,
            outer=1.0,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(image, first_entry)
        assert geometry_started.wait(2)
        queued = pool.submit(image, first_entry)
        assert second_reader_started.wait(2)

        independent = pool.submit(image, second_entry)
        np.testing.assert_array_equal(
            independent.result(timeout=0.5),
            np.arange(4, dtype=np.uint64).reshape(2, 2),
        )
        release_geometry.set()
        first.result(timeout=2)
        queued.result(timeout=2)


@requires_cuda
def test_free_memory_headroom_evicts_inactive_entry_before_load(
    tmp_path,
    monkeypatch,
):
    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path, 1_000)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (40, 60))
    free_bytes = [70]
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: free_bytes[0])
    monkeypatch.setattr(service.residency, "_load", lambda path, gpu: _resident(path, gpu, 40))

    first_entry = service.residency.entry(first)
    free_bytes[0] = 25
    second_entry = service.residency.entry(second)

    assert first_entry.key not in service.residency._entries
    assert second_entry.key in service.residency._entries


@requires_cuda
def test_out_of_memory_evicts_another_entry_and_retries_same_gpu(
    tmp_path,
    monkeypatch,
):
    import weakref

    class LoaderTemporary:
        pass

    first = _master(tmp_path, "first_master.h5")
    second = _master(tmp_path, "second_master.h5")
    service = _service(tmp_path, 1_000)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (40, 60))
    calls: list[str] = []
    temporary_ref: weakref.ReferenceType[LoaderTemporary] | None = None

    def fake_load(path, gpu):
        nonlocal temporary_ref
        calls.append(path.name)
        if path == second and calls.count(second.name) == 1:
            temporary = LoaderTemporary()
            temporary_ref = weakref.ref(temporary)
            raise MemoryError("simulated CUDA allocation failure")
        if path == second:
            assert temporary_ref is not None
            assert temporary_ref() is None
        return _resident(path, gpu, 40)

    monkeypatch.setattr(service.residency, "_load", fake_load)

    first_entry = service.residency.entry(first)
    second_entry = service.residency.entry(second)

    assert calls == [first.name, second.name, second.name]
    assert first_entry.key not in service.residency._entries
    assert second_entry.key in service.residency._entries


@requires_cuda
def test_out_of_memory_retry_preserves_active_entry(tmp_path, monkeypatch):
    files = [_master(tmp_path, f"sample_{index:02d}_master.h5") for index in range(3)]
    service = _service(tmp_path, 1_000)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (40, 60))
    target_calls = 0

    def fake_load(path, gpu):
        nonlocal target_calls
        if path == files[2]:
            target_calls += 1
            if target_calls == 1:
                raise MemoryError("simulated CUDA allocation failure")
        return _resident(path, gpu, 40)

    monkeypatch.setattr(service.residency, "_load", fake_load)
    active_entry = service.residency.entry(files[0])
    inactive_entry = service.residency.entry(files[1])
    service.residency._active_key = active_entry.key

    target_entry = service.residency.entry(files[2])

    assert target_calls == 2
    assert active_entry.key in service.residency._entries
    assert inactive_entry.key not in service.residency._entries
    assert target_entry.key in service.residency._entries


@requires_cuda
def test_out_of_memory_error_does_not_retain_the_loader_exception(
    tmp_path,
    monkeypatch,
):
    master = _master(tmp_path)
    service = _service(tmp_path)
    service.catalog.refresh()
    monkeypatch.setattr(service.residency, "_expected_bytes", lambda *_args: (40, 60))
    monkeypatch.setattr(
        service.residency,
        "_load",
        lambda *_args: (_ for _ in ()).throw(MemoryError("simulated CUDA failure")),
    )

    with pytest.raises(HTTPException) as raised:
        service.residency.entry(master)

    assert raised.value.status_code == 413
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_catalog_and_acquisition_status_use_quantem_gpu_inspection(tmp_path):
    master = _master(tmp_path)
    client = TestClient(create_app(tmp_path))

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
    catalog = Catalog(tmp_path.resolve())
    catalog.refresh()
    original_inspect = catalog.inspect
    calls = []

    def recording_inspect(path):
        calls.append(path)
        return original_inspect(path)

    monkeypatch.setattr(catalog, "inspect", recording_inspect)

    first = catalog.acquisitions()
    second = catalog.acquisitions()

    assert first["ready_token"] == second["ready_token"]
    assert calls == []

    master.touch()
    changed = catalog.acquisitions()

    assert calls == [master]
    assert changed["history"][0]["path"] == str(master)


def test_acquisition_poll_reinspects_pending_master_when_shard_arrives(
    tmp_path,
    monkeypatch,
):
    h5py = pytest.importorskip("h5py")
    master, shard = _externally_linked_master(tmp_path)
    shard.unlink()
    catalog = Catalog(tmp_path.resolve())
    catalog.refresh()
    assert catalog.acquisitions()["pending"][0]["path"] == str(master)
    original_inspect = catalog.inspect
    calls = []

    def recording_inspect(path):
        calls.append(path)
        return original_inspect(path)

    monkeypatch.setattr(catalog, "inspect", recording_inspect)
    unchanged = catalog.acquisitions()
    assert unchanged["pending"][0]["path"] == str(master)
    assert calls == []

    with h5py.File(shard, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4),
        )
    completed = catalog.acquisitions()

    assert calls == [master]
    assert completed["pending"] == []
    assert completed["history"][0]["path"] == str(master)


def test_concurrent_catalog_refreshes_are_coalesced(tmp_path, monkeypatch):
    import concurrent.futures
    import threading

    _master(tmp_path)
    catalog = Catalog(tmp_path.resolve())
    catalog.refresh()
    original_refresh = catalog._rebuild
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def recording_refresh():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return original_refresh()

    monkeypatch.setattr(catalog, "_rebuild", recording_refresh)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(catalog.refresh)
        assert entered.wait(timeout=5)
        second = executor.submit(catalog.refresh)
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
    catalog = Catalog(tmp_path.resolve())
    catalog.refresh()
    second = _master(tmp_path, "second_master.h5")
    scan_finished = threading.Event()
    original_finish = catalog._finish_background_scan

    def recording_finish():
        original_finish()
        scan_finished.set()

    monkeypatch.setattr(catalog, "_finish_background_scan", recording_finish)
    catalog._last_scan_completed = 0

    immediate = catalog._watched_masters()

    assert immediate == [first]
    assert scan_finished.wait(timeout=5)
    assert catalog._watched_masters() == [first, second]


def test_failed_background_discovery_waits_before_retry(tmp_path, monkeypatch):
    _master(tmp_path)
    catalog = Catalog(tmp_path.resolve())
    catalog.refresh()
    catalog._last_scan_completed = 0
    catalog._scan_in_flight = True

    def fail_scan():
        raise OSError("temporary folder failure")

    monkeypatch.setattr(catalog, "_scan_masters", fail_scan)
    catalog._finish_background_scan()

    assert catalog._scan_in_flight is False
    assert catalog._last_scan_completed > 0
    assert catalog._watched_masters()
    assert catalog._scan_in_flight is False


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
    catalog = Catalog(tmp_path.resolve())

    listing = catalog.refresh()

    assert [item["path"] for item in listing["sessions"]] == [
        "collaborator-a/project/shared-session",
        "collaborator-b/project/shared-session",
    ]
    assert catalog.resolve_master(listing["sessions"][0]["path"], first.name) == first
    assert catalog.resolve_master(listing["sessions"][1]["path"], second.name) == second
    with pytest.raises(HTTPException) as error:
        catalog.resolve_master("project/shared-session", first.name)
    assert error.value.status_code == 404


def test_catalog_follows_nonstandard_external_shard_names(tmp_path):
    master, shard = _externally_linked_master(tmp_path)

    item = Catalog(tmp_path.resolve()).refresh()["sessions"][0]["files"][0]

    assert item["loadable"] is True
    assert item["size_bytes"] == master.stat().st_size + shard.stat().st_size


@requires_cuda
def test_exact_virtual_image_and_selected_diffraction_share_resident_plan(tmp_path):
    master = _master(tmp_path)
    app = create_app(tmp_path)
    service = app.state.browse_service
    device = _CountingDevice(service.residency.gpus[0].device)
    service.residency.gpus[0] = replace(service.residency.gpus[0], device=device)
    data = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    _array_resident(service, master, data)
    client = TestClient(app)
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


@requires_cuda
def test_selected_diffraction_ensure_resident_retries_one_conflict(
    tmp_path,
    monkeypatch,
):
    master = _master(tmp_path)
    app = create_app(tmp_path)
    service = app.state.browse_service
    monkeypatch.setattr(service.residency, "_free_bytes", lambda _gpu: None)
    data = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    loads = 0

    def fake_load(path, gpu):
        nonlocal loads
        loads += 1
        return _resident(path, gpu, data.nbytes, session=detector.prepare(data.copy()))

    monkeypatch.setattr(service.residency, "_load", fake_load)
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
        response = TestClient(app).get(
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
        service.residency._load_executor.shutdown(wait=True, cancel_futures=True)

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
@requires_cuda
def test_custom_detector_shapes_return_exact_counts(tmp_path, shape, inner, expected):
    master = _master(tmp_path)
    app = create_app(tmp_path)
    _array_resident(app.state.browse_service, master, np.ones((2, 2, 4, 4), dtype=np.uint16))
    client = TestClient(app)

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
    client = TestClient(create_app(tmp_path))
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


@requires_cuda
def test_oversized_acquisition_is_rejected_before_loading(tmp_path, monkeypatch):
    master = _master(tmp_path)
    app = create_app(tmp_path)
    service = _pool(app.state.browse_service, 1)
    monkeypatch.setattr(
        service.residency,
        "_load",
        lambda *_args, **_kwargs: pytest.fail("oversized acquisition reached CUDA loading"),
    )
    client = TestClient(app)

    response = client.get(
        "/api/browse/realspace",
        params={
            "session": "detector/20260101_session",
            "file": master.name,
            "mode": "BF",
        },
    )

    assert response.status_code == 413
    assert "Serve it from a GPU with more memory" in response.text


def test_uint32_admission_reserves_the_encoded_uint16_bound(tmp_path):
    master = _master(tmp_path, dtype=np.uint32)
    service = BrowseService(tmp_path)

    resident_bytes, peak_bytes = service.residency._expected_bytes(master)

    assert resident_bytes == 2 * 2 * 4 * 4 * np.dtype(np.uint16).itemsize
    assert peak_bytes == resident_bytes + (128 << 20) + 4 * 4 * 4 * 64
    assert service.catalog.sessions()["sessions"][0]["files"][0]["dtype"] == "uint32"


def test_plan_rejects_bins_and_crops_the_acquisition_cannot_serve(tmp_path):
    master = _master(tmp_path)
    service = BrowseService(tmp_path)

    def status(**request):
        with pytest.raises(HTTPException) as raised:
            service.plan("detector/20260101_session", master.name, **request)
        return raised.value.status_code

    assert status(det_bin=3, scan_bin=1, scan_region=None) == 400
    assert status(det_bin=1, scan_bin=3, scan_region=None) == 400
    assert status(det_bin=1, scan_bin=1, scan_region=(0, 3, 0, 2)) == 400


def test_plan_bins_and_crops_the_view_with_partial_edge_bins():
    plan = BrowsePlan((7, 6), (16, 16), detector_bin=4, scan_bin=2, scan_region=(1, 6, 0, 5))
    values = np.arange(7 * 6, dtype=np.uint64).reshape(7, 6)
    pattern = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)

    assert plan.scan_shape == (3, 3)
    assert plan.detector_shape == (4, 4)
    np.testing.assert_array_equal(
        plan.scan_image(values),
        [
            [values[1:3, 0:2].sum(), values[1:3, 2:4].sum(), values[1:3, 4].sum()],
            [values[3:5, 0:2].sum(), values[3:5, 2:4].sum(), values[3:5, 4].sum()],
            [values[5, 0:2].sum(), values[5, 2:4].sum(), values[5, 4]],
        ],
    )
    assert plan.positions(2, 2).tolist() == [5 * 6 + 4]
    assert plan.positions(0, 1).tolist() == [1 * 6 + 2, 1 * 6 + 3, 2 * 6 + 2, 2 * 6 + 3]
    np.testing.assert_array_equal(
        plan.diffraction(pattern),
        pattern.reshape(4, 4, 4, 4).sum(axis=(1, 3), dtype=np.uint64),
    )
    assert plan.source_mask(np.eye(4, dtype=bool)).sum() == 4 * 16


def test_wire_image_preserves_counts_above_float32_precision():
    source = np.asarray([[16_777_217, 40_000_001]], dtype=np.uint64)

    payload, dtype = _wire_image(source)

    assert dtype == "<u4"
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<u4"), source.ravel())


def test_server_source_has_no_quantem_live_dependency():
    for source in (Path(__file__).parents[3] / "src/quantem/gpu/remote").glob("*.py"):
        assert "quantem.live" not in source.read_text(), source.name


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


def test_cli_serve_help_says_the_default_port_may_be_taken(capsys):
    with pytest.raises(SystemExit):
        _parser().parse_args(["serve", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "default: 8780; another local service may already use it, then pass --port" in help_text


def test_cli_serves_without_an_explicit_implementation_revision():
    arguments = _parser().parse_args(["serve", "/data", "--gpus", "auto", "--port", "8780"])

    assert arguments.implementation_revision == __version__
    assert _parser().parse_args(
        ["serve", "/data", "--implementation-revision", "abc123"]
    ).implementation_revision == "abc123"
