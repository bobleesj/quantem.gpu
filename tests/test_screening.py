from __future__ import annotations

import numpy as np


def _run_fake_mps_screening(monkeypatch, tmp_path, chunks):
    from types import SimpleNamespace

    from quantem.gpu import detector, io
    from quantem.gpu.dpc import workflow as dpc_workflow
    from quantem.gpu.screening import workflow

    class FakeFrames:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.uint16)
            self.vi = self
            self.n = int(self.values.shape[0])
            self.nbytes = int(self.values.nbytes)

        def detector_sum_exact(self):
            return self.values.sum(axis=0, dtype=np.uint64)

        def detector_sum(self):
            return self.detector_sum_exact().astype(np.float32)

        def masked_sum(self, mask):
            return self.values[:, np.asarray(mask, dtype=bool)].sum(
                axis=1,
                dtype=np.uint64,
            )

        def center_of_mass(self, _mask):
            zeros = np.zeros(self.n, dtype=np.float32)
            return zeros.copy(), zeros.copy()

    loads = []

    def fake_load(_source, *, scan_region=None, **_kwargs):
        row = 0 if scan_region is None else int(scan_region[0])
        loads.append(row)
        return SimpleNamespace(data=FakeFrames(chunks[row]))

    def fake_auto_probe(mean_dp):
        center = np.unravel_index(int(np.argmax(mean_dp)), mean_dp.shape)
        return tuple(float(value) for value in center), 0.25

    def fake_detector_mask(center, inner, _outer, shape):
        mask = np.zeros(shape, dtype=bool)
        mask[round(center[0]), round(center[1])] = True
        return ~mask if inner > 0 else mask

    monkeypatch.setattr(io, "inspect", lambda _path: SimpleNamespace(
        metadata={"detector_shape": (2, 2), "dtype": "uint16"},
    ))
    monkeypatch.setattr(io, "load", fake_load)
    monkeypatch.setattr(detector, "auto_probe", fake_auto_probe)
    monkeypatch.setattr(detector, "detector_mask", fake_detector_mask)
    monkeypatch.setattr(
        dpc_workflow,
        "find_optimal_rotation",
        lambda *_args, **_kwargs: (None, None, 0.0, False),
    )
    monkeypatch.setattr(workflow, "_mps_chunked_frames_for", lambda data: data)
    monkeypatch.setattr(workflow, "_clear_mps_transients", lambda: None)
    monkeypatch.setattr(
        workflow,
        "_source_fingerprint",
        lambda _master: {"fixture": "stable"},
    )

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    plan = workflow._memory_plan_with_chunk_rows(
        workflow._memory_plan_for_shapes(
            (2, 1),
            (2, 2),
            np.dtype(np.uint16).itemsize,
            1.0,
        ),
        1,
    )
    result = workflow._build_mps_products(
        master,
        scan_shape=(2, 1),
        chunk_rows=1,
        sample_positions=0,
        seed=0,
        rotation_steps=1,
        memory_plan=plan,
        verbose=False,
        skip_mps_memory_check=True,
    )
    return result, loads


def _result(master, workflow):
    metadata = {
        "version": workflow._CACHE_VERSION,
        "source": workflow._source_fingerprint(master),
        "parameters": {
            "center": [3.5, 4.5],
            "radius_px": 2.0,
            "rotation_deg": 17.0,
            "transposed": False,
        },
    }
    zeros = np.zeros((4, 4), dtype=np.float32)
    return workflow.ScreeningResult(
        mean_dp=np.arange(9, dtype=np.float32).reshape(3, 3),
        bright_field=np.ones((4, 4), dtype=np.float32),
        dark_field=np.full((4, 4), 2.0, dtype=np.float32),
        dpc_phase=zeros.copy(),
        com_row=zeros.copy(),
        com_col=zeros.copy(),
        probe_center=(3.5, 4.5),
        probe_radius=2.0,
        rotation_deg=17.0,
        transposed=False,
        metadata=metadata,
    )


def test_screening_cache_roundtrip(monkeypatch, tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    expected = _result(master, workflow)

    workflow._save_cache(expected, cache_path)
    monkeypatch.setattr(
        workflow,
        "_dpc_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a current cache must load the retained phase")
        ),
    )
    actual = workflow._prepare_cache(cache_path, master)

    assert actual is not None
    assert actual.from_cache is True
    assert actual.cache_path == cache_path
    assert actual.probe_center == (3.5, 4.5)
    assert actual.probe_radius == 2.0
    np.testing.assert_array_equal(actual.mean_dp, expected.mean_dp)
    np.testing.assert_array_equal(actual.bright_field, expected.bright_field)
    np.testing.assert_array_equal(actual.dark_field, expected.dark_field)
    np.testing.assert_array_equal(actual.dpc_phase, expected.dpc_phase)
    assert actual.dpc_phase.dtype == np.float32


def test_screening_cache_without_retained_phase_recomputes(tmp_path) -> None:
    """Version-3 caches created by older readers remain compatible."""
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    expected = _result(master, workflow)
    cache_path.parent.mkdir(parents=True)
    np.savez(
        cache_path,
        metadata_json=workflow._metadata_array(expected.metadata),
        mean_dp=expected.mean_dp,
        bright_field=expected.bright_field,
        dark_field=expected.dark_field,
        com_row=expected.com_row,
        com_col=expected.com_col,
    )

    actual = workflow._prepare_cache(cache_path, master)

    assert actual is not None
    assert actual.from_cache is True
    assert actual.dpc_phase.dtype == np.float32
    assert actual.dpc_phase.shape == expected.dpc_phase.shape


def test_screening_cache_path_tracks_exact_cache_version(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")

    path = workflow._cache_path(master, tmp_path / "cache")

    assert workflow._CACHE_VERSION == 3
    assert path.name == "scan_master.screening-v3.npz"


def test_screening_cache_rejects_legacy_first_chunk_version(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    result = _result(master, workflow)
    result.metadata["version"] = 2
    workflow._save_cache(result, cache_path)

    assert workflow._prepare_cache(cache_path, master) is None


def test_screening_cache_rejects_changed_source(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"first")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    workflow._save_cache(_result(master, workflow), cache_path)

    master.write_bytes(b"changed")

    assert workflow._prepare_cache(cache_path, master) is None


def test_screening_cache_rejects_changed_external_shard(tmp_path) -> None:
    import h5py

    from quantem.gpu.screening import workflow

    shard = tmp_path / "scan_data_000001.h5"
    with h5py.File(shard, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.zeros((4, 2, 3), dtype=np.uint16),
        )
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        group = handle.require_group("entry/data")
        group["data_000001"] = h5py.ExternalLink(
            shard.name,
            "/entry/data/data",
        )
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    workflow._save_cache(_result(master, workflow), cache_path)

    with h5py.File(shard, "a") as handle:
        handle.attrs["changed"] = True

    assert workflow._prepare_cache(cache_path, master) is None


def test_strong_cache_identity_avoids_hdf5_reinspection(monkeypatch, tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    stat = master.stat()
    source = {
        "master": str(master.resolve()),
        "files": [
            {
                "path": str(master.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
            }
        ],
        "datasets": [],
        "expectation": {"frames": None, "basis": None},
    }
    monkeypatch.setattr(
        workflow,
        "_source_fingerprint",
        lambda _master: (_ for _ in ()).throw(
            AssertionError("strong cache identity must not re-inspect HDF5")
        ),
    )

    assert workflow._cache_matches(
        {"version": workflow._CACHE_VERSION, "source": source},
        master,
    )


def test_strong_cache_identity_rejects_duplicate_or_changed_files(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"first")
    stat = master.stat()
    record = {
        "path": str(master.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    source = {"master": str(master.resolve()), "files": [record, dict(record)]}
    assert workflow._strong_cached_source_match(source, master) is False

    source["files"] = [record]
    master.write_bytes(b"changed")
    assert workflow._strong_cached_source_match(source, master) is False


def test_strong_cache_identity_accepts_unchanged_master_symlink(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    target = tmp_path / "target_master.h5"
    target.write_bytes(b"stable")
    master = tmp_path / "scan_master.h5"
    master.symlink_to(target.name)
    target_stat = master.stat()
    link_stat = master.lstat()
    source = {
        "master": str(master.absolute()),
        "files": [
            {
                "path": str(master.absolute()),
                "size": int(target_stat.st_size),
                "mtime_ns": int(target_stat.st_mtime_ns),
                "ctime_ns": int(target_stat.st_ctime_ns),
                "device": int(target_stat.st_dev),
                "inode": int(target_stat.st_ino),
                "symlink_target": target.name,
                "symlink_mtime_ns": int(link_stat.st_mtime_ns),
                "symlink_ctime_ns": int(link_stat.st_ctime_ns),
            }
        ],
    }

    assert workflow._strong_cached_source_match(source, master) is True
    assert workflow._strong_cached_source_match(source, target) is True


def test_strong_cache_identity_rejects_repointed_master_symlink(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    first = tmp_path / "first_master.h5"
    first.write_bytes(b"same-size")
    second = tmp_path / "second_master.h5"
    second.write_bytes(b"same-size")
    master = tmp_path / "scan_master.h5"
    master.symlink_to(first.name)
    target_stat = master.stat()
    link_stat = master.lstat()
    source = {
        "master": str(master.absolute()),
        "files": [
            {
                "path": str(master.absolute()),
                "size": int(target_stat.st_size),
                "mtime_ns": int(target_stat.st_mtime_ns),
                "ctime_ns": int(target_stat.st_ctime_ns),
                "device": int(target_stat.st_dev),
                "inode": int(target_stat.st_ino),
                "symlink_target": first.name,
                "symlink_mtime_ns": int(link_stat.st_mtime_ns),
                "symlink_ctime_ns": int(link_stat.st_ctime_ns),
            }
        ],
    }
    master.unlink()
    master.symlink_to(second.name)

    assert workflow._strong_cached_source_match(source, master) is False


def test_reduced_cache_identity_uses_full_inspection_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    source = {
        "master": str(master.resolve()),
        "files": [
            {
                "path": str(master.resolve()),
                "size": int(master.stat().st_size),
                "mtime_ns": int(master.stat().st_mtime_ns),
            }
        ],
        "datasets": [],
        "expectation": {"frames": None, "basis": None},
    }
    monkeypatch.setattr(workflow, "_source_fingerprint", lambda _master: source)

    assert workflow._strong_cached_source_match(source, master) is None
    assert workflow._cache_matches(
        {"version": workflow._CACHE_VERSION, "source": source},
        master,
    )


def test_current_cache_hit_does_not_import_raw_io(monkeypatch, tmp_path) -> None:
    import builtins

    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    stat = master.stat()
    expected = _result(master, workflow)
    expected.metadata["source"] = {
        "master": str(master.resolve()),
        "files": [
            {
                "path": str(master.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
            }
        ],
        "datasets": [],
        "expectation": {"frames": None, "basis": None},
    }
    cache_dir = tmp_path / "cache"
    workflow._save_cache(expected, workflow._cache_path(master, cache_dir))
    real_import = builtins.__import__

    def reject_raw_io_import(name, *args, **kwargs):
        if name == "quantem.gpu.io":
            raise AssertionError("a current cache hit must not import raw I/O")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_raw_io_import)

    actual = workflow.prepare(master, cache=True, cache_dir=cache_dir)

    assert actual.from_cache is True
    np.testing.assert_array_equal(actual.dpc_phase, expected.dpc_phase)


def test_screening_forced_rotation_recomputes_phase(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    result = _result(master, workflow)
    rows, cols = np.indices((4, 4), dtype=np.float32)
    result.com_row = rows
    result.com_col = cols

    rotated = workflow._with_rotation(result, 35.0)

    assert rotated.rotation_deg == 35.0
    assert rotated.transposed is False
    assert rotated.dpc_phase.dtype == np.float32
    assert rotated.dpc_phase.shape == (4, 4)


def test_screening_rejects_packed_uint4_products() -> None:
    from quantem.gpu.screening.workflow import _screening_output_dtype

    try:
        _screening_output_dtype("u4")
    except ValueError as error:
        assert "derived screening products" in str(error)
    else:
        raise AssertionError("packed uint4 must not be accepted for derived products")


def test_screening_load_calls_use_public_api() -> None:
    """Screening must only pass keywords accepted by public ``io.load``."""
    import ast
    import inspect
    import textwrap

    from quantem.gpu import io
    from quantem.gpu.screening import workflow

    tree = ast.parse(textwrap.dedent(inspect.getsource(workflow)))
    accepted = set(inspect.signature(io.load).parameters)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load"
    ]
    unexpected = {
        keyword.arg
        for call in calls
        for keyword in call.keywords
        if keyword.arg is not None and keyword.arg not in accepted
    }

    assert calls
    assert not unexpected, f"unsupported public io.load keywords: {sorted(unexpected)}"


def test_mps_screening_reuses_primary_pass_when_full_scan_mask_matches(
    monkeypatch,
    tmp_path,
) -> None:
    chunks = [
        np.asarray([[[9, 0], [0, 0]]], dtype=np.uint16),
        np.asarray([[[7, 0], [0, 0]]], dtype=np.uint16),
    ]

    result, loads = _run_fake_mps_screening(monkeypatch, tmp_path, chunks)

    assert loads == [0, 1]
    assert result.metadata["parameters"]["probe_source"] == "full_scan_exact"
    assert result.metadata["parameters"]["bootstrap_source"] == "first_chunk"
    assert result.metadata["parameters"]["masks_identical"] is True
    assert result.metadata["parameters"]["pass_count"] == 1
    np.testing.assert_array_equal(
        result.mean_dp,
        np.asarray([[8, 0], [0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.bright_field,
        np.asarray([[9], [7]], dtype=np.float32),
    )


def test_mps_screening_restreams_bf_df_when_full_scan_mask_changes(
    monkeypatch,
    tmp_path,
) -> None:
    chunks = [
        np.asarray([[[9, 0], [0, 0]]], dtype=np.uint16),
        np.asarray([[[0, 0], [0, 30]]], dtype=np.uint16),
    ]

    result, loads = _run_fake_mps_screening(monkeypatch, tmp_path, chunks)

    assert loads == [0, 1, 0, 1]
    assert result.metadata["parameters"]["probe_source"] == "full_scan_exact"
    assert result.metadata["parameters"]["masks_identical"] is False
    assert result.metadata["parameters"]["pass_count"] == 2
    np.testing.assert_array_equal(
        result.mean_dp,
        np.asarray([[4.5, 0], [0, 15]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.bright_field,
        np.asarray([[0], [30]], dtype=np.float32),
    )


def test_cuda_screening_band_bits_preserve_overlapping_mask_membership() -> None:
    """Canonical detector bands may overlap at exact radial boundaries."""

    from quantem.gpu.screening._cuda import _band_bits

    masks = {
        "bright_field": np.asarray([[True, True], [False, False]]),
        "annular_bright_field": np.asarray([[False, True], [False, False]]),
        "annular_dark_field": np.asarray([[False, True], [True, False]]),
        "dark_field": np.asarray([[False, True], [True, True]]),
    }

    np.testing.assert_array_equal(
        _band_bits(masks),
        np.asarray([[1, 15], [12, 8]], dtype=np.uint8),
    )


def test_cuda_screening_com_uses_exact_uint64_moments() -> None:
    """CoM conversion must not first downcast large exact count statistics."""

    from quantem.gpu.screening._cuda import _center_of_mass_from_exact

    total = np.asarray([0, 2**54 + 2], dtype=np.uint64)
    row_moment = np.asarray([7, 3 * (2**54 + 2)], dtype=np.uint64)
    column_moment = np.asarray([9, 5 * (2**54 + 2)], dtype=np.uint64)

    row, column = _center_of_mass_from_exact(
        total,
        row_moment,
        column_moment,
    )

    np.testing.assert_array_equal(row, np.asarray([0.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(
        column,
        np.asarray([0.0, 5.0], dtype=np.float32),
    )


def test_cuda_mask_guard_correction_matches_authoritative_integer_sums() -> None:
    """Retained changed pixels must replace provisional band sums exactly."""

    from quantem.gpu.screening._cuda import _apply_mask_guard_correction

    rng = np.random.default_rng(91)
    counts = rng.integers(0, 65536, size=(7, 12), dtype=np.uint16)
    provisional = np.asarray(
        [1, 1, 2, 2, 4, 4, 8, 8, 3, 5, 10, 12],
        dtype=np.uint8,
    )
    authoritative = provisional.copy()
    authoritative[[1, 4, 8, 10]] ^= np.asarray([1, 4, 2, 8], dtype=np.uint8)
    guard_indices = np.asarray([1, 4, 8, 10], dtype=np.int32)
    exact_flat = np.zeros((7, counts.shape[0]), dtype=np.uint64)
    for bit in range(4):
        exact_flat[3 + bit] = counts[:, (provisional & (1 << bit)) != 0].sum(
            axis=1,
            dtype=np.uint64,
        )

    corrected, changed = _apply_mask_guard_correction(
        exact_flat,
        [(0, counts.shape[0], counts[:, guard_indices].T)],
        guard_indices,
        provisional,
        authoritative,
    )

    assert corrected is True
    assert changed == (1, 1, 1, 1)
    for bit in range(4):
        expected = counts[:, (authoritative & (1 << bit)) != 0].sum(
            axis=1,
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(exact_flat[3 + bit], expected)


def test_cuda_mask_guard_fails_closed_when_changed_pixel_was_not_retained() -> None:
    """An uncovered authoritative mask change must trigger the second pass."""

    from quantem.gpu.screening._cuda import _apply_mask_guard_correction

    exact_flat = np.zeros((7, 2), dtype=np.uint64)
    guard_counts = np.asarray([[3], [4]], dtype=np.uint16)
    provisional = np.asarray([1, 0], dtype=np.uint8)
    authoritative = np.asarray([1, 8], dtype=np.uint8)

    corrected, changed = _apply_mask_guard_correction(
        exact_flat,
        [(0, 2, guard_counts.T)],
        np.asarray([0], dtype=np.int32),
        provisional,
        authoritative,
    )

    assert corrected is False
    assert changed == (0, 0, 0, 0)
    np.testing.assert_array_equal(exact_flat, 0)


def test_cuda_screening_zero_sample_uses_private_exact_engine(
    monkeypatch,
    tmp_path,
) -> None:
    """The default public verb should route internally without a second API."""

    from quantem.gpu.screening import _cuda, workflow

    sentinel = object()
    calls = []

    def fake_build(master, **kwargs):
        calls.append((master, kwargs))
        return sentinel

    monkeypatch.setattr(_cuda, "_build_exact_cuda_products", fake_build)
    master = tmp_path / "scan_master.h5"
    plan = workflow._memory_plan_for_shapes((2, 3), (4, 5), 2, 1.0)

    result = workflow._build_cuda_products(
        master,
        scan_shape=(2, 3),
        chunk_rows=1,
        sample_positions=0,
        seed=0,
        rotation_steps=90,
        output_dtype=np.uint16,
        memory_plan=plan,
        verbose=False,
    )

    assert result is sentinel
    assert calls[0][0] == master
    assert calls[0][1]["scan_shape"] == (2, 3)


def test_exact_cuda_six_gib_auto_plan_accounts_for_decoder_working_set() -> None:
    """The 6 GiB plan must include decode scratch and upload buffers."""

    from quantem.gpu.screening._cuda import (
        _CUDA_WORKING_SET_FRACTION,
        _exact_cuda_memory_plan,
        _exact_cuda_working_set_bytes,
    )
    from quantem.gpu.screening._memory import _memory_plan_for_shapes

    initial = _memory_plan_for_shapes(
        (512, 512),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        6.0,
    )
    planned = _exact_cuda_memory_plan(initial)
    limit = int(6.0 * (1 << 30) * _CUDA_WORKING_SET_FRACTION)

    assert initial.chunk_rows == 85
    assert planned.chunk_rows == 64
    assert planned.chunk_rows_source == "budget_cuda_exact"
    assert _exact_cuda_working_set_bytes(planned, 64) <= limit
    assert _exact_cuda_working_set_bytes(planned, 65) > limit
