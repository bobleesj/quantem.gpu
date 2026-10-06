import numpy as np
import pytest


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


def test_screening_cache_roundtrip_preserves_exact_uint64_products(
    tmp_path,
) -> None:
    """Exact count products must survive cache persistence without conversion."""

    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    expected = _result(master, workflow)
    maximum = np.iinfo(np.uint64).max
    expected.total_intensity = np.full((4, 4), maximum, dtype=np.uint64)
    expected.annular_bright_field = np.full(
        (4, 4),
        maximum - np.uint64(1),
        dtype=np.uint64,
    )
    expected.annular_dark_field = np.full(
        (4, 4),
        np.uint64(2**63 + 17),
        dtype=np.uint64,
    )

    workflow._save_cache(expected, cache_path)
    actual = workflow._prepare_cache(cache_path, master)

    assert actual is not None
    for name in (
        "total_intensity",
        "annular_bright_field",
        "annular_dark_field",
    ):
        observed = getattr(actual, name)
        assert observed is not None
        assert observed.dtype == np.uint64
        np.testing.assert_array_equal(observed, getattr(expected, name))


def test_screening_cache_rejects_partial_or_inexact_count_products(tmp_path) -> None:
    """A cache may contain all exact count maps or none, never a partial set."""

    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    result = _result(master, workflow)
    result.total_intensity = np.ones((4, 4), dtype=np.uint64)

    with pytest.raises(ValueError, match="must include"):
        workflow._save_cache(result, cache_path)

    result.annular_bright_field = np.ones((4, 4), dtype=np.uint64)
    result.annular_dark_field = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="must preserve exact uint64"):
        workflow._save_cache(result, cache_path)

    np.savez(
        cache_path,
        metadata_json=workflow._metadata_array(result.metadata),
        mean_dp=result.mean_dp,
        bright_field=result.bright_field,
        dark_field=result.dark_field,
        dpc_phase=result.dpc_phase,
        com_row=result.com_row,
        com_col=result.com_col,
        total_intensity=result.total_intensity,
    )
    assert workflow._prepare_cache(cache_path, master) is None


def test_screening_cache_without_retained_phase_is_rebuilt(tmp_path) -> None:
    """A version-3 cache written before the phase was cached is not reused."""
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

    assert workflow._prepare_cache(cache_path, master) is None


def test_screening_result_legacy_positional_constructor_is_compatible() -> None:
    """Appending exact products must not move any historical positional field."""

    from quantem.gpu.screening import ScreeningResult

    scan = np.zeros((2, 3), dtype=np.float32)
    result = ScreeningResult(
        scan,
        scan,
        scan,
        scan,
        scan,
        scan,
        (1.0, 2.0),
        3.0,
        4.0,
        False,
        {},
        None,
        False,
        5.0,
    )

    assert result.elapsed_s == 5.0
    assert result.total_intensity is None
    assert result.annular_bright_field is None
    assert result.annular_dark_field is None


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
    assert workflow._strong_cached_source_match(source, target) is None
    dotted_master = master.parent / "unused" / ".." / master.name
    assert workflow._strong_cached_source_match(source, dotted_master) is True


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


def test_strong_cache_identity_reinspects_master_alias_shard_mapping(tmp_path) -> None:
    import h5py

    from quantem.gpu.screening import workflow

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    master_target = source_dir / "scan_master.h5"
    with h5py.File(master_target, "w") as handle:
        group = handle.require_group("entry/data")
        group["data_000001"] = h5py.ExternalLink(
            "scan_data.h5",
            "/entry/data/data",
        )

    aliases = []
    for name, count in (("first", 3), ("second", 7)):
        alias_dir = tmp_path / name
        alias_dir.mkdir()
        alias_master = alias_dir / master_target.name
        alias_master.symlink_to(master_target)
        with h5py.File(alias_dir / "scan_data.h5", "w") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=np.full((1, 2, 2), count, dtype=np.uint16),
            )
        aliases.append(alias_master)

    first_source = workflow._source_fingerprint(aliases[0])
    second_source = workflow._source_fingerprint(aliases[1])

    assert first_source != second_source
    assert workflow._strong_cached_source_match(first_source, aliases[0]) is True
    assert workflow._strong_cached_source_match(first_source, aliases[1]) is None
    assert not workflow._cache_matches(
        {"version": workflow._CACHE_VERSION, "source": first_source},
        aliases[1],
    )


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
