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
        mask[int(round(center[0])), int(round(center[1]))] = True
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


def test_screening_cache_roundtrip(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    expected = _result(master, workflow)

    workflow._save_cache(expected, cache_path)
    actual = workflow._prepare_cache(cache_path, master)

    assert actual is not None
    assert actual.from_cache is True
    assert actual.cache_path == cache_path
    assert actual.probe_center == (3.5, 4.5)
    assert actual.probe_radius == 2.0
    np.testing.assert_array_equal(actual.mean_dp, expected.mean_dp)
    np.testing.assert_array_equal(actual.bright_field, expected.bright_field)
    np.testing.assert_array_equal(actual.dark_field, expected.dark_field)
    assert actual.dpc_phase.dtype == np.float32


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
