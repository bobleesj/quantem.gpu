"""Frozen integer detector workflows on explicit non-accelerated runners."""

import hashlib

import numpy as np
import pytest

from quantem.gpu import detector, io
from tests.parity.resident_integer_assertions import (
    _assert_frozen_products,
    _assert_invalid_requests,
    _assert_selected_frame_is_independent,
)
from tests.parity.resident_integer_oracle import (
    FIXTURE_PATH,
    _fixture,
    _half_open_mask,
    _sum_mask,
    _working_source,
)


def _session(source: np.ndarray, runner: str):
    """Exercise the same public owner on NumPy or explicit CPU tensors."""
    if runner == "torch-cpu":
        torch = pytest.importorskip("torch")
        return detector.prepare(torch.from_numpy(source).to(device="cpu"))
    return detector.prepare(source)


def test_frozen_vectors_match_independent_integer_oracle() -> None:
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == (
        "bc6de8428ffaf53d5e996470c3947328afd072cb72a4416325ce61819a600ea2"
    ), "Do not recapture v1 expectations to make a backend pass."
    case = _fixture()
    source = _working_source(case)
    np.testing.assert_array_equal(
        source.reshape(6, 12), case["expected_working_frames_u16"]
    )
    for request in case["detector_masks"]:
        mask = np.asarray(request["mask_u8"], dtype=np.uint8).reshape(3, 4)
        if "geometry" in request:
            np.testing.assert_array_equal(
                _half_open_mask((3, 4), request["geometry"]), mask
            )
        np.testing.assert_array_equal(
            _sum_mask(source, mask).reshape(-1), request["expected_sum_u64"]
        )
    selected = np.stack([
        source[row, column]
        for row, column in case["selected_scan_row_columns"]
    ])
    np.testing.assert_array_equal(
        selected.reshape(4, 12), case["expected_selected_frames_u16"]
    )
    np.testing.assert_array_equal(
        selected.astype(np.uint64).sum(axis=0).reshape(-1),
        case["selected_scan_sum_u64"],
    )


@pytest.mark.parametrize("runner", ["numpy", "torch-cpu"])
def test_move_resize_repeat_and_selected_diffraction(runner: str) -> None:
    case = _fixture()
    source = _working_source(case)
    original = source.copy()
    session = _session(source, runner)
    _assert_frozen_products(session, case)
    indices = [row * 3 + column for row, column in case["selected_scan_row_columns"]]
    np.testing.assert_array_equal(
        session.reduce_frames_exact(indices).reshape(-1),
        case["selected_scan_sum_u64"],
    )
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("runner", ["numpy", "torch-cpu"])
def test_invalid_request_does_not_change_existing_scientific_source(
    runner: str,
) -> None:
    case = _fixture()
    source = _working_source(case)
    original = source.copy()
    session = _session(source, runner)
    _assert_invalid_requests(session, case)
    np.testing.assert_array_equal(source, original)


def test_full_uint16_file_switching_preserves_counts_above_float32(
    tmp_path,
) -> None:
    import h5py

    case = _fixture()["large_sum"]
    source_a = np.full(case["shape"], case["fill"], dtype=np.uint16)
    for override in case["overrides"]:
        source_a.reshape(-1)[override["flat_index"]] = override["value"]
    source_b = source_a.copy()
    source_b.reshape(-1)[0] = 1023
    paths = []
    for name, source in (("a", source_a), ("b", source_b)):
        master = tmp_path / f"{name}_master.h5"
        shard = tmp_path / f"{name}_data_000001.h5"
        with h5py.File(shard, "w") as handle:
            handle.create_dataset("entry/data/data", data=source.reshape(2, 17, 17))
        with h5py.File(master, "w") as handle:
            handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
                shard.name, "/entry/data/data"
            )
        paths.append(master)
    for index in (0, 1, 0):
        with io.load(
            paths[index], backend="cpu", representation="dense",
            scan_shape=(1, 2), dtype="native", verbose=False,
        ) as loaded:
            result = detector.prepare(loaded).masked_sum_exact(
                np.ones((17, 17), dtype=np.uint8)
            )
            expected = case["expected_full_sum_u64"] if index == 0 else [
                18875103, 18939615
            ]
            np.testing.assert_array_equal(result.reshape(-1), expected)
            assert loaded.dtype == np.dtype("uint16")
            assert loaded.lossless
            assert result[0, 1] != np.float32(result[0, 1])


def test_closed_and_half_open_radius_profiles_are_not_interchangeable() -> None:
    case = _fixture()["geometry_counterexample"]
    closed = detector.detector_mask(
        case["center_row_column"], case["inner_radius_px"],
        case["outer_radius_px"], tuple(case["shape"]),
    )
    half_open = _half_open_mask(tuple(case["shape"]), case)
    np.testing.assert_array_equal(closed.reshape(-1), case["closed_mask_u8"])
    np.testing.assert_array_equal(
        half_open.reshape(-1), case["half_open_mask_u8"]
    )
    assert int(closed.sum()) == 5 and int(half_open.sum()) == 1


def test_selected_diffraction_never_rounds_wider_integer_counts_for_display() -> None:
    source = np.array([16777217, 4294967295], dtype=np.uint32).reshape(1, 1, 1, 2)
    diffraction = detector.prepare(source).frame(0)
    assert diffraction.dtype == source.dtype
    np.testing.assert_array_equal(diffraction, source[0, 0])


@pytest.mark.parametrize("runner", ["numpy", "torch-cpu"])
def test_selected_diffraction_is_not_a_writable_alias_of_source(runner: str) -> None:
    case = _fixture()
    source = _working_source(case)
    original = source.copy()
    _assert_selected_frame_is_independent(_session(source, runner), case)
    np.testing.assert_array_equal(source, original)
