"""CPU-only source-owner contract tests; callbacks use tiny NumPy fixtures."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.detector import prepare
from quantem.gpu.io._resident import CudaResidentSource
from quantem.gpu.io.load import LoadResult


def make_source(source_type=CudaResidentSource):
    counts = np.full((66, 2, 3, 18, 18), 60000, np.uint16)
    counts[0, 0, 0, 0, 0] += 1
    counts[..., -1, -1] = 65535
    owner = SimpleNamespace(counts=counts, calls=[], inside=False)
    valid = np.ones((18, 18), np.bool_)
    valid[-1, -1] = False

    def execute(task):
        assert not owner.inside
        owner.inside = True
        try:
            return task()
        finally:
            owner.inside = False

    def integrate(current, mask):
        assert current is owner and owner.inside
        owner.calls.append(("integrate", mask.copy()))
        return counts[..., mask.astype(bool)].sum(axis=-1, dtype=np.uint32)

    def patterns(current, index):
        assert current is owner and owner.inside
        owner.calls.append(("patterns", index))
        return np.ascontiguousarray(counts.reshape(66, 6, 18, 18)[:, index])

    return source_type(owner, shape=counts.shape, device="cuda:1", generation="fixture-1",
        storage_format="fixture-v1", source_codec="numpy-uint16-v1", sparse_codec="none-v1",
        resident_bytes=123456, valid_pixels=valid, execute=execute,
        integrate=integrate, patterns=patterns), owner


def test_all66_one_backend_call_and_raw_precision_and_validity():
    source, owner = make_source()
    original = owner.counts.copy()
    session = prepare(LoadResult(source, {}))
    actual = session.masked_sum_batch_exact(np.ones((18, 18), bool))
    expected = original[..., source.valid_pixels].sum(axis=-1, dtype=np.uint32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype.str == "<u4" and actual.shape == (66, 2, 3)
    assert int(actual[0, 0, 0]) > 2**24 and int(actual[0, 0, 0]) % 2 == 1
    assert len(owner.calls) == 1
    assert not actual.flags.writeable
    assert source.storage_metadata["source_codec"] == "numpy-uint16-v1"
    with pytest.raises(TypeError):
        source.storage_metadata["source_codec"] = "different-v1"
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    owner.counts[...] = 1
    np.testing.assert_array_equal(actual, expected)
    assert original[0, 0, 0, -1, -1] == 65535


def test_point_dps_all66_and_exact_selected_view():
    source, owner = make_source()
    session = prepare(source)
    patterns = session.frame_batch(5)
    np.testing.assert_array_equal(patterns, owner.counts.reshape(66, 6, 18, 18)[:, 5])
    assert patterns.dtype == np.uint16 and len(owner.calls) == 1
    assert patterns[0, -1, -1] == 65535
    selected = prepare(source[65])
    np.testing.assert_array_equal(selected.frame(0), owner.counts[65, 0, 0])
    assert len(owner.calls) == 2
    assert selected.scan_shape == (2, 3)
    with pytest.raises(ValueError, match="all-acquisition"):
        session.masked_sum_exact(np.ones((18, 18), bool))
    assert len(owner.calls) == 2


def test_explicit_route_precedes_ambiguous_chunk_protocol():
    class SourceWithChunks(CudaResidentSource):
        @property
        def chunks(self):
            raise AssertionError("Compact CUDA data must not enter generic Metal chunk routing")

    source, _ = make_source(SourceWithChunks)
    assert prepare(source).frame_batch(0).shape == (66, 18, 18)
    with pytest.raises(TypeError, match="materialized"):
        np.asarray(source)


def test_invalid_masks_and_indices_do_not_dispatch():
    source, owner = make_source()
    for mask in (np.ones((1, 1)), np.full((18, 18), .5), np.full((18, 18), np.nan)):
        with pytest.raises(ValueError):
            source.masked_sum_batch_exact(mask)
    for index in (-1, 6, True, 1.5):
        with pytest.raises(IndexError):
            source.frame_batch(index)
    assert owner.calls == []
    empty = source.masked_sum_batch_exact(np.zeros((18, 18), bool))
    np.testing.assert_array_equal(empty, np.zeros((66, 2, 3), np.uint32))


def test_executor_is_serialized_across_concurrent_clients():
    source, owner = make_source()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(source.frame_batch, index) for index in (0, 5)]
        actual = [future.result() for future in futures]
    assert len(owner.calls) == 2
    np.testing.assert_array_equal(actual[0], owner.counts[:, 0, 0])
    np.testing.assert_array_equal(actual[1], owner.counts[:, 1, 2])


@pytest.mark.parametrize("kind", ["partial", "float", "executor", "callback"])
def test_failed_or_incomplete_batches_keep_owner_and_stop_dispatch(kind):
    source, owner = make_source()
    if kind == "partial":
        source._integrate = lambda *_: np.zeros((65, 2, 3), np.uint32)
    elif kind == "float":
        source._integrate = lambda *_: np.zeros((66, 2, 3), np.float32)
    elif kind == "executor":
        source._execute = lambda _task: np.zeros((66, 2, 3), np.uint32)
    else:
        def failed(*_):
            raise RuntimeError("simulated owner-thread failure")
        source._integrate = failed
    with pytest.raises((ValueError, RuntimeError)):
        source.masked_sum_batch_exact(np.ones((18, 18), bool))
    assert source._failure is not None and source._owner is owner
    with pytest.raises(RuntimeError, match="failed"):
        source.frame_batch(0)


def test_existing_array_sessions_keep_4d_contract():
    counts = np.arange(2*3*4*5, dtype=np.uint16).reshape(2, 3, 4, 5)
    session = prepare(counts)
    np.testing.assert_array_equal(session.masked_sum_exact(np.ones((4, 5), bool)),
                                  counts.sum(axis=(-2, -1), dtype=np.uint64))
    with pytest.raises(NotImplementedError, match="all-acquisition"):
        session.frame_batch(0)


def test_wrong_device_result_is_rejected_before_host_copy():
    source, _ = make_source()
    def forbidden_get(**kwargs):
        raise AssertionError("Do not copy a result belonging to another CUDA device")
    foreign = SimpleNamespace(shape=(66,2,3), dtype=np.dtype("<u4"),
                              flags=SimpleNamespace(c_contiguous=True),
                              device=SimpleNamespace(id=0), get=forbidden_get)
    source._integrate = lambda *_: foreign
    with pytest.raises(ValueError, match="different CUDA"):
        source.masked_sum_batch_exact(np.ones((18,18), bool))


@pytest.mark.parametrize("readout", ["virtual_images", "diffraction_patterns"])
def test_complete_immutable_responses_are_retained_across_requests(readout):
    """Retain an exact completed batch while the owner publishes the next one."""
    source, owner = make_source()
    session = prepare(source)
    if readout == "virtual_images":
        expected = owner.counts[..., source.valid_pixels].sum(axis=-1, dtype=np.uint32)
        request = lambda: session.masked_sum_batch_exact(np.ones((18, 18), bool))
        callback_name = "_integrate"
    else:
        expected = np.ascontiguousarray(owner.counts[:, 0, 0])
        request = lambda: session.frame_batch(0)
        callback_name = "_patterns"
    owner.response = np.frombuffer(expected.tobytes(), expected.dtype).reshape(expected.shape)

    def response(current, argument):
        assert current is owner and owner.inside
        owner.calls.append((readout, argument))
        return owner.response

    setattr(source, callback_name, response)
    saved = request()
    assert saved is owner.response
    with pytest.raises(ValueError):
        saved.setflags(write=True)
    next_values = np.full(expected.shape, 17, expected.dtype)
    owner.response = np.frombuffer(next_values.tobytes(), next_values.dtype).reshape(next_values.shape)
    current = request()
    assert current is owner.response and current is not saved
    np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(current, next_values)
    assert len(owner.calls) == 2


@pytest.mark.parametrize("storage", [
    "mutable", "readonly_mutable_base", "bytearray", "subclass",
    "partial_bytes", "memoryview_bytes",
])
def test_saved_virtual_images_snapshot_reusable_or_indirect_buffers(storage):
    """Saving a batch must not retain an owner's mutable or partial buffer."""
    source, owner = make_source()
    expected = owner.counts[..., source.valid_pixels].sum(axis=-1, dtype=np.uint32)
    mutable = expected.copy()
    if storage == "mutable":
        response = mutable
    elif storage == "readonly_mutable_base":
        response = mutable.view()
        response.setflags(write=False)
    elif storage == "bytearray":
        allocation = bytearray(expected.tobytes())
        response = np.frombuffer(allocation, expected.dtype).reshape(expected.shape)
        response.setflags(write=False)
    elif storage == "subclass":
        class CountsArray(np.ndarray):
            pass

        response = np.frombuffer(expected.tobytes(), expected.dtype).reshape(expected.shape).view(CountsArray)
    elif storage == "partial_bytes":
        allocation = expected.tobytes() + bytes(expected.dtype.itemsize)
        response = np.frombuffer(allocation, expected.dtype)[:-1].reshape(expected.shape)
    else:
        response = np.frombuffer(memoryview(expected.tobytes()), expected.dtype).reshape(expected.shape)
    owner.response = response

    def integrate(current, _mask):
        assert current is owner and owner.inside
        return owner.response

    source._integrate = integrate
    saved = source.masked_sum_batch_exact(np.ones((18, 18), bool))
    assert type(saved) is np.ndarray and saved is not response
    assert not np.shares_memory(saved, response)
    with pytest.raises(ValueError):
        saved.setflags(write=True)
    mutable.fill(3)
    if storage == "bytearray":
        allocation[:] = bytes(len(allocation))
    owner.response = np.full(expected.shape, 7, expected.dtype)
    current = source.masked_sum_batch_exact(np.ones((18, 18), bool))
    np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(current, np.full(expected.shape, 7, expected.dtype))


def test_executor_cannot_replace_a_completed_batch_with_a_readonly_mutable_alias():
    """A dispatcher cannot weaken the saved scientific batch's ownership."""
    source, owner = make_source()
    execute = source._execute

    def substitute(task):
        owner.reusable_output = execute(task).copy()
        response = owner.reusable_output.view()
        response.setflags(write=False)
        return response

    source._execute = substitute
    with pytest.raises(ValueError, match="immutable batch"):
        source.masked_sum_batch_exact(np.ones((18, 18), bool))
    assert source._failure is not None and source._owner is owner
    with pytest.raises(RuntimeError, match="failed"):
        source.frame_batch(0)


def test_device_batches_keep_the_blocking_snapshot_before_owner_reuse():
    """A completed host result survives reuse of the device readback buffer."""
    source, owner = make_source()
    expected = owner.counts[..., source.valid_pixels].sum(axis=-1, dtype=np.uint32)
    owner.host_buffer = expected.copy()
    copies = []

    def get(*, blocking):
        assert owner.inside and blocking is True
        copies.append(blocking)
        return owner.host_buffer

    device_result = SimpleNamespace(
        shape=expected.shape, dtype=expected.dtype,
        flags=SimpleNamespace(c_contiguous=True),
        device=SimpleNamespace(id=1), get=get,
    )
    source._integrate = lambda *_: device_result
    saved = source.masked_sum_batch_exact(np.ones((18, 18), bool))
    owner.host_buffer.fill(9)
    current = source.masked_sum_batch_exact(np.ones((18, 18), bool))
    np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(current, np.full(expected.shape, 9, expected.dtype))
    assert copies == [True, True]
