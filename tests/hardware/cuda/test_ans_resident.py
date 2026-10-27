"""CUDA tests for exact, bounded resident ANS reads."""

import os

import numpy as np
import pytest

from quantem.gpu.io.dataset import Dataset4dstemGPU

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)

from quantem.gpu.resident.cuda.counts import StreamedCounts


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_streamed_ans_decodes_selected_scan_ranges(dtype):
    """Selected ranges remain exact across entropy chunk boundaries."""
    shape = (6, 300, 3, 4)
    maximum = 251 if dtype == np.uint8 else 4093
    values = (
        cp.arange(np.prod(shape), dtype=cp.uint64) % maximum
    ).astype(dtype).reshape(shape)
    source = StreamedCounts(shape, dtype)
    flat = values.reshape(-1, *shape[2:])
    try:
        for first, stop in ((0, 700), (700, 1500), (1500, 1800)):
            source.append(cp.ascontiguousarray(flat[first:stop]))
        for first, stop in ((510, 515), (695, 705), (1490, 1510)):
            observed = source.decode_scan_range_device(first, stop)
            assert bool(cp.all(observed == flat[first:stop]))
    finally:
        source.release()


@pytest.mark.parametrize("alignment", range(4))
def test_buffered_reads_keep_unaligned_payloads_and_short_tails_exact(alignment):
    """Mixed codecs decode exactly at every byte alignment and partial block."""
    from quantem.gpu import detector

    shape = (1, 1031, 9, 11)
    raw = cp.random.default_rng(7).poisson(5, size=(1031, 9, 11)).astype(cp.uint16)
    flat = raw.reshape(1031, -1)
    scan = cp.arange(1031, dtype=cp.uint32)
    flat[:, 0], flat[:, 1] = 0, 65535
    flat[:, 2] = (scan * 15427 % 65536).astype(cp.uint16)
    flat[:, 3] = cp.where(scan % 13 == 0, 50000 + scan % 500, scan % 7).astype(cp.uint16)
    source = StreamedCounts(shape, np.uint16)
    source.interval = 93
    session = None
    try:
        for first, stop in ((0, 129), (129, 895), (895, 1031)):
            source.append(cp.ascontiguousarray(raw[first:stop]))
        modes, ans_lengths = set(), set()
        for chunk in source.chunks:
            payload, offsets, models, *index = chunk.arrays
            padded = cp.empty(payload.size + alignment, cp.uint8)
            padded[alignment:] = payload
            chunk.arrays = (padded[alignment:], offsets, models, *index)
            modes.update(cp.unique(models).get().tolist())
            ans_lengths.update(cp.unique((offsets[1:] - offsets[:-1])[models < 64] % 4).get().tolist())
        assert {253, 254, 255} <= modes and any(mode < 64 for mode in modes)
        assert int(source.chunks[0].arrays[2][3].get()) < 64
        assert {1, 2, 3} <= ans_lengths
        for first, stop in ((0, 1031), (92, 95), (128, 132), (894, 897), (1029, 1031)):
            decoded = source.decode_scan_range_device(first, stop)
            assert bool(cp.array_equal(decoded, raw[first:stop]))
        session = detector.prepare(source)
        rows, cols = np.indices(shape[2:])
        for mask in ((rows + cols) % 3 == 0, cols < 5, rows >= 4):
            expected = raw[:, cp.asarray(mask)].sum(axis=1, dtype=cp.uint64)
            actual = session.masked_sum(mask, output="native").reshape(-1)
            assert bool(cp.array_equal(actual, expected))
        assert bool(cp.array_equal(session.frame(1030, output="native"), raw[-1]))
    finally:
        if session is not None:
            session.close()
        source.release()


@pytest.mark.parametrize("damage", ["header", "raw", "renormalization", "escape", "trailing"])
def test_buffered_reads_reject_truncated_or_trailing_bytes(damage):
    """Prefetch cannot consume bytes outside a stream or conceal unused bytes."""
    from quantem.gpu import detector

    scan = cp.arange(93, dtype=cp.uint32)
    raw = cp.full((93, 3, 3), 11, dtype=cp.uint16)
    raw[:, 1, 1] = (scan * 15427 % 65536 if damage == "raw" else scan % 7).astype(cp.uint16)
    if damage == "escape":
        raw[-1, 1, 1] = 50000
    source = StreamedCounts((1, 93, 3, 3), np.uint16)
    session = None
    try:
        source.append(raw)
        chunk = source.chunks[0]
        payload, offsets, models, *index = chunk.arrays
        padded = cp.empty(payload.size + 1, cp.uint8)
        padded[1:] = payload
        chunk.arrays = (padded[1:], offsets, models, *index)
        mode = int(models[4].get())
        assert mode == 254 if damage == "raw" else mode < 64
        begin, end = int(offsets[4].get()), int(offsets[5].get())
        offsets[5] = begin + 3 if damage == "header" else end + (1 if damage == "trailing" else -1)
        with pytest.raises(ValueError, match="failed reconstruction"):
            source.decode_scan_range_device(0, 93, detector_region=(1, 2, 1, 2))
        session = detector.prepare(source)
        mask = np.zeros((3, 3), bool)
        mask[1, 1] = True
        with pytest.raises(ValueError, match="exact decoding"):
            session.masked_sum(mask, output="native")
        offsets[5] = end
        assert bool(cp.array_equal(source.decode_scan_range_device(0, 93), raw))
        actual = session.masked_sum(mask, output="native").reshape(-1)
        assert bool(cp.array_equal(actual, raw[:, 1, 1]))
    finally:
        if session is not None:
            session.close()
        source.release()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("interval", [512, 93])
def test_many_field_detector_masks_keep_exact_sums(dtype, interval):
    """Full and incremental masks retain high counts across partial scan blocks."""
    from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute

    shape = (1, 133, 257, 257)
    maximum = np.iinfo(dtype).max
    values = (maximum - cp.arange(np.prod(shape), dtype=cp.uint32) % 31)
    values = values.astype(dtype).reshape(-1, *shape[2:])
    valid = np.ones(shape[2:], bool)
    valid[3, 5] = False
    rows, cols = np.indices(shape[2:])
    checker = ((rows // 8 + cols // 8) % 2) == 0
    changed = checker.copy()
    changed[64:72, 64:72] ^= True
    radius = (rows - 123.5) ** 2 + (cols - 131.5) ** 2
    masks = [
        np.ones(shape[2:], bool),
        checker,
        changed,
        checker,
        (radius >= 40**2) & (radius < 110**2),
        np.zeros(shape[2:], bool),
    ]
    source = StreamedCounts(shape, dtype, valid)
    source.interval = interval
    try:
        source.append(cp.ascontiguousarray(values[:129]))
        source.append(cp.ascontiguousarray(values[129:]))
        query = StreamedSeriesCompute([source])
        for mask in masks:
            expected = values[:, cp.asarray(mask & valid)].sum(axis=1, dtype=cp.uint64)
            observed = query.masked_sum_exact_native(mask).reshape(-1)
            assert bool(cp.array_equal(observed, expected))
        if dtype == np.uint16:
            expected = values[:, cp.asarray(valid)].sum(axis=1, dtype=cp.uint64)
            assert bool(cp.all(expected > np.iinfo(np.uint32).max))
    finally:
        source.release()


def _alternative_stream(mode):
    """One selected sparse, constant or zero stream, with neighboring payloads."""
    raw = cp.zeros((16, 8, 8), cp.uint16)
    raw[:, 1, 0] = 3
    raw[:, 1, 2] = 11
    if mode == 252:
        raw[1, 1, 1], raw[7, 1, 1] = 5, 9
    elif mode == 255:
        raw[:, 1, 1] = 65535
    source = StreamedCounts((1, 16, 8, 8), np.uint16)
    source.append(raw)
    mask = np.zeros((8, 8), bool)
    mask[1, 1] = True
    assert int(source.chunks[0].arrays[2][9].get()) == mode
    return source, raw, mask


def test_seven_acquisitions_keep_tail_scans_and_moments_exact():
    """Joint detector and CoM queries include the scan after sixteen full blocks."""
    from quantem.gpu import detector

    shape = (1, 8193, 17, 19)
    values = cp.arange(np.prod(shape), dtype=cp.uint32).reshape(-1, *shape[2:])
    raw = [(values * (index + 1) % 17).astype(cp.uint16) for index in range(7)]
    sources = [StreamedCounts(shape, np.uint16) for _ in raw]
    session = None
    try:
        for source, counts in zip(sources, raw):
            source.append(counts)
        session = detector.prepare(sources)
        rows, cols = np.indices(shape[2:])
        radius = (rows - 8) ** 2 + (cols - 9) ** 2
        for mask in (radius < 7**2, radius >= 7**2, (rows + cols) % 3 == 0):
            expected = cp.stack([counts[:, cp.asarray(mask)].sum(1, dtype=cp.uint64) for counts in raw])
            actual = session.masked_sum(mask, output="native").reshape(7, -1)
            assert bool(cp.array_equal(actual, expected))
        observed_row, observed_col = session.center_of_mass(mask)
        denominator = cp.maximum(expected.astype(cp.float64), 1)
        for coordinates, observed in ((cols, observed_col), (rows, observed_row)):
            weights = cp.asarray(coordinates * mask)
            moments = cp.stack([(counts * weights).sum((1, 2), dtype=cp.uint64) for counts in raw])
            reference = (moments.astype(cp.float64) / denominator).astype(cp.float32)
            np.testing.assert_array_equal(observed.reshape(7, -1), reference.get())
        np.testing.assert_array_equal(session.frame(8192), cp.stack([counts[-1] for counts in raw]).get())
    finally:
        if session is not None:
            session.close()
        for source in sources:
            source.release()


@pytest.mark.parametrize("operation", ["detector", "moments"])
@pytest.mark.parametrize(
    "mode,damage",
    [(252, "odd"), (252, "duplicate"), (252, "reversed"),
     (252, "backwards_extent"), (255, "short"), (255, "long"),
     (255, "backwards_extent"), (253, "nonempty")],
)
def test_malformed_alternative_streams_invalidate_detector_cache(operation, mode, damage):
    """Malformed shortcuts raise and cannot seed a later incremental detector image."""
    from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute

    source, raw, mask = _alternative_stream(mode)
    try:
        query = StreamedSeriesCompute([source])
        query.masked_sum_native(np.zeros_like(mask))
        payload, offsets = source.chunks[0].arrays[:2]
        original_payload, original_offsets = payload.copy(), offsets.copy()
        begin, end = int(offsets[9].get()), int(offsets[10].get())
        if damage == "duplicate":
            payload[begin + 2:begin + 4] = payload[begin:begin + 2]
        elif damage == "reversed":
            payload[begin:begin + 4] = payload[begin:begin + 4].reshape(2, 2)[::-1].copy().ravel()
        elif damage == "backwards_extent":
            offsets[10] = begin - 1
        else:
            offsets[10] = end + (-1 if damage in {"odd", "short"} else 1)
        with pytest.raises(ValueError, match="exact decoding"):
            if operation == "detector":
                query.masked_sum_native(mask)
            else:
                query._moments(mask.astype(np.uint32), mask.astype(np.uint32))
        assert query.previous_mask is None
        payload[:], offsets[:] = original_payload, original_offsets
        expected = raw[:, cp.asarray(mask)].sum(axis=1, dtype=cp.uint64)
        assert bool(cp.array_equal(query.masked_sum_native(mask).reshape(-1), expected))
        assert not query.last["incremental"]
    finally:
        source.release()


def test_queued_delta_cannot_publish_after_a_malformed_stream():
    """A delta queued before a decode error is reported also fails, then a full retry recovers."""
    from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute

    source, raw, mask = _alternative_stream(252)
    try:
        query = StreamedSeriesCompute([source])
        query.masked_sum_native(np.zeros_like(mask))
        offsets = source.chunks[0].arrays[1]
        end = int(offsets[10].get())
        offsets[10] = end - 1
        query.masked_sum_native(mask, wait=False)
        query.masked_sum_native(mask, wait=False)
        for _ in range(2):
            with pytest.raises(ValueError, match="exact decoding"):
                query.finish()
        assert query.previous_mask is None
        offsets[10] = end
        expected = raw[:, cp.asarray(mask)].sum(axis=1, dtype=cp.uint64)
        assert bool(cp.array_equal(query.masked_sum_native(mask).reshape(-1), expected))
        assert not query.last["incremental"]
    finally:
        source.release()


def test_saved_copy_holds_each_encoded_array_at_its_offset(tmp_path, monkeypatch):
    """Arrays leave the GPU in pinned pieces; pieces far smaller than an array still land in place."""
    from quantem.gpu import io
    from quantem.gpu.formats.qem.snapshot import read_header
    from quantem.gpu.io import qem

    shape = (6, 300, 3, 4)
    values = (cp.arange(np.prod(shape), dtype=cp.uint64) % 4093).astype(cp.uint16).reshape(-1, *shape[2:])
    source = StreamedCounts(shape, np.uint16)
    path = tmp_path / "copy.qem"
    try:
        for first, stop in ((0, 700), (700, 1800)):
            source.append(cp.ascontiguousarray(values[first:stop]))
        with monkeypatch.context() as patch:
            patch.setattr(qem, "BLOCK_BYTES", 1000)
            qem.save_streamed(path, source, {})
        header, start = read_header(path)
        expected = bytearray(header["bytes"])
        for chunk, entry in zip(source.chunks, header["chunks"]):
            for array, span in zip(chunk.arrays, entry["arrays"]):
                expected[span["offset"] : span["offset"] + array.nbytes] = cp.asnumpy(array).tobytes()
        assert path.read_bytes()[start:] == bytes(expected)
    finally:
        source.release()
    with io.load(path, backend="cuda", verbose=False) as reopened:
        assert bool(cp.all(reopened.data.decode_scan_range_device(0, 1800) == values))


def test_resident_read_matches_numpy_region():
    """The public read contract restores requested CUDA rows and columns."""
    import torch

    shape = (5, 6, 4, 7)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 251
    valid = np.ones(shape[2:], dtype=bool)
    valid[2, 4] = False
    source = StreamedCounts(shape, np.uint16, valid)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    loaded = Dataset4dstemGPU(
        source,
        {
            "working_shape": shape,
            "working_dtype": "uint16",
            "representation": "encoded",
        },
    )
    try:
        observed = loaded.read(
            scan_region=(1, 4, 2, 6),
            detector_region=(1, 4, 3, 7),
        )
        assert observed.device.type == "cuda"
        assert observed.dtype == torch.uint16
        expected = values.get()[1:4, 2:6, 1:4, 3:7]
        np.testing.assert_array_equal(observed.cpu().numpy(), expected)
    finally:
        loaded.close()


def test_detector_region_read_decodes_only_the_cropped_counts(monkeypatch):
    """A detector crop decodes the cropped pixels only: whole scan rows straight into the result, others in bounded blocks.

    Decoding the whole region before cropping held a full-detector copy of every
    requested scan position (19 GB for a 512 x 512 x 192 x 192 uint16 scan just to
    keep a 110 x 110 bright-field disk).
    """
    from quantem.gpu.io import read

    shape = (7, 5, 6, 8)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 251
    source = StreamedCounts(shape, np.uint16)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    frame_bytes = 6 * 8 * 2
    monkeypatch.setattr(read, "_BLOCK_BYTES", 2 * 3 * frame_bytes)
    decoded = []
    decode = source.decode_scan_range_device
    def record_decode(first, stop, *, detector_region=None):
        block = decode(first, stop, detector_region=detector_region)
        decoded.append(block.nbytes)
        return block

    monkeypatch.setattr(source, "decode_scan_range_device", record_decode)
    loaded = Dataset4dstemGPU(
        source,
        {"working_shape": shape, "working_dtype": "uint16", "representation": "encoded"},
    )
    try:
        for scan_region in ((0, 7, 0, 5), (1, 6, 1, 4)):
            decoded.clear()
            observed = loaded.read(scan_region=scan_region, detector_region=(1, 5, 2, 7))
            row0, row1, column0, column1 = scan_region
            expected = values.get()[row0:row1, column0:column1, 1:5, 2:7]
            np.testing.assert_array_equal(observed.cpu().numpy(), expected)
            assert sum(decoded) == expected.nbytes
            if (column0, column1) == (0, shape[1]):
                assert len(decoded) == 1
            else:
                assert len(decoded) > 1
                assert max(decoded) <= read._BLOCK_BYTES
    finally:
        loaded.close()


@pytest.mark.parametrize("layout", ["streamed", "paired"])
def test_reads_of_scans_not_held_raise(layout):
    """Scans not yet appended, freed, or released raise instead of returning stale device memory."""
    from quantem.gpu.resident.cuda.paired import PairedCounts

    shape = (4, 512, 8, 8)
    values = (cp.arange(np.prod(shape), dtype=cp.uint64) % 4093).astype(cp.uint16)
    flat = values.reshape(-1, *shape[2:])
    source = (PairedCounts if layout == "paired" else StreamedCounts)(shape, "uint16")
    for first in (0, 512, 1024):
        source.append(cp.ascontiguousarray(flat[first:first + 512]))
    # Fill the memory pool with a pattern a stale read would return.
    cp.full(flat.shape, 0xABCD, cp.uint16)
    assert bool(cp.all(source.decode_scan_range_device(1000, 1536) == flat[1000:1536]))
    for first, stop in ((1500, 1600), (1536, 2048)):
        with pytest.raises(ValueError):
            source.decode_scan_range_device(first, stop)
    source.release_scans_before(512)
    with pytest.raises(ValueError):
        source.decode_scan_range_device(100, 600)
    assert bool(cp.all(source.decode_scan_range_device(512, 600) == flat[512:600]))
    source.release()
    with pytest.raises(ValueError):
        source.decode_scan_range_device(600, 700)


@pytest.mark.parametrize("layout", ["streamed", "paired"])
def test_detector_total_covers_the_whole_scan_or_raises(layout):
    """The total kept while encoding and the total of chunks that arrive encoded are exact; scans arriving or freed raise."""
    from quantem.gpu.resident.cuda.paired import PairedCounts

    shape = (4, 512, 8, 8)
    flat = (cp.arange(np.prod(shape), dtype=cp.uint64) * 7919 % 65536).astype(cp.uint16).reshape(-1, *shape[2:])
    valid = np.ones(shape[2:], bool)
    valid[2, 3] = False
    expected = (flat.sum(axis=0, dtype=cp.uint64) * cp.asarray(valid)).get()
    owner = PairedCounts if layout == "paired" else StreamedCounts
    source = owner(shape, "uint16", valid)
    for first in range(0, 2048, 512):
        with pytest.raises(ValueError, match="not resident"):
            source.detector_total_device()
        source.append(cp.ascontiguousarray(flat[first:first + 512]))
    np.testing.assert_array_equal(source.detector_total_device().get(), expected)
    # chunks that arrive encoded, as from a file, carry no total: the first request adds their streams
    reopened = owner(shape, "uint16", valid)
    reopened.chunks.extend(source.chunks)
    reopened.ready_scans = source.ready_scans
    np.testing.assert_array_equal(reopened.detector_total_device().get(), expected)
    np.testing.assert_array_equal(reopened.detector_total_device().get(), expected)
    source.release_scans_before(1024)
    with pytest.raises(ValueError, match="released"):
        source.detector_total_device()
