"""Paired-count resident layout: exact queries, malformed streams, saved form, real data."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if not cp.cuda.runtime.getDeviceCount():
    pytest.skip("paired layout kernels need a CUDA device", allow_module_level=True)

from quantem.gpu import detector
from quantem.gpu.detector.cuda.paired_series import PairedSeriesCompute
from quantem.gpu.resident.cuda.paired import PairedCounts


def _synthetic(q, scans, seed=193):
    """Counts with a wide literal row, a saturated row and one invalid pixel."""
    rng = cp.random.RandomState(seed)
    raw = rng.poisson(0.4, (scans, q, q)).astype(cp.uint16)
    if q >= 200:
        raw[:] = 65535
        raw[:, 0, :16] = rng.poisson(5, (scans, 16)).astype(cp.uint16)
    else:
        raw[:, 0, :] = rng.randint(0, 65536, (scans, q), dtype=cp.uint16)
        raw[:, 1, :] = 65535
    valid = np.ones((q, q), bool)
    valid[2, 3] = False
    return raw, valid


@pytest.mark.slow
@pytest.mark.parametrize("q,scans", [(17, 512), (19, 1024), (257, 512)])
def test_masks_and_frames_match_direct_sums(q, scans):
    raw, valid = _synthetic(q, scans)
    source = PairedCounts((1, scans, q, q), np.uint16, valid)
    source.append(raw)
    assert bool(cp.array_equal(source.decode_scan_range_device(0, scans), raw).get())
    session = detector.prepare([source])
    assert isinstance(session._backend, PairedSeriesCompute)
    masks = [np.ones((q, q), bool), np.zeros((q, q), bool)]
    for pose in ((q * 0.5 + 0.125, q * 0.5 + 0.375, 2.25, q * 0.45), (-2.125, q - 1.375, 1.25, q * 0.6)):
        masks.append(detector.detector_mask(pose[:2], pose[2], pose[3], (q, q), dtype=np.float64))
    masks += [masks[0], masks[2], masks[0]]  # repeats exercise the incremental path
    largest = 0
    for mask in masks:
        expected = raw[:, cp.asarray(mask & valid)].sum(axis=1, dtype=cp.uint64)
        actual = session.masked_sum(mask, output="native").reshape(-1)
        assert bool(cp.array_equal(actual, expected).get())
        largest = max(largest, int(expected.max().get()))
    assert (largest > 2**32) == (q >= 200)
    for scan in (0, scans - 1):
        frame = session.frame(scan, output="native").reshape(q, q)
        assert bool(cp.array_equal(frame[cp.asarray(valid)], raw[scan][cp.asarray(valid)]).get())


def test_descriptor_rows_hold_exactly_the_fields_the_kernels_read():
    """Python builds the chunk rows that the streamed and paired kernels index: one width, no unused slot."""
    import re

    from quantem.gpu.resident.cuda import counts

    kernels = Path(counts.__file__).with_name("kernels")
    widths = {
        int(re.search(rf"{name} = (\d+);", (kernels / file).read_text()).group(1))
        for name, file in (("SC_DESCRIPTOR", "streamed.cu"), ("PM_DESCRIPTOR", "paired.cu"))
    }
    raw, valid = _synthetic(19, 1024)
    for source in (counts.StreamedCounts((1, 1024, 19, 19), np.uint16, valid), PairedCounts((1, 1024, 19, 19), np.uint16, valid)):
        for first in (0, 512):
            source.append(cp.ascontiguousarray(raw[first : first + 512]))
        rows = detector.prepare([source])._backend.descriptors.get()
        chunk = source.chunks[1]
        assert widths == {rows.shape[1]}
        assert rows[1].tolist() == [
            *(array.data.ptr for array in chunk.arrays[:3]),
            source.decoding.data.ptr,
            *(array.data.ptr for array in chunk.arrays[3:]),
            chunk.first,
            chunk.scans,
            0,
            source.valid.data.ptr,
            source.interval,
        ]


def test_weighted_sums_beyond_the_int32_lane_bound_are_exact():
    """The paired residual decoder adds 32 pixels' weight x count in int32; large weights go in digits."""
    raw, valid = _synthetic(19, 512)
    raw[:256] = 65535
    source = PairedCounts((1, 512, 19, 19), np.uint16, valid)
    source.append(raw)
    session = detector.prepare(source)
    rows, cols = np.indices((19, 19))
    working = raw.astype(cp.uint64) * cp.asarray(valid)
    for weights in (2000 + rows, 1_000_000 * cols + 7):
        assert 32 * 65535 * int(weights.max()) > 2**31
        expected = (working * cp.asarray(weights, dtype=cp.uint64)).sum(axis=(1, 2)).get()
        np.testing.assert_array_equal(session.weighted_sum_exact(weights).reshape(-1), expected)


def test_queries_on_more_than_65535_pixels_are_exact():
    """A 300 x 300 detector: row and column moments decode about 90,000 residual pixels, in two launches."""
    q = 300
    rng = cp.random.RandomState(211)
    raw = rng.poisson(0.7, (512, q, q)).astype(cp.uint16)
    raw[:, 5, :40] = 65535
    valid = np.ones((q, q), bool)
    valid[7, 9] = False
    source = PairedCounts((1, 512, q, q), np.uint16, valid)
    source.append(raw)
    session = detector.prepare(source)
    working = raw * cp.asarray(valid, dtype=cp.uint16)
    rows, cols = np.indices((q, q))
    for weights in (cols, rows + 1):
        expected = (working.astype(cp.uint64) * cp.asarray(weights, cp.uint64)).sum(axis=(1, 2))
        np.testing.assert_array_equal(session.weighted_sum_exact(weights).reshape(-1), expected.get())
    mask = np.random.default_rng(5).random((q, q)) < 0.75
    mask[::2, ::2] = ~mask[::2, ::2]
    expected = working[:, cp.asarray(mask)].sum(axis=1, dtype=cp.uint64)
    np.testing.assert_array_equal(session.masked_sum_exact(mask).reshape(-1), expected.get())
    total = np.maximum(working.sum(axis=(1, 2), dtype=cp.uint64).get().astype(np.float64), 1.0)
    for center, weights in zip(session.center_of_mass(), (rows, cols)):
        moment = (working.astype(cp.uint64) * cp.asarray(weights, cp.uint64)).sum(axis=(1, 2)).get()
        np.testing.assert_array_equal(center.reshape(-1), (moment.astype(np.float64) / total).astype(np.float32))


def test_malformed_header_and_extent_are_rejected():
    raw, valid = _synthetic(19, 1024)
    source = PairedCounts((1, 1024, 19, 19), np.uint16, valid)
    source.append(raw)
    payload, records, models = source.chunks[0].arrays[:3]
    stream = next(i for i, m in enumerate(models.get()) if 64 <= m < 96 and 0 < i % 32 < 30)
    group, lane = divmod(stream, 32)
    begin = int(records[group * 17].get()) + int(records.view(cp.uint16)[group * 34 + 2 + lane].get())
    session = detector.prepare([source])
    mask = np.zeros((19, 19), bool)
    mask.ravel()[stream % 361] = True
    header = payload[begin : begin + 2].copy()
    payload[begin + 1] = int(header[1].get()) | 0x08  # reserved bit
    with pytest.raises(ValueError, match="exact decoding"):
        session.masked_sum(mask, output="native")
    payload[begin : begin + 2] = header
    saved = records.view(cp.uint16)[group * 34 + 2 + lane + 1].copy()
    records.view(cp.uint16)[group * 34 + 2 + lane + 1] = begin - int(records[group * 17].get()) + 1
    with pytest.raises(ValueError, match="exact decoding"):
        session.masked_sum(mask, output="native")
    records.view(cp.uint16)[group * 34 + 2 + lane + 1] = saved
    expected = raw[:, cp.asarray(mask & valid)].sum(axis=1, dtype=cp.uint64)
    assert bool(cp.array_equal(session.masked_sum(mask, output="native").reshape(-1), expected).get())


def test_saved_form_reopens_byte_identical(tmp_path):
    raw, valid = _synthetic(24, 512)
    source = PairedCounts((1, 512, 24, 24), np.uint16, valid)
    source.append(raw)
    written = source.save(tmp_path / "counts.paired")
    reopened = PairedCounts.load(tmp_path / "counts.paired")
    assert written["bytes"] == (tmp_path / "counts.paired").stat().st_size
    for before, after in zip(source.chunks[0].arrays, reopened.chunks[0].arrays):
        assert bool(cp.array_equal(before, after).get())
    assert bool(cp.array_equal(reopened.decode_scan_range_device(0, reopened.chunks[0].scans), raw).get())


def test_saved_forms_open_into_a_caller_block_with_one_shared_reader(tmp_path):
    """An application places a series of saved forms in one device block it reserved, with one reader."""
    from quantem.gpu.io.paired import load_paired_file
    from quantem.gpu.resident.cuda.paired import ResidentFileReader

    raws, paths = [], []
    for index in range(2):
        raw, valid = _synthetic(24, 512, seed=200 + index)
        source = PairedCounts((1, 512, 24, 24), np.uint16, valid)
        source.append(raw)
        paths.append(tmp_path / f"counts-{index}.paired")
        source.save(paths[-1])
        raws.append(raw)
    block = cp.empty(64 << 20, cp.uint8)
    cursor = 0

    def allocate(shape, dtype):
        nonlocal cursor
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        view = block[cursor : cursor + nbytes].view(dtype).reshape(shape)
        cursor += (nbytes + 511) // 512 * 512
        return view

    with ResidentFileReader() as reader:
        loaded = [load_paired_file(path, device=None, verbose=False, reader=reader, allocate=allocate) for path in paths]
        assert not reader.closed
    for item, raw in zip(loaded, raws):
        for array in item.data.chunks[0].arrays:
            assert block.data.ptr <= array.data.ptr and array.data.ptr + array.nbytes <= block.data.ptr + block.nbytes
        assert bool(cp.array_equal(item.data.decode_scan_range_device(0, item.data.chunks[0].scans), raw).get())


@pytest.mark.skipif(not os.environ.get("QUANTEM_GPU_PAIRED_REFERENCE"), reason="set QUANTEM_GPU_PAIRED_REFERENCE to a JSON reference on a native H5 source")
def test_native_source_matches_frozen_virtual_images():
    """Frozen full-array SHA256 references from an independent raw-count reduction.

    The JSON holds ``path`` (native H5 master), ``poses`` (row, col, inner, outer) and
    ``reference_VI_sha256`` in pose order; see docs/developer/paired-resident.md.
    """
    reference = json.loads(Path(os.environ["QUANTEM_GPU_PAIRED_REFERENCE"]).read_text())
    from quantem.gpu import io

    source = io.load(reference["path"], backend="cuda", representation="paired",
                     hot_pixel_correction="none",
                     scan_shape=tuple(reference.get("scan_shape", (512, 512)))).data
    session = detector.prepare([source])
    shape = tuple(source.shape[2:])
    for pose, digest in zip(reference["poses"], reference["reference_VI_sha256"]):
        mask = detector.detector_mask(pose[:2], pose[2], pose[3], shape, dtype=np.float64)
        image = session.masked_sum(mask, output="native")[0].get()
        assert hashlib.sha256(image.tobytes()).hexdigest() == digest


def test_decode_blocks_match_chunk_decodes():
    """Every 512-scan block of a two-chunk source decodes to its exact counts."""
    raw, valid = _synthetic(20, 1024, seed=5)
    source = PairedCounts((2, 512, 20, 20), np.uint16, valid)
    source.append(raw[:512])
    source.append(raw[512:])
    for first in (0, 512):
        assert bool(cp.array_equal(source.decode_blocks(first, 512), raw[first : first + 512]).get())


def test_mixed_chunk_sizes_beyond_the_grid_y_limit():
    """A series whose tail was appended in 512-scan pieces still answers masks exactly."""
    raw, valid = _synthetic(17, 512)
    sources = []
    for _ in range(2):
        source = PairedCounts((1, 512 * 2, 17, 17), np.uint16, valid)
        source.append(cp.concatenate([raw, raw]))  # one 1024-scan chunk
        sources.append(source)
    small = PairedCounts((1, 512 * 2, 17, 17), np.uint16, valid)
    small.append(raw)
    small.append(raw)  # two 512-scan chunks: work list is chunks x max_blocks
    sources.append(small)
    session = detector.prepare(sources)
    assert session._backend.chunk_count * 2 == 8
    mask = np.zeros((17, 17), bool)
    mask[3:9, 4:12] = True
    expected = cp.concatenate([raw, raw])[:, cp.asarray(mask & valid)].sum(axis=1, dtype=cp.uint64)
    got = session.masked_sum(mask, output="native").reshape(3, -1)
    assert all(bool(cp.array_equal(got[i], expected).get()) for i in range(3))


def test_non_blocking_queries_finish_in_order_with_exact_results():
    """Two masks queued back to back give the same sums as blocking calls, and finish() reports each."""
    raw, valid = _synthetic(19, 1024)
    source = PairedCounts((1, 1024, 19, 19), np.uint16, valid)
    source.append(raw)
    session = detector.prepare([source])
    masks = [detector.detector_mask((9.5, 9.5), 0., 5., (19, 19), dtype=np.float64),
             detector.detector_mask((9.25, 9.75), 3., 8., (19, 19), dtype=np.float64)]
    expected = [session.masked_sum(mask, output="native").copy() for mask in masks]
    outs = [cp.empty_like(expected[0]) for _ in masks]
    for mask, out in zip(masks, outs):
        session.masked_sum(mask, output="native", out=out, wait=False)
    timings = [session.finish() for _ in masks]
    assert all(t["gpu_ms"] > 0 and "residual_pixels" in t for t in timings)
    assert all(bool(cp.array_equal(out, want).get()) for out, want in zip(outs, expected))
    frame = session.frame(700, output="native", wait=False)
    session.finish()
    assert bool(cp.array_equal(frame.reshape(19, 19)[cp.asarray(valid)], raw[700][cp.asarray(valid)]).get())


@pytest.mark.parametrize("q,scans,stride", [(17, 2048, 2), (19, 3072, 3), (17, 2048, 8)])
def test_block_stride_sums_every_kth_block_exactly_and_resets_the_baseline(q, scans, stride):
    """A strided sum writes exact values on every stride-th 512-scan block and nothing else."""
    raw, valid = _synthetic(q, scans)
    source = PairedCounts((1, scans, q, q), np.uint16, valid)
    source.append(raw)
    session = detector.prepare([source])
    masks = [detector.detector_mask((q * 0.5 + 0.125, q * 0.5 + 0.375), 2.25, q * 0.45, (q, q), dtype=np.float64),
             detector.detector_mask((q * 0.5 - 1.5, q * 0.5 + 2.0), 1.0, q * 0.4, (q, q), dtype=np.float64)]
    exact = session.masked_sum(masks[0], output="native")   # establishes the incremental baseline
    out = cp.full((1, 1, scans), 7, cp.uint32)
    strided = session.masked_sum(masks[1], output="native", out=out, block_stride=stride)
    expected = raw[:, cp.asarray(masks[1] & valid)].sum(axis=1, dtype=cp.uint64).reshape(1, 1, scans)
    blocks = -(-scans // 512)
    for block in range(blocks):
        rows = slice(block * 512, (block + 1) * 512)
        if block % stride == 0:
            assert bool(cp.array_equal(strided[0, 0, rows], expected[0, 0, rows]).get())
        else:
            assert int(strided[0, 0, rows].min().get()) == 7 and int(strided[0, 0, rows].max().get()) == 7
    assert session._backend.last["block_stride"] == stride and not session._backend.last["incremental"]
    # A second query at the same stride builds on the first incrementally and stays exact.
    nudged = detector.detector_mask((q * 0.5 - 1.25, q * 0.5 + 2.25), 1.0, q * 0.4, (q, q), dtype=np.float64)
    session.masked_sum(nudged, output="native", out=out, block_stride=stride)
    assert session._backend.last["incremental"]
    nudged_expected = raw[:, cp.asarray(nudged & valid)].sum(axis=1, dtype=cp.uint64).reshape(1, 1, scans)
    for block in range(blocks):
        rows = slice(block * 512, (block + 1) * 512)
        if block % stride == 0:
            assert bool(cp.array_equal(out[0, 0, rows], nudged_expected[0, 0, rows]).get())
    # The next ordinary query starts from a fresh full plan and is exact everywhere.
    following = session.masked_sum(masks[1], output="native")
    assert not session._backend.last["incremental"]
    assert bool(cp.array_equal(following.reshape(1, 1, scans), expected).get())
    # And incremental queries work again afterwards.
    again = session.masked_sum(masks[0], output="native")
    assert session._backend.last["incremental"]
    assert bool(cp.array_equal(again, exact).get())
    with pytest.raises(ValueError):
        session.masked_sum(masks[0], output="numpy", block_stride=2)
