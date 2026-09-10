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
from quantem.gpu._compact.paired import PairedCounts, PairedSeriesCompute


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


@pytest.mark.parametrize("q,scans", [(17, 512), (19, 1024), (257, 512)])
def test_masks_and_frames_match_direct_sums(q, scans):
    raw, valid = _synthetic(q, scans)
    source = PairedCounts((1, scans, q, q), np.uint16, valid)
    source.append(raw)
    assert bool(cp.array_equal(source.decode_chunk(0), raw).get())
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
    assert bool(cp.array_equal(reopened.decode_chunk(0), raw).get())


@pytest.mark.skipif(not os.environ.get("QUANTEM_GPU_PAIRED_REFERENCE"), reason="set QUANTEM_GPU_PAIRED_REFERENCE to a JSON reference on a native H5 source")
def test_native_source_matches_frozen_virtual_images():
    """Frozen full-array SHA256 references from an independent raw-count reduction.

    The JSON holds ``path`` (native H5 master), ``poses`` (row, col, inner, outer) and
    ``reference_VI_sha256`` in pose order; see docs/developer/paired-resident.md.
    """
    reference = json.loads(Path(os.environ["QUANTEM_GPU_PAIRED_REFERENCE"]).read_text())
    from quantem.gpu import io

    source = io.load(reference["path"], backend="cuda", representation="paired", scan_shape=tuple(reference.get("scan_shape", (512, 512)))).data
    session = detector.prepare([source])
    shape = tuple(source.shape[2:])
    for pose, digest in zip(reference["poses"], reference["reference_VI_sha256"]):
        mask = detector.detector_mask(pose[:2], pose[2], pose[3], shape, dtype=np.float64)
        image = session.masked_sum(mask, output="native")[0].get()
        assert hashlib.sha256(image.tobytes()).hexdigest() == digest


def test_feed_blocks_match_chunk_decodes():
    """A time-series consumer sees every 512-scan block of every source, exact, with sqrt amplitudes."""
    from quantem.gpu._compact.paired import PairedFeed

    sources, raws = [], []
    for seed in (5, 6):
        raw, valid = _synthetic(20, 1024, seed=seed)
        source = PairedCounts((2, 512, 20, 20), np.uint16, valid)
        source.append(raw[:512])
        source.append(raw[512:])
        sources.append(source)
        raws.append(raw)
    assert bool(cp.array_equal(sources[0].decode_blocks(512, 512), raws[0][512:]).get())
    seen = []
    for block in PairedFeed(sources, amplitude=True):
        expected = raws[block.source][block.first : block.first + block.scans]
        assert bool(cp.array_equal(block.raw, expected).get())
        assert bool(cp.allclose(block.amplitude, cp.sqrt(expected.astype(cp.float32))).get())
        seen.append((block.source, block.first))
    assert seen == [(0, 0), (1, 0), (0, 512), (1, 512)]
    assert [b.first for b in PairedFeed(sources[:1], block_scans=1024, order="source")] == [0, 512]


def test_mixed_chunk_sizes_beyond_the_grid_y_limit():
    """A series whose tail was appended in 512-scan pieces still answers masks exactly."""
    raw, valid = _synthetic(17, 512)
    sources = []
    for n in range(2):
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
