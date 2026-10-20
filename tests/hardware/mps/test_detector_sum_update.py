"""A dragged detector mask updates the previous exact image on Metal and equals the full sum."""

import numpy as np
import pytest


def _disk(shape, row, col, outer, inner=0.0):
    rows, cols = np.indices(shape)
    distance = np.hypot(rows - row, cols - col)
    return (distance <= outer) & (distance >= inner)


def _resident(counts, valid, scans_loaded=None):
    from quantem.gpu.device.metal_runtime import release_buffer, shared_array, upload_shared
    from quantem.gpu.resident.mps.counts import MPSStreamedCounts
    from quantem.gpu.resident.mps.spatial import build_index

    resident = MPSStreamedCounts(counts.shape, counts.dtype, valid)
    flat = counts.reshape(-1, *counts.shape[2:])
    stop = flat.shape[0] if scans_loaded is None else scans_loaded
    for first in range(0, stop, 512):
        raw = np.ascontiguousarray(flat[first : min(first + 512, stop)])
        buffer = upload_shared(raw, "test native counts")
        try:
            resident.append(shared_array(buffer, raw.dtype, raw.shape))
            resident.spatial_chunks.append(build_index(resident, buffer, len(raw)))
        finally:
            release_buffer(buffer)
    return resident


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dragged_masks_equal_full_sums(dtype, monkeypatch):
    pytest.importorskip("Metal")
    from quantem.gpu import detector
    from quantem.gpu.resident.mps.counts import MPSStreamedCounts

    rng = np.random.default_rng(20261020)
    shape = (23, 47, 40, 44)
    counts = rng.poisson(2.0, shape).astype(dtype)
    counts[:, :, 20, 22] = np.iinfo(dtype).max
    counts[3, 5] = np.iinfo(dtype).max
    valid = rng.random(shape[2:]) > 0.05
    resident = _resident(counts, valid)
    updates = []
    original = MPSStreamedCounts.detector_sum_update_device

    def spy(self, previous, pixels, signs):
        updates.append(len(pixels))
        return original(self, previous, pixels, signs)

    monkeypatch.setattr(MPSStreamedCounts, "detector_sum_update_device", spy)
    try:
        session = detector.prepare(resident)
        masks = [_disk(shape[2:], 20 + 0.5 * step, 22, 11) for step in range(6)]
        masks += [_disk(shape[2:], 22, 22, 11 + 0.7 * step) for step in range(4)]
        masks += [_disk(shape[2:], 20, 22, 18, 9), _disk(shape[2:], 20.5, 22.5, 18, 9), _disk(shape[2:], 20.5, 22.5, 18, 9)]
        masks += [rng.random(shape[2:]) > 0.5]  # scattered: more changed pixels than an update decodes
        masks += [np.zeros(shape[2:], bool), np.ones(shape[2:], np.uint8)]
        for mask in masks:
            exact = (counts * (mask.astype(bool) & valid)).sum(axis=(2, 3), dtype=np.uint64)
            result = session.masked_sum_exact(mask)
            np.testing.assert_array_equal(result, exact)
            np.testing.assert_array_equal(session.masked_sum(mask), exact.astype(np.float32))
            result[...] = 0  # the caller's copy is its own; the kept image must not change
        # every small step was an update, the scattered and full masks were not
        assert len(updates) >= 10
        assert 0 in updates
        assert max(updates) <= 768
        with pytest.raises(ValueError, match="binary"):
            session.masked_sum_exact(np.full(shape[2:], 2))
    finally:
        resident.release()


def test_update_waits_for_a_complete_source(monkeypatch):
    """Scans still arriving take the full sum: an image kept before they arrived would be stale."""
    pytest.importorskip("Metal")
    from quantem.gpu import detector
    from quantem.gpu.resident.mps.counts import MPSStreamedCounts

    rng = np.random.default_rng(7)
    shape = (32, 32, 24, 24)
    counts = rng.poisson(1.0, shape).astype(np.uint16)
    valid = np.ones(shape[2:], bool)
    resident = _resident(counts, valid, scans_loaded=512)
    calls = []
    monkeypatch.setattr(MPSStreamedCounts, "detector_sum_update_device",
                        lambda self, *args: calls.append(1) or (_ for _ in ()).throw(AssertionError("update on a partial source")))
    try:
        session = detector.prepare(resident)
        for step in range(3):
            mask = _disk(shape[2:], 12 + 0.5 * step, 12, 6)
            loaded = (counts * mask).sum(axis=(2, 3), dtype=np.uint64).reshape(-1)[:512]
            np.testing.assert_array_equal(session.masked_sum_exact(mask).reshape(-1)[:512], loaded)
        assert not calls
    finally:
        resident.release()
