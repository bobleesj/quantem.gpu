"""Metal virtual-image results belong to the caller, and repeated queries hold memory flat.

``MetalVirtualImage`` once handed the caller its reused output buffer (and the
backend its cached detector totals), so the next query changed an earlier
result. PyObjC never frees a Metal buffer when its wrapper is collected, so
per-query allocations and the image's own buffers also stayed allocated until
the process exited.
"""

import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="needs an Apple GPU",
)


def _chunks(values: np.ndarray, frames_per_chunk: int):
    """Copy ``(frames, row, col)`` counts into test-owned Metal chunks."""
    from quantem.gpu.device import metal_runtime

    chunks = []
    for first in range(0, len(values), frames_per_chunk):
        part = values[first : first + frames_per_chunk]
        buffer = metal_runtime.allocate_shared(part.nbytes, "test frames")
        chunk = metal_runtime.shared_array(buffer, part.dtype, part.shape)
        chunk[:] = part
        chunks.append(chunk)
    return chunks


def _release(chunks) -> None:
    from quantem.gpu.device import metal_runtime

    for chunk in chunks:
        metal_runtime.release_buffer(chunk._mtl)


def _masks(shape):
    rows, cols = np.indices(shape)
    disk = (rows - shape[0] / 2) ** 2 + (cols - shape[1] / 2) ** 2 <= (shape[0] / 4) ** 2
    # More than 512 row spans: the dense-mask kernel instead of the span kernel.
    noisy = np.random.default_rng(3).random(shape) < 0.5
    return disk, noisy


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_results_do_not_change_on_the_next_query(dtype):
    from quantem.gpu import detector
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    values = np.random.default_rng(7).integers(0, 1000, (100, 40, 48)).astype(dtype)
    chunks = _chunks(values, 70)
    try:
        frames = ChunkedFrames(chunks)
        disk, noisy = _masks((40, 48))
        for first_mask, second_mask in ((disk, noisy), (noisy, disk)):
            first = frames.vi.masked_sum(first_mask)
            expected = first.copy()
            frames.vi.masked_sum(second_mask)
            np.testing.assert_array_equal(first, expected)
        total = frames.vi.sum_frames([1, 2, 80])
        expected = total.copy()
        frames.vi.sum_frames([5, 6])
        np.testing.assert_array_equal(total, expected)

        session = detector.prepare(frames)
        full = np.ones((40, 48), dtype=bool)
        total = session.masked_sum_exact(full)
        expected = total.copy()
        total[...] = 0
        np.testing.assert_array_equal(session.masked_sum_exact(full), expected)
        image = session.masked_sum_exact(disk)
        expected = image.copy()
        session.masked_sum_exact(~disk)
        np.testing.assert_array_equal(image, expected)
        pattern = session.reduce_frames([1, 2, 3], "mean")
        expected = pattern.copy()
        session.reduce_frames([4, 5], "mean")
        np.testing.assert_array_equal(pattern, expected)
        np.testing.assert_array_equal(
            session.masked_sum_exact(disk).reshape(-1),
            values[:, disk].sum(axis=1, dtype=np.uint64),
        )
    finally:
        _release(chunks)


def test_repeated_queries_keep_metal_memory_flat():
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    values = np.random.default_rng(11).integers(0, 1000, (300, 64, 64)).astype(np.uint16)
    chunks = _chunks(values, 120)
    disk, noisy = _masks((64, 64))
    selections = [np.arange(count) for count in (10, 200, 300)]

    def round_of_queries():
        frames = ChunkedFrames(chunks)
        frames.ensure_fast_interaction(verbose=False)
        for vi in (frames.vi, frames.fast_vi):
            vi.masked_sum(disk[: vi.det[0], : vi.det[1]])
            vi.masked_sum(noisy[: vi.det[0], : vi.det[1]])
            vi.detector_sum_exact()
            vi.center_of_mass()
            for selection in selections:
                vi.sum_frames(selection)

    try:
        round_of_queries()
        gc.collect()
        before = int(torch.mps.driver_allocated_memory())
        for _ in range(100):
            round_of_queries()
        gc.collect()
        # Memory must not grow; an earlier test's buffer freed during the loop may lower it.
        assert int(torch.mps.driver_allocated_memory()) <= before
    finally:
        _release(chunks)
