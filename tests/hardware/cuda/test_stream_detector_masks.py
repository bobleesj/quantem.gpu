"""Fixed detectors follow consecutive batches without expanding their counts."""

import cupy as cp
import numpy as np
import pytest

from quantem.gpu.detector.cuda.dense import DetectorMasks


@pytest.mark.parametrize("shape", [(96, 96), (192, 192)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_detector_masks_keep_exact_counts_across_batches(shape, dtype):
    masks = np.zeros((3, *shape), bool)
    masks[0] = True
    masks[1, ::3, 1::4] = True
    detectors = DetectorMasks(masks)
    for count in (0, 1, 17):
        original = np.arange(count * np.prod(shape), dtype=np.uint64).reshape(count, *shape)
        original = (original % 23).astype(dtype)
        original[:, ::2, ::2] = np.iinfo(dtype).max
        patterns = cp.asarray(original)
        expected = np.stack([
            original[:, mask].sum(axis=1, dtype=np.uint64) for mask in masks
        ], axis=1).astype(np.float32)
        actual = detectors.integrate(patterns)
        np.testing.assert_array_equal(actual.get(), expected)
        np.testing.assert_array_equal(patterns.get(), original)


def test_detector_masks_cross_stream_matches_existing_reduction():
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    masks = np.ones((2, 96, 96), bool)
    masks[1, 8:] = False
    with cp.cuda.Stream(non_blocking=True):
        detectors = DetectorMasks(masks)
    with cp.cuda.Stream(non_blocking=True) as stream:
        patterns = (cp.arange(17 * 96 * 96).reshape(17, 96, 96) % 11).astype(cp.uint16)
        actual = detectors.integrate(patterns)
        stream.synchronize()
    reference = CudaKernelCompute(patterns)
    expected = np.stack([reference.masked_sum(mask).reshape(-1) for mask in masks], axis=1)
    np.testing.assert_array_equal(actual.get(), expected)
