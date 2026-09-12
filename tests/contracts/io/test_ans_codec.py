"""Exact native-count save, reference decode and resident query workflows."""

import os

import numpy as np
import pytest

from quantem.gpu.io import load, save
from quantem.gpu.io._ans import ANSFile, write_ans_reference


def _counts(dtype):
    counts = np.random.default_rng(7).poisson(3, (5, 61, 3, 4)).astype(dtype)
    counts[:, :, 0, 0] = 0
    counts[:, :, 2, 3] = np.iinfo(dtype).max
    counts[1, :, 1, 1] = np.arange(61, dtype=dtype)
    return counts


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_native_counts_roundtrip_with_tail_and_sentinels(tmp_path, dtype):
    counts = _counts(dtype)
    result = save(
        tmp_path / "counts.ans", counts, format="quantem", compression="ans", backend="cpu",
        batch_size=128,
        metadata={"detector_mask_policy": "preserve-stored-counts"},
    )
    with ANSFile(result.path) as source:
        decoded = np.concatenate([
            source.decode_block_reference(block) for block in range(3)
        ]).reshape(counts.shape)
        np.testing.assert_array_equal(decoded, counts)
        assert decoded.dtype == counts.dtype
        assert np.any(source.arrays["literal"] == 0)
        assert np.any(source.arrays["literal"] == 1)
    np.testing.assert_array_equal(load(result.path, backend="cpu", representation="dense").data, counts)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cuda_saved_counts_patterns_masks_and_packed_roundtrip(tmp_path, dtype):
    if os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1":
        pytest.skip("Set QUANTEM_CUDA_ANS_TEST=1 in an owned GPU test window.")
    cp = pytest.importorskip("cupy")

    counts = _counts(dtype)
    path = write_ans_reference(tmp_path / "counts.ans", counts, block_frames=128)
    with cp.cuda.Device(0):
        source = load(path, backend="cuda", representation="encoded", device=0).data
        try:
            blocks = [source.decode_block_device(i).get() for i in range(3)]
            np.testing.assert_array_equal(
                np.concatenate(blocks).reshape(counts.shape), counts,
            )
            positions = np.asarray([[4, 60], [0, 0], [4, 60]], np.int64)
            np.testing.assert_array_equal(
                source.gather_diffraction_device(positions).get(),
                counts[positions[:, 0], positions[:, 1]],
            )
            for mask in [np.ones((3, 4), np.uint8), np.eye(3, 4, dtype=np.uint8)]:
                np.testing.assert_array_equal(
                    source.detector_sum_device(mask).get(),
                    (counts.astype(np.uint64) * mask).sum((2, 3)),
                )
            packed = source.to_packed()
            try:
                np.testing.assert_array_equal(
                    packed.gather_diffraction_device(positions).get(),
                    counts[positions[:, 0], positions[:, 1]],
                )
            finally:
                packed.release()
        finally:
            source.release()


def test_legacy_rans_frozen_detector_products_and_raw_pattern():
    """The retained seven-tilt representation uses the canonical CUDA decoder."""
    import json
    from pathlib import Path

    build_path = os.environ.get("QUANTEM_LEGACY_RANS_BUILD")
    product_root = os.environ.get("QUANTEM_LEGACY_RANS_PRODUCTS")
    if (os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1"
            or not build_path or not product_root):
        pytest.skip("Set the CUDA gate and both legacy evidence paths for real parity.")
    cp = pytest.importorskip("cupy")
    from quantem.gpu.io._ans_legacy import _legacy_rans_arguments
    from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts

    records = json.loads(Path(build_path).read_text())["tilts"]
    for ordinal, record in enumerate(records):
        root = Path(product_root) / f"tilt-{ordinal + 1}" / "products"
        manifest = json.loads((root / "exact-product-manifest.json").read_text())
        artifacts = manifest["artifacts"]
        arguments = _legacy_rans_arguments(record)
        source = None
        try:
            with cp.cuda.Device(0):
                source = CudaANSResidentCounts(**arguments)
                for name in ("bf", "adf", "abf", "custom"):
                    mask = np.load(root / artifacts[f"{name}_mask_u8"]["path"])
                    expected = np.load(root / artifacts[f"{name}_plane_u32"]["path"])
                    np.testing.assert_array_equal(
                        source.detector_sum_device(mask).get(), expected,
                    )
                position = manifest["product_definition"]["selected_scan"]
                np.testing.assert_array_equal(
                    source.extract_diffraction_device(*position).get(),
                    np.load(root / artifacts["selected_dp_raw_u16"]["path"]),
                )
        finally:
            if source is not None:
                with cp.cuda.Device(0):
                    source.release()
            arguments["payload"]._mmap.close()
