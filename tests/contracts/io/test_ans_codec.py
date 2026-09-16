"""Retained legacy rANS parity against frozen detector products."""

import os

import numpy as np
import pytest


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
