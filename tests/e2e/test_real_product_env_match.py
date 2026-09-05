from __future__ import annotations

import os

import numpy as np
import pytest


def _parse_region() -> tuple[int, int, int, int]:
    text = os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_REGION", "0,64,0,64")
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 4:
        raise ValueError(
            "QUANTEM_GPU_PRODUCT_MATCH_REGION must be r0,r1,c0,c1"
        )
    return values


def _requested_backends() -> tuple[str, ...]:
    text = os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_BACKENDS", "cuda")
    return tuple(part.strip().lower() for part in text.split(",") if part.strip())


def _to_numpy(data) -> np.ndarray:
    if type(data).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(data)
    return np.asarray(data)


def _require_cuda_vram(min_gb: float = 3.0) -> None:
    cp = pytest.importorskip("cupy")
    try:
        free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")
    if free_gb < float(min_gb):
        pytest.skip(
            f"not enough free CUDA memory for real-data agreement "
            f"({free_gb:.1f} GB free, need {min_gb:.1f} GB)"
        )


def _detector_mask(
    det_shape: tuple[int, int],
    lo_px: float,
    hi_px: float,
) -> np.ndarray:
    row = np.arange(det_shape[0], dtype=np.float32)[:, None]
    col = np.arange(det_shape[1], dtype=np.float32)[None, :]
    center = ((det_shape[0] - 1) / 2.0, (det_shape[1] - 1) / 2.0)
    dist = np.sqrt((row - center[0]) ** 2 + (col - center[1]) ** 2)
    return (dist >= float(lo_px)) & (dist <= float(hi_px))


def _masked_sum_reference(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scan_shape = tuple(int(v) for v in data.shape[:2])
    flat = data.reshape(-1, int(np.prod(data.shape[-2:])))
    selected = mask.reshape(-1)
    out = np.empty(flat.shape[0], dtype=np.float32)
    chunk = int(os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_REF_CHUNK", "4096"))
    for start in range(0, flat.shape[0], max(1, chunk)):
        stop = min(flat.shape[0], start + max(1, chunk))
        out[start:stop] = (
            flat[start:stop, selected]
            .sum(axis=1, dtype=np.uint64)
            .astype(np.float32)
        )
    return out.reshape(scan_shape)


def _center_of_mass_reference(
    data: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scan_shape = tuple(int(v) for v in data.shape[:2])
    rows = np.arange(data.shape[-2], dtype=np.float64)[:, None]
    cols = np.arange(data.shape[-1], dtype=np.float64)[None, :]
    mask64 = None if mask is None else mask.astype(np.float64, copy=False)
    flat = data.reshape(-1, *data.shape[-2:])
    com_row = np.empty(flat.shape[0], dtype=np.float32)
    com_col = np.empty(flat.shape[0], dtype=np.float32)
    chunk = int(os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_REF_CHUNK", "4096"))
    for start in range(0, flat.shape[0], max(1, chunk)):
        stop = min(flat.shape[0], start + max(1, chunk))
        block = flat[start:stop].astype(np.float64, copy=False)
        if mask64 is not None:
            block = block * mask64
        denom = np.maximum(block.sum(axis=(1, 2)), 1e-10)
        com_row[start:stop] = ((block * rows).sum(axis=(1, 2)) / denom).astype(
            np.float32
        )
        com_col[start:stop] = ((block * cols).sum(axis=(1, 2)) / denom).astype(
            np.float32
        )
    com_row = (com_row - float(com_row.mean())).reshape(scan_shape)
    com_col = (com_col - float(com_col.mean())).reshape(scan_shape)
    return com_row, com_col


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_MASTER"),
    reason="set QUANTEM_GPU_PRODUCT_MATCH_MASTER for real-data product agreement",
)
@pytest.mark.parametrize("backend", _requested_backends())
def test_real_crop_products_match_independent_reference(backend: str) -> None:
    """Real-data BF/ADF/DF/CoM agreement for a user-selected crop.

    The master path is intentionally supplied by environment variable so public
    test code does not publish local fixture names. Use
    QUANTEM_GPU_PRODUCT_MATCH_REGION to scale from a smoke crop to a full
    signoff region.
    """
    from quantem.gpu.detector import masked_sum
    from quantem.gpu.dpc import center_of_mass
    from quantem.gpu.io import load

    master = os.environ["QUANTEM_GPU_PRODUCT_MATCH_MASTER"]
    if not os.path.exists(master):
        pytest.skip("real-data agreement master is not available on this host")
    if backend == "cuda":
        _require_cuda_vram()

    kwargs = {
        "backend": backend,
        "scan_region": _parse_region(),
        "verbose": False,
        "det_bin": int(os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_DET_BIN", "1")),
    }
    result = load(master, **kwargs)
    data_np = _to_numpy(result.data)
    det_shape = tuple(int(v) for v in data_np.shape[-2:])
    radius = min(
        float(os.environ.get("QUANTEM_GPU_PRODUCT_MATCH_BF_RADIUS", "30")),
        min(det_shape) / 2,
    )

    masks = {
        "bf": _detector_mask(det_shape, 0.0, radius),
        "adf": _detector_mask(det_shape, radius, radius * 2.0),
        "df": _detector_mask(det_shape, radius, np.inf),
    }

    for mask in masks.values():
        got = masked_sum(result.data, mask)
        expected = _masked_sum_reference(data_np, mask)
        np.testing.assert_array_equal(got, expected)

    got_row, got_col = center_of_mass(result.data)
    ref_row, ref_col = _center_of_mass_reference(data_np)
    np.testing.assert_allclose(got_row, ref_row, rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_col, ref_col, rtol=0, atol=1e-5)

    got_row, got_col = center_of_mass(result.data, mask=masks["bf"])
    ref_row, ref_col = _center_of_mass_reference(data_np, masks["bf"])
    np.testing.assert_allclose(got_row, ref_row, rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_col, ref_col, rtol=0, atol=1e-5)
