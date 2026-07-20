from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

from quantem.gpu.image.dft_upsample import (
    cross_correlation_shift_cp,
    dft_upsample_cp,
)
from quantem.gpu.parallax_utils import (
    _bin_mapping_only,
    synchronize_shifts_cp,
)
from quantem.gpu.ssb.optics.aberration_fitting import fit_aberrations_svd_polar


QUANTEM_SRC_ENV = "QUANTEM_ORIGINAL_SRC"


def _quantem_src() -> Path:
    raw_path = os.environ.get(QUANTEM_SRC_ENV)
    if raw_path:
        return Path(raw_path).expanduser()
    return Path(__file__).resolve().parents[2] / "quantem" / "src"


def _load_quantem_module(name: str, relpath: str):
    """Load original QuantEM reference modules without importing its top-level package."""
    path = _quantem_src() / relpath
    if not path.exists():
        pytest.skip(f"original QuantEM source is not available; set {QUANTEM_SRC_ENV}.")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not load original QuantEM module; check {QUANTEM_SRC_ENV}.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reference_modules():
    """Return original QuantEM helpers used as migration parity references."""
    if "quantem.core.config" not in sys.modules:
        _load_quantem_module("quantem.core.config", "quantem/core/config.py")
    if "quantem.core.utils.utils" not in sys.modules:
        _load_quantem_module("quantem.core.utils.utils", "quantem/core/utils/utils.py")
    if "quantem.core.utils.imaging_utils" not in sys.modules:
        _load_quantem_module(
            "quantem.core.utils.imaging_utils",
            "quantem/core/utils/imaging_utils.py",
        )
    if "quantem.diffractive_imaging.complex_probe" not in sys.modules:
        _load_quantem_module(
            "quantem.diffractive_imaging.complex_probe",
            "quantem/diffractive_imaging/complex_probe.py",
        )
    dpu = _load_quantem_module(
        "quantem.diffractive_imaging.direct_ptycho_utils",
        "quantem/diffractive_imaging/direct_ptycho_utils.py",
    )
    imaging = sys.modules["quantem.core.utils.imaging_utils"]
    return imaging, dpu


def _make_test_image(seed: int = 0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _fourier_shift_image(image: np.ndarray, shift_row: float, shift_col: float) -> np.ndarray:
    n_row, n_col = image.shape
    row_freq = np.fft.ifftshift(np.arange(n_row)) - math.floor(n_row / 2)
    col_freq = np.fft.ifftshift(np.arange(n_col)) - math.floor(n_col / 2)
    phase = np.exp(
        -2j
        * math.pi
        * (shift_row * row_freq[:, None] / n_row + shift_col * col_freq[None, :] / n_col)
    )
    return np.fft.ifft2(np.fft.fft2(image) * phase).real.astype(np.float32)


def _make_circular_bf_mask(size_row: int, size_col: int, bf_radius: int) -> np.ndarray:
    center_row, center_col = size_row // 2, size_col // 2
    rows = np.fft.ifftshift(np.arange(size_row) - center_row)
    cols = np.fft.ifftshift(np.arange(size_col) - center_col)
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")
    return grid_row**2 + grid_col**2 <= bf_radius**2


def test_dft_upsample_matches_original_quantem() -> None:
    imaging, _ = _reference_modules()
    rng = np.random.default_rng(42)
    real = rng.standard_normal((64, 64)).astype(np.float32)
    imag = rng.standard_normal((64, 64)).astype(np.float32)
    data_np = real + 1j * imag
    upsample_factor = 10
    xy_shift_np = np.array([3.5, -2.0])

    ref = imaging.dftUpsample_torch(
        torch.from_numpy(data_np),
        upsample_factor,
        torch.tensor(xy_shift_np, dtype=torch.float32),
    ).numpy()

    out = dft_upsample_cp(cp.asarray(data_np), upsample_factor, cp.asarray(xy_shift_np))

    np.testing.assert_allclose(cp.asnumpy(out), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("shift_row", "shift_col"),
    [(3.7, -2.3), (0.0, 0.0), (-5.1, 4.8), (0.25, -0.75)],
)
def test_cross_correlation_shift_matches_original_quantem(
    shift_row: float,
    shift_col: float,
) -> None:
    imaging, _ = _reference_modules()
    image_ref = _make_test_image(seed=7)
    image_shifted = _fourier_shift_image(image_ref, shift_row, shift_col)

    ref = imaging.cross_correlation_shift_torch(
        torch.from_numpy(image_ref),
        torch.from_numpy(image_shifted),
        10,
    ).numpy()
    out = cross_correlation_shift_cp(cp.asarray(image_ref), cp.asarray(image_shifted), 10)

    np.testing.assert_allclose(cp.asnumpy(out), ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(float(out[0].get()), -shift_row, atol=0.02)
    np.testing.assert_allclose(float(out[1].get()), -shift_col, atol=0.02)


@pytest.mark.parametrize("bin_factor", [1, 2, 3])
def test_bin_mapping_matches_original_quantem(bin_factor: int) -> None:
    _, dpu = _reference_modules()
    mask_np = _make_circular_bf_mask(32, 32, 10)
    inds_row_np, inds_col_np = np.where(mask_np)
    vbf_np = np.zeros((int(mask_np.sum()), 4, 4), dtype=np.float32)

    ref = dpu._bin_mask_and_stack_centered(
        torch.from_numpy(mask_np),
        torch.from_numpy(inds_row_np.astype(np.int64)),
        torch.from_numpy(inds_col_np.astype(np.int64)),
        torch.from_numpy(vbf_np),
        bin_factor,
    )
    ref_mask, ref_inds_row, ref_inds_col, _, ref_mapping = ref

    out = _bin_mapping_only(
        cp.asarray(mask_np),
        cp.asarray(inds_row_np.astype(np.int64)),
        cp.asarray(inds_col_np.astype(np.int64)),
        bin_factor,
    )
    out_mask, out_inds_row, out_inds_col, out_mapping = out

    np.testing.assert_array_equal(cp.asnumpy(out_mask), ref_mask.numpy())
    np.testing.assert_array_equal(cp.asnumpy(out_inds_row), ref_inds_row.numpy().astype(np.int64))
    np.testing.assert_array_equal(cp.asnumpy(out_inds_col), ref_inds_col.numpy().astype(np.int64))
    np.testing.assert_array_equal(cp.asnumpy(out_mapping), ref_mapping.numpy().astype(np.int64))


def test_synchronize_shifts_matches_original_quantem() -> None:
    _, dpu = _reference_modules()
    rng = np.random.default_rng(7)
    num_nodes = 20
    rel_shifts_np = rng.standard_normal((num_nodes - 1, 2)).astype(np.float32)

    rel_shifts_torch = [
        (i, i + 1, torch.from_numpy(rel_shifts_np[i])) for i in range(num_nodes - 1)
    ]
    rel_shifts_cp = [(i, i + 1, cp.asarray(rel_shifts_np[i])) for i in range(num_nodes - 1)]

    ref = dpu._synchronize_shifts(num_nodes, rel_shifts_torch, device="cpu")
    out = synchronize_shifts_cp(num_nodes, rel_shifts_cp)

    np.testing.assert_allclose(cp.asnumpy(out), ref.numpy().astype(np.float32), atol=1e-4)


def test_aberration_svd_polar_matches_original_quantem() -> None:
    _, dpu = _reference_modules()
    gpts = (32, 32)
    bf_radius = 8
    sampling = (0.05, 0.05)
    wavelength = 0.0197

    mask_np = _make_circular_bf_mask(gpts[0], gpts[1], bf_radius)
    kxa = np.fft.fftfreq(gpts[0], sampling[0]).astype(np.float64)
    kya = np.fft.fftfreq(gpts[1], sampling[1]).astype(np.float64)
    kx_bf = np.broadcast_to(kxa[:, None], gpts)[mask_np]
    ky_bf = np.broadcast_to(kya[None, :], gpts)[mask_np]
    basis = np.stack([kx_bf, ky_bf], axis=1) * wavelength
    known_matrix = np.array([[60.0, 5.0], [5.0, 40.0]], dtype=np.float64)
    shifts_ang = basis @ known_matrix

    ref = dpu.fit_aberrations_from_shifts(
        torch.from_numpy(shifts_ang.astype(np.float32)),
        torch.from_numpy(mask_np),
        wavelength,
        gpts,
        sampling,
    )
    out = fit_aberrations_svd_polar(shifts_ang, mask_np, wavelength, gpts, sampling)

    for key in ("C10", "C12", "phi12", "rotation_angle"):
        np.testing.assert_allclose(out[key], ref[key], atol=1e-3)
