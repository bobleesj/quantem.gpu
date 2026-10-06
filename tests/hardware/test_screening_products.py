"""Screening products on CUDA and MPS equal exact sums of the stored counts."""

import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import detector, screening
from quantem.gpu.dpc.workflow import find_optimal_rotation
from quantem.gpu.screening.workflow import _dpc_phase


def write_master(root, counts, pixel_mask):
    """Write an Arina-style master over two bitshuffle-LZ4 shards."""
    frames = counts.reshape(-1, *counts.shape[2:])
    master = root / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        for index, part in enumerate(np.array_split(frames, 2), start=1):
            shard = root / f"scan_data_{index:06d}.h5"
            with h5py.File(shard, "w") as data_file:
                data_file.create_dataset(
                    "entry/data/data", data=part, chunks=(1, *counts.shape[2:]),
                    **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
                )
            handle[f"entry/data/data_{index:06d}"] = h5py.ExternalLink(shard.name, "/entry/data/data")
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = np.uint32(len(frames))
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = pixel_mask
    return master


def test_screening_products_equal_exact_count_reference(tmp_path):
    """A cache miss builds every product from exact sums, flagged pixels counted as zero."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    rng = np.random.default_rng(7)
    scan_rows, scan_cols = np.indices((32, 32))
    rows, cols = np.indices((24, 28))
    shift_row = 1.5 * np.sin(scan_rows / 4.0)[..., None, None]
    shift_col = 1.2 * np.cos(scan_cols / 5.0)[..., None, None]
    disk = np.hypot(rows - 11.4 - shift_row, cols - 13.7 - shift_col) < 6.5
    counts = rng.poisson(np.where(disk, 60.0, 1.5)).astype(np.uint16)
    pixel_mask = np.zeros((24, 28), dtype=np.uint32)
    pixel_mask[[2, 17], [3, 25]] = 1
    counts[..., pixel_mask != 0] = 0xFFFF
    master = write_master(tmp_path, counts, pixel_mask)

    result = screening.prepare(master, backend=backend, cache_dir=tmp_path / "cache", rotation_steps=30)

    working = np.where(pixel_mask != 0, 0, counts).astype(np.uint64)
    # One mean rule everywhere: the exact total divided in float64, rounded once to float32.
    # These totals stay below 2^24, where the earlier float32(total) / n gave the same values.
    mean_dp = (working.sum(axis=(0, 1)) / (32 * 32)).astype(np.float32)
    center, radius = detector.fit_probe(mean_dp)
    band = {
        name: detector.detector_mask(center, inner * radius, outer * radius, (24, 28))
        for name, (inner, outer) in {
            "bright_field": (0.0, 1.0), "annular_bright_field": (0.5, 1.0),
            "annular_dark_field": (1.0, 2.0), "dark_field": (1.0, np.inf),
        }.items()
    }
    total = working.sum(axis=(2, 3))
    com_row = np.zeros((32, 32))
    com_col = np.zeros((32, 32))
    np.divide((working * rows.astype(np.uint64)).sum(axis=(2, 3)), total, out=com_row, where=total != 0)
    np.divide((working * cols.astype(np.uint64)).sum(axis=(2, 3)), total, out=com_col, where=total != 0)
    com_row = com_row.astype(np.float32)
    com_col = com_col.astype(np.float32)
    com_row -= float(com_row.mean())
    com_col -= float(com_col.mean())
    _, _, rotation_deg, transposed = find_optimal_rotation(com_row, com_col, rotation_steps=30)
    expected = {
        "mean_dp": mean_dp,
        "total_intensity": total,
        "bright_field": working[..., band["bright_field"]].sum(axis=-1).astype(np.float32),
        "annular_bright_field": working[..., band["annular_bright_field"]].sum(axis=-1),
        "annular_dark_field": working[..., band["annular_dark_field"]].sum(axis=-1),
        "dark_field": working[..., band["dark_field"]].sum(axis=-1).astype(np.float32),
        "com_row": com_row,
        "com_col": com_col,
        "dpc_phase": _dpc_phase(com_row, com_col, rotation_deg, transposed),
    }
    reopened = screening.prepare(master, backend=backend, cache_dir=tmp_path / "cache")
    assert (result.from_cache, reopened.from_cache) == (False, True)
    assert result.probe_center == center and result.probe_radius == radius
    assert (result.rotation_deg, result.transposed) == (rotation_deg, transposed)
    for name, values in expected.items():
        for products in (result, reopened):
            assert getattr(products, name).dtype == values.dtype
            np.testing.assert_array_equal(getattr(products, name), values, err_msg=name)


def test_screening_mean_pattern_divides_the_exact_total_once(tmp_path):
    """Bright patterns whose totals pass 2^24: the screening mean pattern rounds once, from float64."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    # 900 positions: dividing by a power of two would round the same either way.
    counts = np.random.default_rng(9).integers(19_000, 23_000, (30, 30, 24, 28)).astype(np.uint16)
    master = write_master(tmp_path, counts, np.zeros((24, 28), dtype=np.uint32))
    result = screening.prepare(master, backend=backend, cache_dir=tmp_path / "cache", rotation_steps=30)
    total = counts.sum(axis=(0, 1), dtype=np.uint64)
    expected = (total / 900).astype(np.float32)
    assert np.any(total.astype(np.float32) / np.float32(900) != expected)
    np.testing.assert_array_equal(result.mean_dp, expected)
