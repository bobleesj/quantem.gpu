"""Release-mode GPU check of the median fill of source-flagged pixels on original loads.

A ``.median`` load must hold what Python ``io.load`` computes from the filled
counts: each flagged pixel takes the median of its valid neighbors, the mean of
two middle values rounded half to even. That holds for the resident's counts,
its per-pixel totals and every frame's exact DPC moments, whether a decode
window fuses its moments into the unshuffle or measures them after the fill.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Requires a physical macOS Metal device"
)

ROWS = COLUMNS = 64
SCAN = (64, 128)
# Two 4096-frame windows: the first lies inside one file, the second straddles
# the file boundary with a 1024-frame slice.
SHARDS = (5120, 3072)


@pytest.fixture(scope="module")
def median_loader(tmp_path_factory):
    """Build without diagnostic flags, as in the shipping application."""
    if not shutil.which("swift"):
        pytest.skip("Install the macOS Command Line Tools to run Metal parity")
    root = Path(__file__).resolve().parents[3]
    scratch = tmp_path_factory.mktemp("hot-pixel-median-build")
    result = subprocess.run(
        [
            "swift",
            "build",
            "-c",
            "release",
            "--package-path",
            str(Path(__file__).parent / "swift_original_window_moments"),
            "--scratch-path",
            str(scratch),
            "--product",
            "OriginalHotPixelMedianParity",
        ],
        env={**os.environ, "QGPU_SOURCE_ROOT": str(root)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return scratch / "release/OriginalHotPixelMedianParity"


def _acquisition(folder, dtype, np, h5py, plugin):
    """Write a two-file synthetic acquisition whose flagged pixels hold the dtype's maximum."""
    rng = np.random.default_rng(20261025)
    counts = rng.poisson(3.0, size=(sum(SHARDS), ROWS, COLUMNS)).astype(dtype)
    mask = np.zeros((ROWS, COLUMNS), dtype=np.uint32)
    mask[20, 31] = 1  # interior: eight neighbors, an even count
    mask[0, 40] = 2  # edge: five neighbors
    mask[63, 63] = 4  # corner: three neighbors
    mask[40, 10:12] = 8  # adjacent pair: seven neighbors each
    mask[50, 50] = mask[50, 51] = mask[51, 50] = 16  # corner of an L: six neighbors
    counts[:, mask != 0] = np.iinfo(dtype).max
    shards = []
    start = 0
    for index, frames in enumerate(SHARDS, start=1):
        shard = folder / f"synthetic_data_{index:06d}.h5"
        with h5py.File(shard, "x") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=counts[start : start + frames],
                chunks=(1, ROWS, COLUMNS),
                **plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
        shards.append(shard)
        start += frames
    master = folder / "synthetic_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = list(SCAN)
        for index, shard in enumerate(shards, start=1):
            group[f"data_{index:06d}"] = h5py.ExternalLink(
                shard.name, "/entry/data/data"
            )
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask", data=mask
        )
    return master, shards, counts, mask


def _products(np, counts):
    """Per-pixel totals and per-frame total, row and column moments, exactly."""
    flat = counts.reshape(len(counts), -1).astype(np.uint64)
    pixel = np.arange(ROWS * COLUMNS, dtype=np.uint64)
    moments = np.stack(
        [
            flat.sum(axis=1),
            (flat * (pixel // COLUMNS)).sum(axis=1),
            (flat * (pixel % COLUMNS)).sum(axis=1),
        ],
        axis=1,
    )
    return flat.sum(axis=0), moments


def _load(loader, master, folder, policy, np):
    """Run the public loader once and read back what its resident holds."""
    out = folder / f"resident-{policy}"
    out.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("QGPU_", "QG_BENCH_"))
    }
    result = subprocess.run(
        [str(loader), str(master), str(folder / f"index-{policy}"), str(out), policy],
        env={**env, "QGPU_ORIGINAL_PROFILE": "1"},
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    profile = next(
        json.loads(line.split(" ", 1)[1])
        for line in result.stderr.splitlines()
        if line.startswith("ORIGINAL_PACK_PROFILE ")
    )
    scans = summary["scans"]
    flagged = summary["flagged"]
    return (
        summary,
        profile,
        np.fromfile(out / "detector_sum.u64", dtype="<u8"),
        np.fromfile(out / "moments.u64", dtype="<u8").reshape(scans, 3),
        np.fromfile(out / "columns.u32", dtype="<u4").reshape(len(flagged), scans),
    )


# uint16 fuses DPC into the unshuffle only in windows inside one file; uint32
# fuses every window.
@pytest.mark.parametrize(("dtype", "fused_windows"), [("uint16", 1), ("uint32", 2)])
def test_median_load_holds_python_fill_in_counts_totals_and_moments(
    median_loader, tmp_path, dtype, fused_windows
):
    """Every frame, fused or not, holds the round-half-even median fill."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    from quantem.gpu.resident.hot_pixels import fill_flagged, median_plan

    master, shards, counts, mask = _acquisition(tmp_path, dtype, np, h5py, plugin)
    before = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in [master, *shards]
    ]
    filled = fill_flagged(counts.copy(), median_plan(mask), "median")
    flagged = np.flatnonzero(mask)
    # The fixture separates the roundings: a floored mean is one lower in some frames.
    window = counts[:, 19:22, 30:33].reshape(len(counts), 9).astype(np.uint64)
    middle = np.sort(np.delete(window, 4, axis=1), axis=1)[:, 3:5].sum(axis=1)
    assert ((middle >> np.uint64(1)) != filled[:, 20, 31]).sum() > 100

    summary, profile, totals, moments, columns = _load(
        median_loader, master, tmp_path, "median", np
    )
    assert summary["flagged"] == flagged.tolist()
    assert summary["excluded"] == []
    assert profile["decode_window_frames"] == 4096
    assert profile["fused_dpc_windows"] == fused_windows
    expected_totals, expected_moments = _products(np, filled)
    expected_columns = filled.reshape(len(filled), -1)[:, flagged].T.astype(np.uint32)
    differing = np.flatnonzero((columns != expected_columns).any(axis=0))
    assert differing.size == 0, (
        f"{differing.size} frames differ at flagged pixels, first {differing[:4]}"
    )
    assert np.array_equal(totals, expected_totals)
    bad_frames = np.flatnonzero((moments != expected_moments).any(axis=1))
    assert bad_frames.size == 0, (
        f"{bad_frames.size} frames hold other DPC moments, windows "
        f"{sorted(set((bad_frames // 4096).tolist()))}"
    )

    _, _, raw_totals, raw_moments, raw_columns = _load(
        median_loader, master, tmp_path, "preserve", np
    )
    expected_totals, expected_moments = _products(np, counts)
    assert np.array_equal(raw_totals, expected_totals)
    assert np.array_equal(raw_moments, expected_moments)
    assert (raw_columns == np.iinfo(dtype).max).all()
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in [master, *shards]
    ] == before
