"""Release-mode GPU parity for fused original-file loading and short shards."""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def paired_loader(tmp_path_factory):
    """Build without diagnostic flags, as in the shipping application."""
    if not shutil.which("swift"):
        pytest.skip("Install the macOS Command Line Tools to run Metal parity")
    root = Path(__file__).resolve().parents[3]
    scratch = tmp_path_factory.mktemp("paired-loader-build")
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
            "OriginalWindowMomentsParity",
            "-Xswiftc",
            "-enable-testing",
        ],
        env={**os.environ, "QGPU_SOURCE_ROOT": str(root)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return scratch / "release/OriginalWindowMomentsParity"


def test_mixed_fused_and_short_shards_preserve_exact_moments(paired_loader, tmp_path):
    """GPU fallback and fusion agree, even with high counts and retired probes."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    # One fused slice and one 97-frame standalone slice in the same window.
    scans, rows, cols = 2145, 64, 64
    pattern = (np.arange(rows * cols, dtype=np.uint16) % 7).reshape(rows, cols)
    values = np.broadcast_to(pattern, (scans, rows, cols)).copy()
    values[2047:2049] = 65535
    values[-1, -1, -1] = 65535
    shards = []
    for index, block in enumerate((values[:2048], values[2048:]), start=1):
        shard = tmp_path / f"synthetic_data_{index:06d}.h5"
        with h5py.File(shard, "x") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=block,
                chunks=(1, rows, cols),
                **plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
        shards.append(shard)
    master = tmp_path / "synthetic_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [65, 33]
        for index, shard in enumerate(shards, start=1):
            group[f"data_{index:06d}"] = h5py.ExternalLink(
                shard.name, "/entry/data/data"
            )
    sources = [master, *shards]
    original_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("QGPU_", "QG_BENCH_"))
    }
    results = []
    variants = [
        {"QGPU_PAIRED_FUSED_MOMENTS": "0"},
        {},
        {"QGPU_PAIRED_READ_AHEAD": "0"},
        {
            "QGPU_PAIRED_PROBE_SKIP_UNSHUFFLE": "1",
            "QGPU_PAIRED_PROBE_FUSED_DECODE": "1",
            "QGPU_PROBE_DENSE_SHA": "all",
        },
    ]
    for index, overrides in enumerate(variants):
        result = subprocess.run(
            [str(paired_loader), str(master), str(tmp_path / f"indexes-{index}")],
            env={**env, **overrides},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "QGPU_DENSE_SHA" not in result.stderr
        record = next(
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{") and '"dense_sha256"' in line
        )
        assert record["slice_frames"] == [2048, 97]
        # Only six bounded DPs use a CPU oracle; all moments compare GPU paths.
        expected = [
            hashlib.sha256(values[frame].astype("<u2").tobytes()).hexdigest()
            for frame in record["sample_frames"]
        ]
        assert record["sample_hashes_u16_le"] == expected
        results.append(record)
    assert len({row["moments_sha256"] for row in results}) == 1
    assert len({row["dense_sha256"] for row in results}) == 1
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    ] == original_hashes
