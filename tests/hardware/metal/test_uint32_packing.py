"""Exact original uint32 HDF5-to-Metal packed-resident scientist workflow."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Requires macOS Metal")


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    """Build the public resident workflow without experimental tuning flags."""
    root = Path(__file__).resolve().parents[3]
    scratch = tmp_path_factory.mktemp("uint32-build")
    subprocess.run(
        ["swift", "build", "-c", "release", "--package-path",
         str(Path(__file__).parent / "swift_original_packing"),
         "--scratch-path", str(scratch), "--product", "UInt32PackingParity"],
        env={**os.environ, "QGPU_SOURCE_ROOT": str(root)}, check=True, timeout=600,
    )
    return scratch / "release" / "UInt32PackingParity"


@pytest.mark.parametrize("signal", ["sparse", "boundaries", "full_range"])
def test_uint32_original_counts_and_integrated_images(executable, tmp_path, signal):
    """Preserve all bits, high-count sums, empty masks and switch-and-return."""
    import h5py
    import hdf5plugin
    import numpy as np

    shape = (32, 64, 64)
    rng = np.random.default_rng(20260909)
    if signal == "full_range":
        values = rng.integers(0, 2**32, shape, dtype=np.uint32)
    elif signal == "boundaries":
        edges = np.array([0, 1, 255, 256, 16383, 16384, 65535, 65536,
                          2**24 + 1, 2**31, 2**32 - 2, 2**32 - 1], dtype=np.uint32)
        values = np.resize(edges, shape)
    else:
        values = rng.poisson(0.08, shape).astype(np.uint32)
    sources = []
    # A shard boundary inside a packing window exercises nonzero output offsets.
    for number, chunk in enumerate(np.split(values, [13]), start=1):
        source = tmp_path / f"counts_data_{number:06d}.h5"
        with h5py.File(source, "x") as handle:
            handle.create_dataset("entry/data/data", data=chunk, chunks=(1, 64, 64),
                                  **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        sources.append(source)
    master = tmp_path / "counts_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [4, 8]
        for number, source in enumerate(sources, start=1):
            group[f"data_{number:06d}"] = h5py.ExternalLink(source.name, "/entry/data/data")
    sources.append(master)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    oracle = tmp_path / "oracle.u32"
    values.astype("<u4").tofile(oracle)
    result = subprocess.run([str(executable), str(master), str(oracle)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("UINT32_EXACT_PASS") == 2
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
