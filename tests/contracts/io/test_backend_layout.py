"""Canonical I/O modules replace retired aliases and keep platforms lazy."""

import subprocess
import sys
from importlib import import_module
from importlib.util import find_spec

import pytest


@pytest.mark.parametrize(
    "backend,legacy,canonical,symbols",
    [
        ("cpu", "reference", "dense", ("load_master", "_bin_sum")),
        ("cuda", "compact_h5", "packed", (
            "CudaCompactH5ResidentSource", "load_compact_h5_cuda",
            "warm_compact_h5_cuda_kernels", "_cuda_kernels",
        )),
        ("mps", "compact_v3", "packed", (
            "MPSCompactV3Resident", "load_compact_v3_mps",
            "read_compact_v3_index", "MPSCompactV3Error",
        )),
        ("mps", "decoder", "dense", (
            "MPSChunked4DSTEM", "MPSMasterPlan", "load_master",
            "load_master_chunked", "load_prepared_frames", "clear_mps_cache",
        )),
    ],
)
def test_only_canonical_backend_imports_are_supported(
    backend, legacy, canonical, symbols,
):
    prefix = f"quantem.gpu.io.backends.{backend}"
    assert find_spec(f"{prefix}.{legacy}") is None
    if (backend, canonical) == ("mps", "dense"):
        pytest.importorskip("Metal")
    current = import_module(f"{prefix}.{canonical}")
    for name in symbols:
        assert callable(getattr(current, name))


def test_common_io_imports_do_not_load_accelerator_runtimes():
    script = """
import sys
import quantem
# Native QuantEM is the dataset owner and a required dependency.
before = set(sys.modules)
from quantem.gpu import io
from quantem.gpu.io.backends.mps import series
assert io.__all__ == ['discover', 'inspect', 'load', 'save']
added = set(sys.modules) - before
assert not any(name == 'Metal' or name.startswith('Metal.') for name in added)
assert not any(name == 'cupy' or name.startswith('cupy.') for name in added)
assert 'quantem.gpu.io.backends.mps.dense' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
