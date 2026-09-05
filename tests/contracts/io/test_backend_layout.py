"""Representation-named IO modules preserve old imports and lazy platforms."""

import ast
from importlib import import_module
from importlib.resources import files
import subprocess
import sys

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
def test_legacy_imports_are_the_same_implementation(
    backend, legacy, canonical, symbols,
):
    if (backend, canonical) == ("mps", "dense"):
        pytest.importorskip("Metal")
    prefix = f"quantem.gpu.io.backends.{backend}"
    old = import_module(f"{prefix}.{legacy}")
    current = import_module(f"{prefix}.{canonical}")
    for name in symbols:
        assert getattr(old, name) is getattr(current, name)

    # Compatibility paths cannot hide another decoder or cache implementation.
    source = files(prefix).joinpath(f"{legacy}.py").read_text()
    tree = ast.parse(source)
    assert isinstance(tree.body[0], ast.Expr)  # module documentation
    assert all(isinstance(node, ast.ImportFrom) for node in tree.body[1:])


def test_common_io_imports_do_not_load_accelerator_runtimes():
    script = """
import sys
from quantem.gpu import io
from quantem.gpu.io.backends.mps import series
assert io.__all__ == ['discover', 'inspect', 'load', 'save']
assert 'Metal' not in sys.modules
assert 'cupy' not in sys.modules
assert 'quantem.gpu.io.backends.mps.dense' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
