"""Canonical I/O modules replace retired paths and keep platforms lazy."""

import subprocess
import sys
from importlib import import_module
from importlib.util import find_spec

import pytest


@pytest.mark.parametrize(
    "backend,canonical,symbols",
    [
        ("cpu", "quantem.gpu.io.hdf5.cpu", ("load_master",)),
        ("mps", "quantem.gpu.io.hdf5.mps.decode", ("MPSDecompressor", "load_prepared_frames")),
    ],
    ids=["cpu", "mps"],
)
def test_only_canonical_backend_imports_are_supported(backend, canonical, symbols):
    assert find_spec("quantem.gpu.io.backends") is None
    if backend == "mps":
        pytest.importorskip("Metal")
    current = import_module(canonical)
    for name in symbols:
        assert callable(getattr(current, name))


def test_common_io_imports_do_not_load_accelerator_runtimes():
    script = """
import sys
import quantem
# Native QuantEM is the dataset owner and a required dependency.
before = set(sys.modules)
from quantem.gpu import io
from quantem.gpu.io.hdf5 import mps
assert io.__all__ == ['discover', 'inspect', 'load', 'save']
added = set(sys.modules) - before
assert not any(name == 'Metal' or name.startswith('Metal.') for name in added)
assert not any(name == 'cupy' or name.startswith('cupy.') for name in added)
assert 'quantem.gpu.io.hdf5.mps.decode' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
