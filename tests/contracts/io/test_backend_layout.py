"""Importing the I/O package keeps the accelerator runtimes lazy."""

import subprocess
import sys


def test_common_io_imports_do_not_load_accelerator_runtimes():
    script = """
import sys
import quantem
# Native QuantEM is the dataset owner and a required dependency.
before = set(sys.modules)
from quantem.gpu import io
from quantem.gpu.io.hdf5 import mps
assert io.__all__ == ['DataRepresentation', 'Dataset4dstemGPU', 'PairedLoader', 'discover', 'inspect', 'load', 'save']
added = set(sys.modules) - before
assert not any(name == 'Metal' or name.startswith('Metal.') for name in added)
assert not any(name == 'cupy' or name.startswith('cupy.') for name in added)
assert 'quantem.gpu.io.hdf5.mps.decode' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
