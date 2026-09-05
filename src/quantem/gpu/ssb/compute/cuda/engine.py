"""Compatibility imports; canonical implementation: ``quantem.gpu.ssb.backends.cuda.engine``."""

from quantem.gpu.ssb.backends.cuda.engine import (
    SSBEngine as SSBEngine,
    _PreparedCudaBfSubset as _PreparedCudaBfSubset,
    _choose_reduce_block as _choose_reduce_block,
)
