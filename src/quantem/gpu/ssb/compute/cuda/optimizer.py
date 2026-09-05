"""Compatibility imports; canonical implementation: ``quantem.gpu.ssb.backends.cuda.optimizer``."""

from quantem.gpu.ssb.backends.cuda.optimizer import (
    _build_param_arrays as _build_param_arrays,
    _eval_single as _eval_single,
    _sequential_optimize as _sequential_optimize,
    _suggest as _suggest,
    batch_nelder_mead as batch_nelder_mead,
    batch_optimize as batch_optimize,
)
