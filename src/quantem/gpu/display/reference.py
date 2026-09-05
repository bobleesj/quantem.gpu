"""Compatibility imports; canonical implementation: ``quantem.gpu.display.backends.cpu``."""

from quantem.gpu.display.backends.cpu import (
    DisplayScale as DisplayScale,
    __all__ as __all__,
    colorize as colorize,
    dequantize_uint8 as dequantize_uint8,
    histogram as histogram,
    normalize as normalize,
    transform as transform,
)
