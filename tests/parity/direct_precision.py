"""Shared float32 parity requirements for direct ptychography backend tests."""

from quantem.gpu.direct_ptycho.contract import BackendPrecision


PRECISION = BackendPrecision()
PHASE_RTOL = 2.0e-4
PHASE_ATOL = 2.0e-4
LOSS_RTOL = 1.0e-5
LOSS_ATOL = 1.0e-6
