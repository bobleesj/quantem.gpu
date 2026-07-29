"""Shared float32 parity requirements for SSB backend tests."""

from quantem.gpu.ssb.compute import SSBPrecision


PRECISION = SSBPrecision()
PHASE_RTOL = 2.0e-4
PHASE_ATOL = 2.0e-4
LOSS_RTOL = 1.0e-5
LOSS_ATOL = 1.0e-6
