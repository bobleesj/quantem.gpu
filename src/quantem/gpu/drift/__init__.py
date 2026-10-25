"""How far the specimen moved between the frames of a time series, measured between each frame's phase image, and the
series moved back into register.

    found = drift(phases, sampling_A=0.124)    # every pair, lattice-safe; also drift(stream) or drift([result, ...])
    found.show(phases)                         # the path, and the mean before and after correction
    aligned, mean = found.corrected(phases)    # every frame on frame 0, over the region all frames share
    found = running(phases, sampling_A=0.124)  # low dose: each frame against the running sum of the others
    shift = register(reference, moving, sampling_A=0.124).shift_px

Translation only: the frames are compared on their own pixel grids, displacement_A[f] where frame f sees the specimen
against frame 0, I_f(r) = I_0(r - d_f)."""

from quantem.gpu.drift.registration import bands, peaks, prepared, register, register_spectra, spots
from quantem.gpu.drift.series import Drift, drift, footprint, running, shifted

__all__ = ["Drift", "bands", "drift", "footprint", "peaks", "prepared", "register", "register_spectra", "running", "shifted", "spots"]
