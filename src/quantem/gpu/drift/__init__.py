"""How far the specimen moved between the frames of a time series, measured between each frame's phase image.

    found = drift.drift(phases, sampling_A=sampling_A)     # every pair, lattice-safe
    found = drift.running(phases, sampling_A=sampling_A)   # each frame against the running sum of the others
    shift = drift.register(reference, moving, sampling_A=sampling_A).shift_px

Translation only: the frames are compared on their own pixel grids, displacement_A[f] where frame f sees the specimen
against frame 0."""

from quantem.gpu.drift.registration import bands, peaks, prepared, register, register_spectra, spots
from quantem.gpu.drift.series import Drift, drift, footprint, running, shifted

__all__ = ["Drift", "bands", "drift", "footprint", "peaks", "prepared", "register", "register_spectra", "running", "shifted", "spots"]
