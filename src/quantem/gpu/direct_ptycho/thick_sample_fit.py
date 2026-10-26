"""Shared constants of the joint (thick-sample) SSB fit.

The joint fit itself is the resident search of ``_resident_search.fit_resident`` with the trust-region Newton on the
exact derivatives of ``cuda`` or ``mps.exact_derivatives`` (``_exact_newton``). Units: C10, C12, depth_spread_A in the
engine's C10 unit (Angstrom); phi12 rad; tilt mrad, scan frame (row, col).
"""

PARAMETERS = ("C10", "C12", "phi12", "tilt_row_mrad", "tilt_col_mrad", "depth_spread_A")
# The depth spreads the joint fit searches, in nm (public) and in the engine's Angstrom: DirectPtycho.find_aberrations(tilt=True)
# returns a depth_spread_nm inside it, and its parameter slices span it.
DEPTH_SPREAD_RANGE_NM = (2.0, 60.0)
DEPTH_SPREAD_RANGE_A = (20.0, 600.0)
