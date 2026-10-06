# Frozen scientific-operation references

These synthetic numerical fixtures were moved unchanged from Live4DSTEM
commit `54208a8`, originally quantem commit `faab8731`. Copyright (c) 2025
ophusgroup, MIT License; see the repository LICENSE. No private acquisition
is included.

- `numpy.json`: counts, median correction, means and Gaussian references.
- `normalized_grid.json`: interpolation boundary references.
- `torch_mps_parameters.json`: frozen Fourier, gradient and window references.

Run `bash scripts/check_metal_scientific_numerics.sh` on a Command Line Tools
Mac, or `swift test --filter MetalScientificNumericsTests` with full Xcode.
Neither check requires Live4DSTEM or quantem. Do not regenerate expected values
to make a migration pass. App-specific ownership and saved-result checks remain
in Live4DSTEM; its full seven-tilt journey checks the complete merge separately.
