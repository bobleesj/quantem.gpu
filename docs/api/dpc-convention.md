# DPC integration convention

`quantem.gpu.dpc.integrate(row, col)` returns the zero-mean scalar whose
periodic gradient best matches the supplied scan-frame vector field. With
frequencies measured in cycles per scan pixel, the Fourier multiplier is
`-1j / (2*pi)`. There is no implicit polarity reversal.

A known scalar's analytic gradient reconstructs with positive sign and unit
gain. Uncalibrated detector CoM inputs yield detector-pixel × scan-pixel units,
not radians. Detector angles, scan spacing and electron wavelength are needed
for physical phase calibration. Rotation remains ambiguous modulo 180 degrees
when inferred only from gradient consistency.

Python and browser WebGPU use the same convention. When detector axes are
exchanged, exchange `(row, col)` first and then apply the rotation, `R(theta) D`.
Do not exchange them again after rotation. Constant fields and the scalar DC
component cannot be recovered by periodic integration.

## Existing results

The previous public integrator returned `-pi/2` times the sign-preserving
integral. For identical inputs, new output is `-2/pi` times old output.
Screening cache version 9 invalidates previous cached products. QuantEM.live
records `dpc_integration_convention="gradient-integral/1"` and requires a new
screening pass for older records. Exported figures and existing dashboard
arrays are not rewritten just by opening them. Rebuild browser bundles and
restart existing kernels when installing this change.

## Parity checks

- `tests/parity/test_dpc_integration_convention.py`: analytic sign, gain,
  constant and transverse fields, forced rotations and exchanged axes.
- `tests/parity/test_dpc_rotation_agreement.py`: fitted and forced geometry.
- `tests/webgpu/dpc-python-parity.ts`: actual detector-counts-to-iDPC browser
  path against Python reference arrays, including 180-degree and axis exchange.
- `tests/webgpu/run_dpc_python_parity.py`: headed hardware WebGPU runner.

The browser fixture format is recorded in the runner docstring. Real-data
fixtures must retain native detector sampling and document any invalid-pixel
correction. A backend parity check proves consistent computation, not physical
phase calibration or element identification.

## Independent reference comparison

Original QuantEM's iCoM operator uses `-1j*q/q_squared` with reciprocal-space
CoM rather than an already calibrated phase gradient. Converting gradient
inputs by `1/(2*pi)` makes the operators agree; do not globally change that
implementation's scale or sign.

py4DSTEM's DPC implementation uses `0.25` in an iterative residual update.
It is not a one-shot integration normalization. Its finite-difference forward
model and boundary handling also differ from this periodic Fourier integral.
The pinned reference used for the operator audit is py4DSTEM commit
`62c042eedd71ebcb7c52ecb5294182df132f9b5d`, `process/phase/dpc.py`.
