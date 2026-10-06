# Device, optics, screening, parallax, and scan-rotation API

This page completes the Python API map for public namespaces that do not need a
full operation-specific reference page. All coordinates exposed to users are
`(row, column)`.

## Device selection

| Call | Purpose | Failure behavior |
|---|---|---|
| `device.profile(device=None)` | notebook-friendly environment and selected-device summary | automatic selection may report CPU for diagnostics |
| `device.detect()` | choose a native CUDA or MPS accelerator | raises when neither GPU runtime is available |
| `device.resolve(name="auto")` | validate `cuda`, `mps`, or explicit browser `webgpu` | raises on an unavailable or unknown runtime |

CPU reported by `profile()` is diagnostic. Scientific GPU calls do not silently
turn that diagnostic result into a CPU execution path.

## Electron optics

`quantem.gpu.optics` exports `wavelength_A_from_kV(voltage_kV)`, the
relativistic electron wavelength in Å for an accelerating voltage in kV.
`quantem.gpu.optics.physics.ssb_upsampling_factor` chooses the SSB output grid
from voltage (kV), convergence semi-angle (mrad), and scan step (Å).

(screening-products)=
## Screening products

```python
from quantem.gpu import screening

products = screening.prepare(
    "scan_master.h5",
    backend="auto",
    scan_shape=(512, 512),
)

print(products.metadata["timing"])
```

`screening.prepare` builds or reopens derived detector and DPC products and
returns `ScreeningResult`. Its public controls are `backend`, `scan_shape`, an
optional fixed `rotation_angle_deg`, the cache policy (`cache`, `cache_dir`,
`refresh`), and the number of rotation search steps. A cache miss loads the
acquisition once into encoded CUDA or MPS storage; a cache hit needs no GPU. A
cache reopen is not a raw HDF5 load. The raw source remains the scientific
evidence source, and metadata retains the source fingerprint, parameters, and
timing.

Every preparation publishes mean DP, BF, DF, CoM, rotation, and iDPC, plus
`total_intensity`, `annular_bright_field`, and `annular_dark_field` as exact
`uint64` count maps, on CUDA and MPS alike. Those three fields are `None` only
when reopening a cache written without them; a caller must check them
explicitly. Caches contain either all three exact maps or none, and a partial
or non-`uint64` set is never reused. Screening needs unsigned integer counts: a
float acquisition raises `ValueError`. No scan crop or detector binning is
applied.

`screening.prepare` is currently a Python CUDA/MPS API. Native Swift/Metal and
WebGPU expose reusable detector and DPC operations, but they do not implement
this prepared-product cache contract. Applications must not label an
independently assembled native or browser product set as a
`ScreeningResult`.

## Parallax reconstruction

```python
from quantem.gpu import io, parallax

data = io.load("scan_master.h5", backend="cuda")
result = parallax.run(data, voltage_kV=200, scan_sampling=0.5, fit_aberrations=True)
```

`parallax.run` takes the acquisition `io.load` returns on CUDA and returns
`ParallaxResult`. The acquisition stays encoded: only the bright-field pixels
are read, in bounded bands of scan rows, as exact counts. The bright-field
center `(row, column)` and radius are fitted from the mean diffraction pattern
unless given. Parallax runs on CUDA only; an acquisition on an Apple GPU (MPS)
raises `NotImplementedError`. Persist center, BF and sampling radii,
accelerating voltage, scan sampling, upsampling factor, aberration-fit choice,
and package revision with the result. The alignment bins the detector with
float32 atomic additions, so the last bits of the shifts can differ between
runs on the same input.

## Scan rotation

`geometry.rotate_scan(data, angle_degrees)` rotates the scan plane of a 4D-STEM
acquisition; diffraction patterns are never rotated. Multiples of 90 degrees are
exact and keep the dtype. On the encoded acquisition `io.load` returns on CUDA,
the result is a new encoded acquisition on the same GPU, built band by band
without decoding the source whole; its counts are reordered, never
interpolated, so other angles need `interpolation="nearest"`. Dense NumPy, CuPy
and Torch arrays also rotate with bilinear interpolation at other angles. An
encoded acquisition on an Apple GPU (MPS) raises `NotImplementedError`.

## Integration boundary

QuantEM.GPU owns these typed calculations, validation, backend selection, and
scientific provenance. A consuming application owns presentation, user-facing
policy, cache scheduling, and lifecycle. Private compute modules are not a
substitute for these public entry points.
