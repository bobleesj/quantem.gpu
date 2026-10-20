# Device, optics, screening, parallax, and scan-rotation API

This page completes the Python API map for public namespaces that do not need a
full operation-specific reference page. All coordinates exposed to users are
`(row, column)`.

## Device selection

| Call | Purpose | Failure behavior |
|---|---|---|
| `device.profile(device=None)` | notebook-friendly environment and selected-device summary | automatic selection may report CPU for diagnostics |
| `device.resolve_device(device="auto")` | resolve `"auto"`, `"cuda"`, `"cuda:N"`, `"mps"` or `"cpu"` to a Torch device string for code that also runs on CPU | `"auto"` picks CUDA, then MPS, then CPU and prints its choice once |
| `device.runtime_notice()` | the one-line install hint when a GPU is present but its runtime is missing, else `None` | printed once at first GPU use |
| `device.detect()` | choose a native CUDA or MPS accelerator | raises when neither GPU runtime is available |
| `device.resolve(name="auto")` | validate `cuda` or `mps` | raises on an unavailable or unknown runtime |
| `device.release_cached_memory(verbose=False)` | collect garbage, empty the Torch, CuPy and MLX caches on every GPU the process has opened, and return the gigabytes this process handed back to the driver | live arrays are untouched; returns `0.0` without a GPU |

CPU reported by `profile()` is diagnostic. Scientific GPU calls do not silently
turn that diagnostic result into a CPU execution path.

`release_cached_memory()` returns cached GPU memory to the driver after `del`
and `close()`:

```python
from quantem.gpu import device

data.close()
del data
device.release_cached_memory(verbose=True)
```

It prints one line per GPU the process has opened, for example
`GPU 0: released 2.1 GB, 32.0 GB in use, 69.9 GB free`, and returns the total
released (`2.147483648`). "Released" is what this process handed back: the
drop in the bytes Torch has reserved plus the bytes of CuPy's pool on CUDA, and
the drop of the Metal driver's total for the process on Apple silicon. "In use"
and "free" are the GPU's totals from the driver and include other processes.
Cached FFT plans of the calling thread are dropped. Arrays something still
refers to stay, including the locals of a failed call that the traceback of the
last error keeps (`sys.last_exc` in a notebook). A GPU the process has not
opened is not visited. The measurements are in the
[2026-10-11 record](../maintainer/2026-10-11-release-cached-memory.md).

## Electron optics

`quantem.gpu.optics` exports `wavelength_A_from_kV(voltage_kV)`, the
relativistic electron wavelength in Å for an accelerating voltage in kV, and
`interaction_parameter(voltage_kV)`, the relativistic interaction parameter
σ = 2π m e λ / h² in rad/(V·Å) (6.526 × 10⁻⁴ at 300 kV) computed with that
same wavelength: a projected potential of V_p V·Å shifts the phase by σ V_p.
`quantem.gpu.optics.physics.ssb_upsampling_factor` chooses the SSB output grid
from voltage (kV), convergence semi-angle (mrad), and scan step (Å, a
`(row, col)` pair).

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
optional fixed `scan_detector_rotation_deg` with `scan_detector_rotation_flipped`, the cache policy (`cache`, `cache_dir`,
`refresh`), and the number of rotation search steps. A cache miss loads the
acquisition once into encoded CUDA or MPS storage; a cache hit needs no GPU. A
cache reopen is not a raw HDF5 load. The raw source remains the scientific
evidence source, and metadata retains the source fingerprint, parameters, and
timing.

Every preparation publishes mean DP, BF, DF, CoM, rotation, and iDPC, plus
`total_intensity`, `annular_bright_field`, and `annular_dark_field` as exact
`uint64` count maps, on CUDA and MPS alike. A cache written without those
three maps is rebuilt, not reused. Screening needs unsigned integer counts: a
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
result = parallax.run(data, fit_aberrations=True, scan_sampling_A=(0.5, 0.5), semiangle_mrad=30)
```

`parallax.run` takes the acquisition `io.load` returns on CUDA and returns
`ParallaxResult`. The acquisition stays encoded: only the bright-field pixels
are read, in bounded bands of scan rows, as exact counts. The bright-field
center `bright_field_center_px` `(row, column)` and radius
`bright_field_radius_px` are fitted from the mean diffraction pattern
unless given. `fit_aberrations=True` needs `scan_sampling_A` (Å per scan
pixel, `(row, col)`) and the detector calibration, `semiangle_mrad` or
`detector_sampling_mrad` (mrad per detector pixel, `(row, col)`). The fit is
returned as `ParallaxResult.aberrations` (`C10_nm`
and `C12_nm` in nm, `phi12_deg` in degrees, scan frame) and
`ParallaxResult.scan_detector_rotation_deg` (degrees, in [0, 360)), in the
units and convention `SSB` takes.
Parallax runs on CUDA only; an acquisition on an Apple GPU (MPS)
raises `NotImplementedError`. Persist center, BF radius,
scan sampling, the semiangle or detector sampling, upsampling factor,
aberration-fit choice, and package revision with the result.

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
