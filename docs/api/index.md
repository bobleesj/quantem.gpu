# API guide

This guide maps public scientific contracts to their stable entry points. Use
it when integrating QuantEM.GPU; use the scientific-kernel and runtime sections
when implementing or optimizing those contracts.

```python
import quantem.gpu as qgpu

qgpu.device.detect()
```

| Domain | Stable entry point | Primary result |
|---|---|---|
| device selection | `quantem.gpu.device` | explicit backend/device description |
| discovery, load, and save | `quantem.gpu.io` | typed data plus provenance metadata |
| BF, DF, ADF, mean diffraction | `quantem.gpu.detector` | scan- or detector-shaped product |
| CoM, DPC, and iDPC | `quantem.gpu.dpc` | `DPCResult` |
| parallax reconstruction | `quantem.gpu.parallax` | domain reconstruction result |
| SSB fitting/reconstruction | `quantem.gpu.SSB` | `SSBResult` or `SSBSeriesResult` |
| display/export math | `quantem.gpu.display`, `quantem.gpu.movie` | display buffer or encoded artifact |
| device selection | `quantem.gpu.device.detect`, `profile`, `resolve` | explicit backend/device description |
| electron optics | `quantem.gpu.optics` | wavelength, reciprocal cutoff, aberration phase, and fit results |
| cached screening | `quantem.gpu.screening.prepare` | `ScreeningResult` with derived launch products and provenance |
| parallax | `quantem.gpu.parallax.run` | `ParallaxResult` |

Native Swift/Metal products for macOS and iOS clients:

- `MetalImageFFT.logMagnitude` for Browser FFT of an already-transferred 2D
  product. See [Native Metal image endpoints](metal_image.md).
- `MetalImageRuntime` for histogram, range, and display contracts.
- `Native4DSTEMIO` for Python-free HDF5/EMD discovery, prepared QH5 indexes,
  and bounded native frame windows.
- `Metal4DSTEMStreamingIO` for bounded native QH5 decode, exact `uint64`
  products, source audits, and on-demand full-resolution diffraction frames.
- `Metal4DSTEMLoadPlan`, `Metal4DSTEMStreamingPlan`,
  `Metal4DSTEMResidentCacheIO`, and `Metal4DSTEMResidentSummaryIO` for explicit
  native load, resource, resident-cache, and exact prepared-product provenance.
  See [Native 4D-STEM load and cache contract](native_4dstem_io.md).
- `MetalSSBEngine` for exact native 512×512 SSB reconstruction,
  phase-variance evaluation, and deterministic 200-trial TPE plus Nelder–Mead
  fitting. See [SSB API](ssb.md).

Native clients call these endpoints directly. They are not a local Python
backend.

The API is still release-candidate level. Prefer public functions documented
here over internal backend modules.

## Complete public namespace map

| Namespace/product | Stable public surface | Contract page |
|---|---|---|
| `device` | `detect`, `profile`, `resolve` | [Device selection and supporting APIs](core.md) |
| `io` | `discover`, `inspect`, `load`, `save` | [I/O API](io.md) |
| `detector`, `dpc` | detector reductions, `DPCResult` | [Detector and DPC API](images_dpc.md) |
| `optics` | wavelength/convergence conversions, aberration phase and fitting | [Device selection and supporting APIs](core.md) |
| `screening` | `prepare`, `ScreeningResult` | [Device selection and supporting APIs](core.md) |
| `parallax` | `run`, `ParallaxResult` | [Device selection and supporting APIs](core.md) |
| `SSB` | `SSB`, `SSBResult`, series results | [SSB API](ssb.md) |
| `display`, `movie` | display transforms and encoded artifacts | [Movie API](movie.md) and [display kernels](../kernels/display-export.md) |
| SwiftPM products | `MetalImageFFT`, `MetalImageRuntime`, `Native4DSTEMIO`, `Metal4DSTEMKernels`, `Metal4DSTEMStreamingIO`, `MetalSSBKernels` | [Native Metal image](metal_image.md), [native load/cache](native_4dstem_io.md), and [SSB](ssb.md) |
| Remote services | browse, MAPED, and SSB protocol services | [QuantEM.GPU Remote](../remote/index.md) |

Backend modules, private helpers, launch geometry, cache scheduling, and UI
state are deliberately not public API.
