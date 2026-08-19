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

Native Swift/Metal products for macOS and iOS clients:

- `MetalImageFFT.logMagnitude` for Browser FFT of an already-transferred 2D
  product. See [Native Metal image endpoints](metal_image.md).
- `MetalImageRuntime` for histogram, range, and display contracts.
- `Native4DSTEMIO` for Python-free HDF5/EMD discovery.
- `Metal4DSTEMLoadPlan`, `Metal4DSTEMStreamingPlan`, and
  `Metal4DSTEMResidentCacheIO` for explicit native load, resource, and cache
  provenance. See [Native 4D-STEM load and cache contract](native_4dstem_io.md).

Native clients call these endpoints directly. They are not a local Python
backend.

The API is still release-candidate level. Prefer public functions documented
here over internal backend modules.
