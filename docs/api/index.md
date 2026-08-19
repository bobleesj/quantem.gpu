# API guide

This is a practical API guide for the migrated compute package. It focuses on
the public functions scientists and downstream packages should call first.

```python
import quantem.gpu as qgpu

qgpu.device.detect()
```

Main namespaces:

- `quantem.gpu.io` for HDF5 discovery, inspection, loading, and saving.
- `quantem.gpu.detector` for BF, DF, ADF, and virtual-detector images.
- `quantem.gpu.dpc` and `quantem.gpu.parallax` for their scientific workflows.
- `quantem.gpu.SSB` for SSB fitting and reconstruction.

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
