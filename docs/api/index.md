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

The API is still release-candidate level. Prefer public functions documented
here over internal backend modules.
