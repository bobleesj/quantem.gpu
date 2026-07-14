# API guide

This is a practical API guide for the migrated compute package. It focuses on
the public functions scientists and downstream packages should call first.

```python
import quantem.gpu as qgpu

qgpu.device_report()
```

Main namespaces:

- `quantem.gpu.io` for HDF5 load/decompress/save helpers.
- `quantem.gpu` top-level exports for BF, DF, ADF, and DPC images.
- `quantem.gpu.ssb` for SSB compute APIs.

The API is still release-candidate level. Prefer public functions documented
here over internal backend modules.
