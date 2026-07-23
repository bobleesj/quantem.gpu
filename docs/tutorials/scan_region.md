# Load a scan region

Use `load(..., scan_region=...)` when a reconstruction, denoise workflow, or
screening step needs only a rectangular scan patch.

```python
from quantem.gpu import load

result = load(
    "scan_master.h5",
    backend="cuda",
    scan_region=(0, 32, 0, 32),  # row_start, row_stop, col_start, col_stop
)

patch = result.data
print(patch.shape)
print(result.metadata["full_scan_shape"])
print(result.metadata["scan_region"])
```

This is different from loading the full scan and slicing afterward. The
accelerated crop path reads the selected HDF5 detector-frame chunks,
decompresses them, and assembles only the requested scan patch.
