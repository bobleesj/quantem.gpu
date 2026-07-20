# Load an HDF5 master

Use `quantem.gpu.load()` for bitshuffle/LZ4 HDF5 4D-STEM masters. The returned
object is a `LoadResult` with `data` and `metadata`.

```python
from quantem.gpu import load

result = load(
    "scan_master.h5",
    backend="auto",
    det_bin=4,
)

data = result.data
metadata = result.metadata
print(data.shape, data.dtype)
print(metadata.get("scan_shape"), metadata.get("detector_shape"))
```

On CUDA, `data` is a device array. On MPS, data may be a chunk-backed object
that keeps the heavy detector frames device-owned for product computation. CPU is
available as a reference path, but it is not the target for large interactive
work.

Display is a widget concern:

```python
from quantem.widget import Show4DSTEM

Show4DSTEM(result.data)
```

Keep load parameters explicit in reports:

- `backend`
- `det_bin`
- `dtype`
- `scan_region`, if used
- public-safe file label, scan shape, detector shape, and timing
