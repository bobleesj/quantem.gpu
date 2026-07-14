# IO API

Primary imports:

```python
from quantem.gpu import load, load_scan_region
from quantem.gpu.io import discover_masters, get_metadata, is_master_ready
```

## `load`

Use `load()` for full-field or crop-first HDF5 loading:

```python
full = load("scan_master.h5", backend="auto", det_bin=4)
crop = load("scan_master.h5", backend="cuda", scan_region=(0, 32, 0, 32))
```

Important keyword arguments:

| Argument | Meaning |
|---|---|
| `backend` | `"auto"`, `"cuda"`, `"mps"`, or `"cpu"` |
| `det_bin` | detector binning factor |
| `dtype` | optional browse dtype such as `"u8"` when supported |
| `scan_region` | `(row_start, row_stop, col_start, col_stop)` crop |

## `load_scan_region`

`load_scan_region()` remains available as a compatibility helper. New code should
prefer `load(..., scan_region=...)`.

## Metadata and readiness

Use metadata/readiness helpers before launching a large decode:

```python
from quantem.gpu.io import discover_masters, get_metadata, is_master_ready

masters = discover_masters("/data/session")
for master in masters:
    print(master, is_master_ready(master), get_metadata(master))
```
