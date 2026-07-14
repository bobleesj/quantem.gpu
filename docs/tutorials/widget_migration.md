# Use quantem.gpu from quantem.widget

The widget migration keeps existing display APIs while moving heavy IO and
compute into `quantem.gpu`.

New loading examples should use `load(..., scan_region=...)`:

```python
from quantem.widget import Show4DSTEM, load

result = load(
    "scan_master.h5",
    scan_region=(0, 32, 0, 32),
    backend="cuda",
)

Show4DSTEM(result.data)
```

Existing code that imports `load_scan_region` can keep working for one release:

```python
from quantem.widget import load_scan_region

result = load_scan_region("scan_master.h5", scan_region=(0, 32, 0, 32))
```

Internally, widget compatibility wrappers should re-export from `quantem.gpu`
instead of maintaining second copies of GPU decompression or product kernels.

The target ownership split is:

| Package | Owns |
|---|---|
| `quantem.gpu` | HDF5 GPU IO/decompress, load-to-device, product compute, SSB compute, device policy |
| `quantem.widget` | anywidget front end, controls, export, interaction, display |
| `quantem.live` | app/dashboard orchestration that calls `quantem.gpu` over time |
