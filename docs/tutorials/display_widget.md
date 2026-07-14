# Display with quantem.widget

Use `quantem.gpu` for heavy IO and compute, then hand arrays or reduced images
to `quantem.widget` for display.

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
instead of maintaining second copies of GPU decompression or BF/DF/DPC kernels.

The target ownership split is:

| Package | Owns |
|---|---|
| `quantem.gpu` | HDF5 GPU IO/decompress, load-to-device, BF/DF/DPC compute, SSB compute, movie rendering helpers, device policy |
| `quantem.widget` | anywidget front end, controls, interaction, display, and user-facing export buttons |
| `quantem.live` | app/dashboard orchestration that calls `quantem.gpu` over time |

## Display ptychographic SSB results

Ptychographic SSB compute should stay in `quantem.gpu.ssb`; display should stay
in `quantem.widget`.

```python
from quantem.gpu.ssb import ssb
from quantem.widget import Show2D

result = ssb(data, voltage_kV=300, semiangle_mrad=21.4, scan_sampling_A=0.5)

Show2D(result.phase)
Show2D(abs(result.object_wave))
```

If a future widget exposes an SSB workflow, it should call `quantem.gpu.ssb` for
the reconstruction and keep controls, previews, and export in widget code.
