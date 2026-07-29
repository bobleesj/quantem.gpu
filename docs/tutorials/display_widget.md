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

Widget code should call the same `load(..., scan_region=...)` API instead of
maintaining second copies of GPU decompression or BF/DF/DPC kernels.

The target ownership split is:

| Package | Owns |
|---|---|
| `quantem.gpu` | HDF5 GPU IO/decompress, load-to-device, BF/DF/DPC compute, SSB compute, movie rendering helpers, device policy |
| `quantem.widget` | anywidget front end, controls, interaction, display, and user-facing export buttons |
| `quantem.live` | app/dashboard orchestration that calls `quantem.gpu` over time |

## Display ptychographic SSB results

Ptychographic SSB compute should stay in `quantem.gpu.ssb`; interactive display
should stay in `quantem.widget.ShowPtycho`.

```python
from quantem.gpu import SSB
from quantem.widget import Show2D

workflow = SSB.from_array(
    data,
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=21.4,
    scan_sampling_A=0.5,
)
result = workflow.run(trials=200, refinement="nelder-mead")

Show2D(result.phase)
Show2D(abs(result.object_wave))
```

For interactive ptychography tuning, pass the same workflow or shared result
to `quantem.widget.ShowPtycho`. The
widget owns controls, previews, and export; this package owns the reconstruction
math and backend policy.
