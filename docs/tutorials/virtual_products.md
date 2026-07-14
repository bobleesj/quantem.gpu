# Compute virtual products

Common virtual detector products are exposed directly from `quantem.gpu`.

```python
from quantem.gpu import adf, bf, df, dpc, load, mean_dp

result = load("scan_master.h5", backend="auto", det_bin=4)
data = result.data

bright = bf(data)
annular = adf(data, inner=40, outer=90, unit="px")
dark = df(data)
dp = mean_dp(data)
dpc_result = dpc(data)
```

The reduced products are small arrays suitable for `Show2D`:

```python
from quantem.widget import Show2D

Show2D(bright)
Show2D(annular)
Show2D(dpc_result.phase)
```

For scan-region workflows, load the patch first and compute products on the
patch:

```python
patch = load(
    "scan_master.h5",
    scan_region=(160, 224, 160, 224),
    backend="cuda",
).data

patch_bf = bf(patch)
```

Keep the detector geometry with the result when saving product metrics. At
minimum record BF disk center/radius, detector mask units, backend, and
`det_bin`.
