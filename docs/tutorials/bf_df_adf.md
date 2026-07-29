# Compute BF, DF, and ADF images

Use `quantem.gpu` for common BF/DF/ADF image reductions from 4D-STEM data.

```python
from quantem.gpu import adf, bf, df, mean_dp
from quantem.gpu.io import load

result = load("scan_master.h5", backend="auto", det_bin=1)
data = result.data

bright = bf(data)
annular = adf(data, inner=40, outer=90, unit="px")
dark = df(data)
dp = mean_dp(data)
```

The reduced images are small arrays suitable for `Show2D`:

```python
from quantem.widget import Show2D

Show2D(bright)
Show2D(annular)
Show2D(dark)
```

For crop workflows, load the scan patch first and compute the image on the
patch:

```python
patch = load(
    "scan_master.h5",
    scan_region=(160, 224, 160, 224),
    backend="cuda",
).data

patch_bf = bf(patch)
```

Keep the detector geometry with saved results. At minimum record BF disk
center/radius, detector mask units, backend, and `det_bin`. Use `det_bin=2` or
`4` only for an explicitly labeled preview or memory-limited run.
