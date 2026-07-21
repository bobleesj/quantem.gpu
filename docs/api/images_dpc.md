# BF, DF, ADF, and DPC API

Top-level image and DPC imports:

```python
from quantem.gpu import (
    adf,
    auto_probe,
    bf,
    center_of_mass,
    com,
    detector_mask,
    df,
    dpc,
    idpc,
    load_calibration_products,
    masked_sum,
    mean_dp,
)
```

Typical workflow:

```python
from quantem.gpu import adf, bf, dpc, load

data = load("scan_master.h5", backend="auto", det_bin=4).data
bf_image = bf(data)
adf_image = adf(data, inner=40, outer=90, unit="px")
dpc_result = dpc(data)
```

These functions accept loaded arrays, `LoadResult`-style objects, and migrated
chunk-backed MPS data where supported.

For a screen page or exported viewer that needs BF/DF/CoM/DPC products on
launch, prefer `load_calibration_products()` and reuse the product cache instead
of reloading and reducing the full raw HDF5 volume on every open.

For visual review, hand the reduced image to `quantem.widget.Show2D`.
