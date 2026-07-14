# Product API

Top-level product imports:

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
    masked_sum,
    mean_dp,
    virtual,
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

For visual review, hand the reduced product to `quantem.widget.Show2D`.
