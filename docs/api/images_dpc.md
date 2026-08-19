# Detector and DPC API

The public API is grouped by scientific domain:

```python
from quantem.gpu import detector, dpc, io, screening

loaded = io.load("scan_master.h5", backend="auto", det_bin=1)
data = loaded.data

bright_field = detector.bf(data)
annular_dark_field = detector.adf(
    data,
    inner=40,
    outer=90,
    unit="px",
)
dpc_result = dpc.run(data)
```

For a client that needs BF, DF, CoM, and DPC products at launch, use
`screening.prepare()` and reuse its small derived cache rather than reducing
the full HDF5 volume on every open.

Native-detector agreement and scientific-count claims should start from
`det_bin=1` and a count-preserving dtype. Presentation remains outside this
compute API.
