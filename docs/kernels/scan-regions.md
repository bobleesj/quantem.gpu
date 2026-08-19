# Explicit scan regions

A scan region is a deliberate real-space subset, not an automatic performance
shortcut. Its public order is

```text
(row_start, row_stop, column_start, column_stop)
== (y_start, y_stop, x_start, x_stop)
```

and each interval is half-open. The coordinate convention is
`(row, column) ≡ (y, x)`.

```python
from quantem.gpu import io

result = io.load(
    "scan_master.h5",
    backend="cuda",
    scan_region=(0, 32, 0, 48),
)

print(result.data.shape)
print(result.metadata["full_scan_shape"])
print(result.metadata["scan_region"])
```

For $I[r_y,r_x,q_y,q_x]$, the selection above keeps
$0\le r_y<32$ and $0\le r_x<48$ while preserving the requested detector
coverage.

## Optimization model

Crop-aware loading maps the selected scan rows and columns to source frame
indices before reading. It plans only the compressed spans needed for the
region, coalesces nearby spans when measured to be beneficial, decodes the
selected frames, and writes a compact destination without loading the full
scan and slicing afterward.

The optimization must not alter detector sampling, dtype, mask, or binning.
Reports always identify the source full scan shape and the selected half-open
region. Full-scan performance signoff never substitutes a cropped run.

Parity covers non-square regions, non-zero starts, one-pixel regions, source
row boundaries, and the complete region equal to the source shape.
