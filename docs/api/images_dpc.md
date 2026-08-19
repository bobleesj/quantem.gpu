# Detector and DPC API

The detector and DPC namespaces own reusable reductions from
$I[s_r,s_c,q_r,q_c]$ to detector-shaped or scan-shaped scientific products.
They do not own presentation, cache policy, or automatic resource choices.

## Inputs and outputs

| Call | Input | Output |
|---|---|---|
| `detector.mean_dp(data)` | one supported 4D source | detector-shaped mean diffraction pattern |
| `detector.bf(data, ...)` | 4D source plus bright-field geometry | scan-shaped bright-field sum |
| `detector.adf(data, ...)` | 4D source plus annular limits | scan-shaped annular dark-field sum |
| `detector.df(data, ...)` | 4D source plus inner limit | scan-shaped dark-field sum |
| `dpc.run(data, ...)` | 4D source plus optional detector mask/rotation | `DPCResult` with CoM, aligned DPC, phase, and rotation metadata |

The current detector convenience functions return NumPy product arrays after
backend execution. `DPCResult.phase`, `com_row`, `com_col`,
`com_row_aligned`, and `com_col_aligned` are float32 scan-shaped arrays.

## Shapes, coordinates, dtypes, and units

Every input follows `(scan_row, scan_column, detector_row, detector_column)`.
Every BF/DF/ADF and DPC product follows `(scan_row, scan_column)`. Mean
diffraction follows `(detector_row, detector_column)`.

`center=(row, column)` is always detector order. Radii with `unit="px"` are
detector pixels. Radii with `unit="mrad"` require convergence-semiangle
calibration in the source metadata. Integer detector sums widen according to
the shared accumulation contract; DPC moments and phase products are float32.

## Errors and unsupported requests

- A milliradian detector limit without `semiangle_mrad` raises with a corrective
  instruction; it is never interpreted as pixels.
- A supplied scan shape whose product differs from the number of diffraction
  patterns raises rather than reshaping silently.
- Missing accelerator capability fails explicitly; it is not reported as GPU
  work after a hidden CPU fallback.
- Row and column components may not be swapped to match a private launch layout.

## Provenance

An application records the source identity, original and loaded shapes,
detector bin/crop, source and accumulation dtypes, detector mask or
center/radius, calibration and units, selected backend/device, DPC rotation,
transpose choice, and package revision. A binned input produces a binned-source
product and must not be labeled native detector resolution.

## Minimal example

```python
from quantem.gpu import detector, dpc, io

loaded = io.load("scan_master.h5", backend="auto", det_bin=1)

bright_field = detector.bf(loaded.data)
annular_dark_field = detector.adf(
    loaded.data,
    inner=40,
    outer=90,
    unit="px",
)
dpc_result = dpc.run(loaded.data)
```

For a client that needs several launch products from the same source, use
`screening.prepare()` and reuse its small derived state instead of traversing
the complete detector volume independently for every product.

## Integration boundary

QuantEM.GPU owns detector geometry, exact reduction arithmetic, CoM/DPC/iDPC
math, backend dispatch, and result fields. A consuming application owns when to
run the operations, cache admission, memory-policy choices, and presentation.

See [BF, DF, and ADF reductions](../kernels/virtual-detectors.md) and
[CoM, DPC, and iDPC](../kernels/com-dpc-idpc.md) for the equations,
optimization model, source map, and parity gates.
