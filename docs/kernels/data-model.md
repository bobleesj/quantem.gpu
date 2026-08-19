# Data model and coordinates

A 4D-STEM acquisition records one two-dimensional diffraction pattern at each
two-dimensional probe position:

$$
I(\mathbf r,\mathbf q)=I[r_y,r_x,q_y,q_x].
$$

The public array convention is

$$
(\text{row},\text{column}) \equiv (y,x).
$$

Therefore $\mathbf r=(r_y,r_x)$ is the **scan coordinate** in real space and
$\mathbf q=(q_y,q_x)$ is the **detector coordinate** in diffraction space.
Code may use `row`/`column`; equations may use $y$/$x$. They mean the same
ordered axes.

## Shapes and indexing

For a source with scan shape $(N_{r_y},N_{r_x})$ and detector shape
$(N_{q_y},N_{q_x})$:

```text
data.shape == (scan_rows, scan_columns, detector_rows, detector_columns)
           == (N_ry, N_rx, N_qy, N_qx)
```

`data[row, column]` selects one diffraction pattern. A scan-shaped result such
as BF or iDPC has shape `(scan_rows, scan_columns)`. A detector-shaped result
such as the mean diffraction pattern has shape
`(detector_rows, detector_columns)`.

## Regions are half-open

Both scan and detector regions use:

```text
(row_start, row_stop, column_start, column_stop)
== (y_start, y_stop, x_start, x_stop)
```

This is a half-open interval: the start is included and the stop is excluded.
Cropping is an explicit scientific choice, never an implicit memory or speed
policy.

## Binning preserves counts

For detector bin factors $(b_y,b_x)$, the binned count at output detector
coordinate $(Q_y,Q_x)$ is

$$
I_b[r_y,r_x,Q_y,Q_x]
=\sum_{i=0}^{b_y-1}\sum_{j=0}^{b_x-1}
I[r_y,r_x,b_yQ_y+i,b_xQ_x+j].
$$

The implementation widens the accumulator before summation. Incomplete edge
bins are retained, so output dimensions use ceiling division. The output
metadata records the original and output shapes, factors, source dtype,
accumulation dtype, and output dtype.

## Units

| Quantity | Coordinates | Typical units |
|---|---|---|
| Scan position $\mathbf r$ | $(r_y,r_x)$ | scan pixels, nm, or Å |
| Detector position $\mathbf q$ | $(q_y,q_x)$ | detector pixels, mrad, or reciprocal length |
| Detector mask $M(\mathbf q)$ | $(q_y,q_x)$ | dimensionless weight |
| Detector signal $I(\mathbf r,\mathbf q)$ | all four axes | detector counts |

Never label an uncalibrated detector pixel as mrad or a scan index as physical
length. Calibration origin and units are part of provenance.

## Implementation invariant

CUDA thread indices, Metal grid coordinates, and WebGPU invocation IDs may be
laid out for coalescing. The adapter must translate that private layout back to
the public `(row, column) ≡ (y, x)` contract. Transposition, flattening, or
detector-major storage is an implementation detail and requires parity tests
that catch row/column swaps and rectangular-shape mistakes.
