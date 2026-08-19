# Data model and coordinates

A 4D-STEM acquisition records one two-dimensional diffraction pattern at each
two-dimensional probe position:

$$
I(\mathbf R,\mathbf k)=I[R_r,R_c,k_r,k_c].
$$

The public array convention is

$$
(\text{row},\text{column}) \equiv (r,c).
$$

Therefore $\mathbf R=(R_r,R_c)$ is the **probe/scan coordinate** in real space
and $\mathbf k=(k_r,k_c)$ is the **detector scattering coordinate** in
diffraction space.
Code uses `row`/`column`; equations use the matching $r$/$c$ component
subscripts. Both express the same ordered axes.

## Shapes and indexing

For a source with scan shape $(N_{R_r},N_{R_c})$ and detector shape
$(N_{k_r},N_{k_c})$:

```text
data.shape == (scan_rows, scan_columns, detector_rows, detector_columns)
           == (N_{R_r}, N_{R_c}, N_{k_r}, N_{k_c})
```

`data[row, column]` selects one diffraction pattern. A scan-shaped result such
as BF or iDPC has shape `(scan_rows, scan_columns)`. A detector-shaped result
such as the mean diffraction pattern has shape
`(detector_rows, detector_columns)`.

## Regions are half-open

Both scan and detector regions use:

```text
(row_start, row_stop, column_start, column_stop)
```

This is a half-open interval: the start is included and the stop is excluded.
Cropping is an explicit scientific choice, never an implicit memory or speed
policy.

## Binning preserves counts

For detector bin factors $(b_r,b_c)$, the binned count at output detector
coordinate $(Q_r,Q_c)$ is

$$
I_b[R_r,R_c,Q_r,Q_c]
=\sum_{i=0}^{b_r-1}\sum_{j=0}^{b_c-1}
I[R_r,R_c,b_rQ_r+i,b_cQ_c+j].
$$

The implementation widens the accumulator before summation. Incomplete edge
bins are retained, so output dimensions use ceiling division. The output
metadata records the original and output shapes, factors, source dtype,
accumulation dtype, and output dtype.

## Units

| Quantity | Coordinates | Typical units |
|---|---|---|
| Probe/scan position $\mathbf R$ | $(R_r,R_c)$ | scan pixels, nm, or Å |
| Detector position $\mathbf k$ | $(k_r,k_c)$ | detector pixels, mrad, or reciprocal length |
| Detector mask $M(\mathbf k)$ | $(k_r,k_c)$ | dimensionless weight |
| Detector signal $I(\mathbf R,\mathbf k)$ | all four axes | detector counts |

Never label an uncalibrated detector pixel as mrad or a scan index as physical
length. Calibration origin and units are part of provenance.

## Implementation invariant

CUDA thread indices, Metal grid coordinates, and WebGPU invocation IDs may be
laid out for coalescing. The adapter must translate that private layout back to
the public `(row, column) ≡ (r, c)` contract. Transposition, flattening, or
detector-major storage is an implementation detail and requires parity tests
that catch row/column swaps and rectangular-shape mistakes.
