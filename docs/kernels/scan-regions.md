# Explicit scan regions

A scan region is a deliberate real-space subset, not an automatic performance
shortcut. Its public order is

```text
full source geometry → explicit half-open scan region → source-frame mapping
                     → selected compressed reads/decode → compact 4D result
                     → full and selected geometry provenance
```

```text
(row_start, row_stop, column_start, column_stop)
```

and each interval is half-open. The coordinate convention is
`(row, column) ≡ (r, c)`.

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

For $I[R_r,R_c,k_r,k_c]$, the selection above keeps
$0\le R_r<32$ and $0\le R_c<48$ while preserving the requested detector
coverage.

## Coordinate, shape, dtype, unit, and provenance contract

For region $(r_0,r_1,c_0,c_1)$, the output scan shape is
$(r_1-r_0,c_1-c_0)$ and detector axes, detector sampling, and requested dtype
remain unchanged unless a separate explicit detector operation says otherwise.
Scan coordinates are indices until scan calibration is supplied.

`LoadResult.metadata` records `full_scan_shape`, the half-open `scan_region`,
the selected scan shape, detector shape, detector/scan bins, source and output
dtypes, source identity, backend/device, and package revision. An application
must display the region as a subset and may not relabel it as full scan
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

## Source map and gates

| Layer | Source |
|---|---|
| Public region normalization and metadata | `src/quantem/gpu/io/load.py` |
| CUDA selected-span load/decode | `src/quantem/gpu/io/backends/cuda` |
| Python MPS selected-span load/decode | `src/quantem/gpu/io/backends/mps` |
| WebGPU local-file planning | `src/quantem/gpu/io/backends/webgpu` |
| Independent shape/reference checks | `tests/io` and real-region parity tests |

Acceptance requires byte-exact selected counts against slicing the same decoded
source, identical row-major ordering, complete provenance, and honest
cold/warm/prepared timings. Full-scan performance claims require a full scan.
