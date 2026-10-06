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

An ordinary acquisition loads complete and encoded; `io.load` rejects
`scan_region` for it. Read the region from the loaded acquisition instead:

```python
from quantem.gpu import io

data = io.load("scan_master.h5", backend="cuda")
patch = data.read(scan_region=(0, 32, 0, 48))

print(patch.shape)
```

`read` decodes the region in bounded scan blocks and returns a Torch tensor on
the source GPU; `data[0:32, 0:48]` selects the same values. Scaled precision
storage applies the region before resident allocation:

```python
result = io.load(
    "float32_master.h5",
    backend="cuda",
    dtype="scaled_uint16",
    scan_region=(0, 32, 0, 48),
)

print(result.shape)
print(result.metadata["source_shape"])
print(result.metadata["selection"]["scan_region"])
```

For $I[R_r,R_c,k_r,k_c]$, each selection above keeps
$0\le R_r<32$ and $0\le R_c<48$ while preserving the requested detector
coverage.

## Coordinate, shape, dtype, unit, and provenance contract

For region $(r_0,r_1,c_0,c_1)$, the output scan shape is
$(r_1-r_0,c_1-c_0)$ and detector axes, detector sampling, and requested dtype
remain unchanged unless a separate explicit detector operation says otherwise.
Scan coordinates are indices until scan calibration is supplied.

For precision storage, `Dataset4dstemGPU.metadata` records `source_shape`,
`selection` with the half-open `scan_region` and `detector_region`, the selected
`scan_shape`, `detector_shape`, `source_dtype`, `working_dtype`,
`storage_dtype`, and the measured `precision` report. A tensor returned by
`read` or indexing carries no metadata, so the caller records its region. An
application must display the region as a subset and may not relabel it as full
scan coverage.

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
| Bounded reads of a loaded acquisition | `src/quantem/gpu/io/read.py` |
| Precision region normalization and metadata | `src/quantem/gpu/io/precision.py` |
| Selected compressed-chunk reads | `src/quantem/gpu/formats/hdf5/reads.py` |
| CUDA selected-span load/decode | `src/quantem/gpu/io/hdf5/cuda` |
| Python MPS selected-span load/decode | `src/quantem/gpu/io/hdf5/mps` |
| WebGPU local-file planning | `src/quantem/gpu/io/hdf5/webgpu` |
| Independent shape/reference checks | `tests/hardware/cuda/test_ans_resident.py`, `tests/hardware/mps/test_resident_read.py`, and `tests/hardware/test_regional_precision.py` |

Acceptance requires byte-exact selected counts against slicing the same decoded
source, identical row-major ordering, complete provenance, and honest
cold/warm/prepared timings. Full-scan performance claims require a full scan.
