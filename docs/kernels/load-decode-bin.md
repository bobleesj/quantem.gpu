# Load, decode, and bin

The load pipeline converts compressed detector evidence into a typed,
accelerator-resident 4D-STEM representation without changing its scientific
meaning.

```python
from quantem.gpu import io

result = io.load(
    "scan_master.h5",
    backend="auto",
    dtype="u16",
    det_bin=1,
)

print(result.data.shape, result.data.dtype)
print(result.metadata)
```

`det_bin=1` preserves native detector sampling. Detector binning is explicit,
count-preserving, and recorded. Scan cropping is never introduced as an
automatic resource policy.

## Coordinate and layout contract

The logical output is

$$
I[R_r,R_c,q_r,q_c],
\qquad (\text{row},\text{column})\equiv(r,c).
$$

In plain terms, `(row, column)` is `(r, c)` for both scan and detector axes.

Storage shards may flatten $(R_r,R_c)$ into a frame index, and a device layout
may be detector-major or packed. `LoadResult` metadata must still report the
logical scan and detector shapes, source/output dtype, crop/bin plan, and
source identity.

## Pipeline stages

```text
discover/open -> metadata and index -> read spans -> decode
              -> bad-pixel correction -> dtype/bin/layout -> resident result
```

Profile these stages separately where the storage/runtime makes that possible:

1. source discovery, file open, and metadata;
2. index mapping and compressed-span planning;
3. storage read or memory-map page-in;
4. bitshuffle/LZ4 decode;
5. bad-pixel handling and value-range audit;
6. dtype conversion and exact detector/scan reduction;
7. destination allocation and layout conversion;
8. first usable scientific product; and
9. final provenance/cache work.

On unified memory, storage page-in, decode, and device access may overlap. Do
not invent a separate “upload” number when bytes are mapped without a copy.

## Optimization model

The reusable fast path should:

- align reads to source shards or compressed blocks;
- keep file descriptors, indexes, pipelines, and masks prepared;
- use double or triple buffering only within the measured memory plan;
- overlap read/decode with reduction when command dependencies permit;
- fuse dtype conversion and detector binning with decode when parity remains
  exact;
- reuse pinned/shared/device buffers rather than allocate per batch;
- avoid per-batch host synchronization and device-to-host copies; and
- materialize only the requested resident layout or product.

Compressed size is not a memory estimate. Admission includes decoded bytes,
decoder scratch, output buffers, reduction products, allocator reserve, and
other active GPU users.

## Resource-policy boundary

The package estimates the cost of each exact plan. A client may choose an
automatic detector bin for a memory-limited machine only when it records and
presents:

- requested and selected detector bin;
- source and output detector shapes;
- scan region and scan bin;
- source, accumulation, and resident dtypes;
- predicted and measured memory; and
- the reason the plan changed.

The resulting array is binned evidence, not native-resolution evidence.

## Sparse and selected-position loading

Iterative methods may request globally distributed scan positions without
loading the full volume:

```python
batch = io.load(
    master_paths,
    scan_indices=per_frame_indices,
    scan_shape=(512, 512),
    backend="cuda",
)
```

The loader sorts and de-duplicates storage indices for efficient reads, then
restores the requested scientific order. A random sample must record its seed
and selected indices.

## Source map and gates

| Layer | Source |
|---|---|
| Public Python contract | `src/quantem/gpu/io` |
| CUDA decode | `src/quantem/gpu/io/backends/cuda` |
| Python MPS/Metal decode | `src/quantem/gpu/io/backends/mps` |
| WebGPU local-file decode | `src/quantem/gpu/io/backends/webgpu` |
| Native IO and Metal load plans | `src/quantem/gpu/swift/Sources/Native4DSTEMIO` and `Metal4DSTEMKernels` |
| Independent reference | `src/quantem/gpu/io/backends/cpu` |

Acceptance covers exact decoded counts, odd detector shapes, incomplete edge
bins, bad pixels, dtype/overflow behavior, full source identity, memory-budget
failure, and cold/warm/prepared timing. See
[Cross-backend parity](../performance/parity.md).
