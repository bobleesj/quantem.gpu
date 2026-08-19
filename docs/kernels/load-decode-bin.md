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
I[R_r,R_c,k_r,k_c],
\qquad (\text{row},\text{column})\equiv(r,c).
$$

In plain terms, `(row, column)` is `(r, c)` for both scan and detector axes.

Storage shards may flatten $(R_r,R_c)$ into a frame index, and a device layout
may be detector-major or packed. `LoadResult` metadata must still report the
logical scan and detector shapes, source/output dtype, crop/bin plan, and
source identity.

## Dtype and memory contract

The load path keeps four precision decisions separate:

1. **source dtype** — the detector counts stored in HDF5, commonly `uint16`;
2. **working dtype** — compact decode/staging precision, such as an audited
   lossless `uint8` low-byte path;
3. **accumulation dtype** — widened integer precision used for detector or scan
   sums; and
4. **resident dtype** — the array representation delivered to downstream
   kernels and recorded in provenance.

`uint16` is exact for native `uint16` counts only while every correction and
sum fits 0 through 65,535. Detector binning therefore widens accumulation and,
when needed, the resident result to `uint32`. `uint8` is exact only for a native
`uint8` source or after a complete identity-bound range audit proves every
corrected count is at most 255. An explicit `dtype="u8"` without that proof is
a saturating browse transform and must retain its saturation count.

For output detector bin $b$ and $w$ resident bytes per value,

$$
B_{\mathrm{payload}}
=N_{R_r}N_{R_c}
\left\lceil\frac{N_{k_r}}{b}\right\rceil
\left\lceil\frac{N_{k_c}}{b}\right\rceil w.
$$

This is only the resident payload. Peak memory also includes live compressed
bytes, decode/reduction scratch, staging or upload buffers, allocator reserve,
products, and other active GPU users. Report predicted payload and measured
peak separately; on Apple unified memory also report process footprint,
pressure, and swap.

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

## Count-preserving detector binning

For detector bin factor $b$, each output detector pixel is the exact sum of one
$b\times b$ source block:

$$
I_b[R_r,R_c,k'_r,k'_c]
=\sum_{i=0}^{b-1}\sum_{j=0}^{b-1}
I[R_r,R_c,bk'_r+i,bk'_c+j].
$$

This preserves the complete scan grid and total detector counts. It does not
crop real space, interpolate detector values, average counts, or label the
result as native detector resolution. Incomplete detector-edge blocks are
summed over the source pixels that exist, using the same rule on every backend.

The efficient path performs this sum while decoded chunks are already on the
accelerator. It writes the final resident dtype/layout directly instead of
materializing both a full unbinned volume and a second binned copy. Correctness
still requires widened accumulation, explicit overflow behavior, bad-pixel
ordering, original/output detector shapes, selected bin, and the resource-policy
reason in provenance.

```text
compressed chunk
      ↓ GPU decode
decoded source counts ──► bad-pixel policy ──► exact b×b sum
                                                   ↓
                                  final resident binned counts
```

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
- resident payload, predicted peak, measured peak, and the measurement source;
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
