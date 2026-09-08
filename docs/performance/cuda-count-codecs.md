# CUDA count codecs: speed and peak memory

Adding ANS to bitpacking saves resident storage, but costs decoding time. It
does **not** imply a twofold memory increase. This investigation measures the
tradeoff on three complete uint16 acquisitions and explicitly separates
payload, calculated resident layout, shared scratch reservation, and unmeasured
full-load peak memory.

These are **experimental CUDA codecs**, not qualification of the currently
published ANS file reader, writer, or conversion API. The pipeline is:

```text
compressed HDF5 bytes → CUDA LZ4/bitshuffle → bounded native uint16 block
  → direct count ANS, bitpacking, or bitpacking followed by ANS
  → exact regional indexes + source residual reductions → uint32 detector sums
```

Three complete arrays of shape `(512, 512, 192, 192)` contain
28,991,029,248 counts, or 54 GiB of dense uint16 data. Axes are scan row, scan
column, detector row, detector column. No crop, binning, count truncation, or
changed validity mask is used. All scientific encoding, histogram/model
fitting, decoding, index construction, detector reductions and comparisons run
on one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, physical GPU1,
with CUDA runtime and driver API version 13.0.

The portable {download}`benchmark record <data/cuda-count-codecs-2026-09-08.json>`
owns the exact bytes, timing distributions, source identities, code hashes,
protocol, and qualification limits. The full private fixture and executable
prototype are retained separately; the summary alone is not an executable
benchmark distribution. The following tables abbreviate the device name to
RTX PRO 6000 Blackwell Max-Q.

## What memory was actually measured?

The probe processes 512 scan positions at a time, using reusable GPU arenas.
The four codecs execute sequentially in those same arenas. Complete source
coverage therefore does not mean complete candidate residency.

| Scope | Memory kind | MiB | Device tested | Date tested |
| --- | --- | --- | --- | --- |
| Codec and validation arenas | Measured shared-pool high-water | 150.65 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| H5 IO pool | Measured shared-pool high-water | 143.24 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Combined pools | Measured shared-pool high-water | 293.89 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |

The 150.65 MiB includes bounded input, payload capacity, codec/model/index
scratch, and inverse/query validation buffers. GPU H5 input raises the combined
pool reservation to 293.89 MiB. It does not allocate another complete dense
acquisition. Both quantities exclude CUDA-context/driver allocations outside
the pools, pinned host memory, process RSS and pre-existing residents.

**Per-codec peak differences were not measured.** Reserving the same scratch
arenas for all arms masks their individual allocation requirements. Identical
probe reservation does not prove that standalone packing and ANS loaders have
identical peaks. Full-load, full-series and per-codec peak fields remain null
in the record. Process RSS peak was not measured either.

The earlier run's whole-card sample includes an existing large resident
session. It is not attributable to the candidate codec and cannot establish
the codec's incremental peak. Admission requires the complete device baseline
and the actual overlapping allocations of the intended workflow.

## Resident storage and the additional ANS structures

The following values are sums of materialized per-block payload lengths and
metadata sizes, not measurements of simultaneously resident full candidates.
The calculated layout includes used payload, directories, expanded model
tables and exact interaction indexes. It excludes allocator padding, context,
transient scratch and application outputs.

| Representation | Memory kind | GiB | Device tested | Date tested |
| --- | --- | --- | --- | --- |
| Direct count ANS | Used source payload | 3.4621 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Calculated resident layout | 4.0333 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Used source payload | 4.4174 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Calculated resident layout | 5.0410 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Used source payload | 3.6083 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Calculated resident layout | 4.3638 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |

Compared with packing alone, packing followed by ANS saves **828.49 MiB of
payload** across these three acquisitions. The fitted byte-model tables add
135 MiB in the undeduplicated resident calculation, leaving **693.49 MiB net
resident-layout savings**. Their directories and interaction indexes have the
same size. These model tables are counted per block in that calculation;
this probe reuses one table arena while streaming.

Direct count ANS has the smallest calculated layout here. Its fixed model
configuration is shared once; byte-ANS models depend on the block data. This
result does not prove optimal compression for other dose distributions or
detector geometry.

The common exact interaction indexes occupy 314.62 MiB over the three sources.
They store bitpacked 8-by-8 and 32-by-32 detector-region sums, independently of
the source codec. Query kernels combine these sums with exact decoded source
residuals. They do not convert the entire ANS source into a second packed
resident before each query.

Five byte-model contexts are active per real block: 92,160 bytes of expanded
tables. The prototype actually reserves all 17 contexts: 313,344 bytes.
Active model size and allocated table capacity must not be interchanged in a
peak-memory estimate. The common scratch figure already includes the table
capacity; adding it again would double count memory.

If a future full-resident implementation kept all 17 contexts for every block,
it would require another **324 MiB** above the active-table calculation for
these three acquisitions. Packed-byte ANS would then have a calculated layout
of **4.6802 GiB**, leaving 369.49 MiB savings over packing alone. This is an
allocation scenario, not an additional measurement. Model-table compaction or
sharing must be implemented and verified before claiming the smaller layout
as an actual resident allocation.

## What does the extra decoding time buy?

These totals cover all three complete acquisitions. CUDA stage sums and
observed wall intervals are separate measurements.

| Representation | Timing boundary | Statistic | Seconds | Device tested | Date tested |
| --- | --- | --- | --- | --- | --- |
| Direct count ANS | Encode CUDA stage sum | Sum over 1,536 blocks | 0.5531 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Encode call wall | Sum over 1,536 blocks | 0.6090 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Full inverse CUDA | Sum over 1,536 blocks | 0.4274 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Full inverse wall | Sum over 1,536 blocks | 0.4459 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Encode CUDA stage sum | Sum over 1,536 blocks | 0.5808 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Encode call wall | Sum over 1,536 blocks | 0.6635 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Full inverse CUDA | Sum over 1,536 blocks | 0.1991 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Full inverse wall | Sum over 1,536 blocks | 0.2098 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Encode CUDA stage sum | Sum over 1,536 blocks | 1.0119 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Encode call wall | Sum over 1,536 blocks | 1.1910 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Full inverse CUDA | Sum over 1,536 blocks | 0.4122 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Full inverse wall | Sum over 1,536 blocks | 0.4429 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |

Packing followed by ANS reduces packing-only payload by 18.3%, while its
full-inverse CUDA duration is 2.07 times longer. The large-ADF query penalty
is smaller: median CUDA duration is 1.46 times longer in this prototype.
Direct ANS is smaller than either, with a small query/inverse penalty relative
to packed-byte ANS. For a capacity-constrained workload that may be an
acceptable tradeoff; it does not establish the latency of a complete
interactive series.

The following large-ADF queries use center `(95.5, 95.5)`, inner radius 40 and
outer radius 80 in detector-pixel coordinates. **Each timing produces 512 scan
outputs, not a full 512-by-512 image batch.** Masks and plans are prepared once
and reused on fresh source blocks. Each distribution has 1,536 samples.

| Representation | Timing boundary | Statistic | Milliseconds | Device tested | Date tested |
| --- | --- | --- | --- | --- | --- |
| Direct count ANS | CUDA events | p50 | 0.2899 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | CUDA events | p95 | 0.3100 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | CUDA events | Maximum | 3.2880 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Synchronized call wall | p50 | 0.2955 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Synchronized call wall | p95 | 0.3182 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Direct count ANS | Synchronized call wall | Maximum | 3.2928 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | CUDA events | p50 | 0.1950 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | CUDA events | p95 | 0.2133 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | CUDA events | Maximum | 0.9191 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Synchronized call wall | p50 | 0.2002 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Synchronized call wall | p95 | 0.2195 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking without ANS | Synchronized call wall | Maximum | 1.9787 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | CUDA events | p50 | 0.2853 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | CUDA events | p95 | 0.3039 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | CUDA events | Maximum | 0.7911 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Synchronized call wall | p50 | 0.2911 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Synchronized call wall | p95 | 0.3120 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |
| Bitpacking then ANS | Synchronized call wall | Maximum | 3.1887 | RTX PRO 6000 Blackwell Max-Q | 2026-09-08 |

ANS adds dependent state transitions and renormalization to bit extraction.
Packed-byte encoding also adds width analysis, packing, empirical histograms
and model fitting. A faster ANS kernel by itself therefore does not make the
complete encode chain faster. The single-block profiler also finds too few
active blocks in shared reductions. Its numbers do not bound a joint
all-acquisition architecture.

## Correctness and timing boundaries

Every source count is compared after each full inverse on CUDA. Twelve detector
poses compare every output against independent dense GPU reductions, including
translations, fractional centers/radii, boundaries, empty/full masks and large
jumps. Original pixel-validity semantics and float64 inclusive detector
boundaries are retained. The maximum possible 192-by-192 uint16 detector sum
fits uint32. All four codec arms, including the separately recorded bitplane
candidate, report zero inverse or query mismatches.

Identical kernels pass memcheck and initcheck on adversarial and real blocks.
The GPU H5 input is independently compared with HDF5 counts for the first real
smoke block; full codec/query checks are relative to GPU-loaded input. A prior
complete CUDA codec run with HDF5 filter input has identical per-block payload,
model, directory and index sizes. That agreement is an additional check, not a
full bitwise cross-decoder input proof.

CPU work in the reported GPU H5 path consists of compressed-byte IO,
chunk-header processing, configuration/orchestration and evidence aggregation.
There is no CPU codec or count-reduction performance result in these tables.
OS page-cache state is uncontrolled, so these are not cold-storage timings.

CUDA events and wall clocks bracket individually synchronized calls. Encoding
stage sums omit histogram reset and final scalar metadata transfers; the
observed encode-call wall includes them. Index construction, H5 input and
validation are separate. The GPU H5 interval includes transfers and internal
synchronization, not only LZ4/bitshuffle instructions. Do not turn sums of these
block measurements into load-to-interaction time or displayed FPS.

## How to qualify peak memory before integration

Use the {ref}`memory protocol <cuda-compressed-resident-memory>` for
each standalone codec and conversion direction. At minimum:

1. Record the device/process baseline before allocation and identify every
   live input owner. Keep other jobs and their allocations intact.
2. Capture allocated and reserved high-water marks through H5 decode, encode,
   index construction, first exact query, output handoff and release. Track
   buffers outside the primary allocator and host pinned/RSS memory separately.
3. Measure conversion while retaining source ownership. If conversion returns
   an independent packed owner, the ANS source and packed destination coexist
   at peak. Full dense expansion is not required for that overlap to matter.
4. Keep every intended full-resolution acquisition resident, vary detector
   center and radii on fresh requests, validate complete outputs and measure
   the real consumer's completion boundary. Include empty/boundary masks and
   large jumps. A bounded-block result cannot satisfy this gate.
5. Repeat load/release cycles and check the post-release baseline. Attribute
   retained pools/caches explicitly before claiming a leak or a memory saving.

The current record leaves these full-load/per-codec peak gates open. No runtime
implementation, public representation selector, source ownership rule, or
current backend qualification is changed by this documentation.

## Implementation and evidence map

| Concern | Location or status |
|---|---|
| Public representation and conversion contract | [Count representations](../api/representations.md) |
| Current CUDA ANS source/packed conversion | `src/quantem/gpu/io/backends/cuda/_ans.py` and adjacent CUDA source; distinct from these prototype kernels |
| HDF5 compressed-byte preparation and CUDA decompression | `src/quantem/gpu/io/load.py`; the exact experimental revision is retained in the record |
| Experimental direct/count and packed-byte encoders, inverse and indexed queries | Frozen source hashes in the portable record; not a current public-API benchmark |
| Per-codec process peak, complete-series residency, source-to-ready wall and presented FPS | Unqualified by this investigation |
