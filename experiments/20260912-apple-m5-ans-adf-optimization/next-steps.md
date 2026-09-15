# Bounded exact polar interaction index

## Evidence boundary

This is a proposed exact interaction index for the paired runtime tANS resident.
It is not an implemented Metal path or a measured speedup. The CPU geometry
census in `polar-planner-census.json` reports the existing CUDA planner's cost
proxy. It does not predict Metal wall time, GPU time, presentation rate, or load
overhead.

Every detector count remains exact. Index fields contain integer sums of the
original valid detector counts. Residual pixels make the field decomposition
exact for each requested signed mask delta. There is no detector crop, bin,
clipping, subsampling, cached virtual image, or approximate circular mask.

## Existing implementation references

- `src/quantem/gpu/_compact/paired.py::polar_layout` defines the radial-angular
  permutation using the sort key `(floor(radius**1.5 / 45), angle, radius)`.
- `src/quantem/gpu/_compact/paired.py::polar_planner` decomposes signed masks
  into root fields, leaf corrections, and exact residual pixels. The unweighted
  option order is zero, positive one, negative one, so zero wins ties.
- `src/quantem/gpu/_compact/streamed.py::field_count` defines the 8-pixel and
  32-pixel field hierarchy.
- `src/quantem/gpu/_compact/kernels/paired.cu::pm_fields` constructs exact field
  sums from native counts and the validity mask.
- `pm_field_sizes`, `pm_pack_fields`, and `pm_field` select the exact raw or
  minimum-plus-delta bit width, pack fields, and recover one field value.
- `pm_index_sum` loads field descriptors cooperatively and adds signed field
  values to the virtual detector image.
- `pm_weights` is the optional resident-stream decode-cost model. The initial
  Metal port should use the unweighted planner so geometry is isolated first.

## Proposed Metal representation

For a 192 by 192 detector, retain 576 leaves of 64 detector pixels and 36 roots
of 16 leaves, for 612 exact fields. A fixed permutation contains 36,864 Int32
entries, or 147,456 bytes. It may be generated deterministically in Swift or
stored as a hash-bound package resource.

Construct field sums while each original HDF5 dense window already exists. A
32,768-scan window needs:

```text
32,768 scans * 612 fields * 4 bytes = 80,216,064 bytes
```

This approximately 76.5 MiB field-sum buffer is transient and must be discarded
after packing. For each 512-scan packet and field, store a bit width and choose
between raw values and one UInt32 minimum followed by packed unsigned deltas,
using the same size comparison as `pm_field_sizes`.

The existing CUDA capacity record reports 6.7 GiB of index storage for 69
acquisitions, approximately 104 MiB per acquisition and 728 MiB for seven.
These are cross-backend planning estimates, not measured Metal allocation.
Metal admission must use the actual packed byte count and include offsets,
widths, the permutation, alignment, transient field sums, and concurrent-load
overlap.

An uncompressed UInt32 prototype would require:

```text
612 fields * 262,144 scans * 4 bytes = 641,728,512 bytes/acquisition
seven acquisitions = 4,492,099,584 bytes
```

That is not the intended bounded representation and must not be used to claim
the packed-index memory or performance result.

## Planner and query

For each source, first intersect the requested binary mask with that source's
detector validity mask. Form the signed difference from its previous complete
mask. Apply `polar_planner` exactly:

1. Each 64-pixel leaf selects its majority coefficient from zero, positive one,
   and negative one.
2. Pixels differing from that coefficient become signed residual corrections.
3. Each 16-leaf root selects the majority leaf coefficient.
4. Leaf coefficients are stored relative to the root coefficient.

The query should dispatch one 128-thread group per 128 consecutive scans within
a 512-scan packet. Threads cooperatively load each selected field's packed word
offset, minimum, width, and signed coefficient into threadgroup memory. For the
151 fields selected by the 20-pixel census, four 32-bit values per field require
2,416 bytes of threadgroup storage. Each thread extracts its scan's exact field
value and accumulates signed contributions modulo UInt32. The existing paired
tANS detector path then processes residual detector pixels. The previous mask
may advance only after both stages complete without a stream or bounds failure.

The CPU census measured these geometric decompositions:

| Transition | Changed pixels | Fields | Residual pixels | Existing planner cost / direct cost |
| --- | ---: | ---: | ---: | ---: |
| ADF center 1 | 568 | 0 | 568 | 1.000 |
| ADF center 8 | 4,937 | 74 | 1,631 | 0.334 |
| ADF center 20 | 11,156 | 151 | 1,964 | 0.179 |

The residual totals include exact corrections outside the original symmetric
difference when a majority field is selected. These ratios use the CUDA
planner's proxy of one unit per field plus four units per residual pixel. They
are not measured Metal timings. The current grouping offers no geometric saving
for the one-pixel transition.

## Required gates

- Compare the generated permutation and field membership with the referenced
  Python implementation, including padded leaves and root boundaries.
- Round-trip every packed field value exactly for zero-width, raw-width,
  minimum-plus-delta, UInt32 limit, truncated, and invalid-offset cases.
- Verify full 262,144-value detector maps for the complete A1/candidate/A2 mask
  sequence on all seven acquisitions.
- Compare every tested full map with the independent runtime-rANS reducer after
  applying the same per-source validity policy.
- Retain original-HDF5 sampled diffraction and high-count sentinel checks. A
  promotion claim also requires the established full-volume or frozen-product
  oracle rather than only in-process agreement.
- Measure field construction, size calculation, prefix, packing, resident-ready
  time, query index time, residual time, and complete query time separately.
- Record first load, metadata-assisted reopen if added, transient peak, retained
  bytes, application allocation, teardown residual, memory pressure, and swap.
- Run matched A/B/A trials with identical masks, source identity, executable and
  Metal resource hashes, warmup policy, queue topology, and presentation policy.
- Reject the candidate if exactness fails, retained or peak memory exceeds the
  declared ceiling, loading becomes unacceptable, or thick-delta improvement
  does not replicate across sources and controls.
