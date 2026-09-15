# Reuse-first ANS resident design

Status: architectural proposals; compact offsets and uniform-cost joint
planning now have default-off prototypes. The other designs remain proposals.
The [first measured experiment](../../experiments/20260913-apple-m5-ans-reuse-first/README.md)
saved 247.7 MB (2.09%) of resident storage but did not improve speed: compact
offsets took 68.73 ms versus 67.86/67.00 ms bracket controls. Joint planning
removed one field query and no residual pixels. Neither is promoted.
Source inspected at `cc3177d` with the existing dirty experimental work retained.
No app, data, branch, installed release, or remote was changed.

## Objective and evidence

Optimize **exact work avoided per retained byte**, with seven full uint16
512x512x192x192 acquisitions remaining resident. Do not trade accuracy, native
shape, arbitrary detector selection, or selected-DP access for a benchmark win.
Keep the existing source validity policy separate from stored counts.

Latest diagnostic all-seven return timings: residual about 43-46 ms, index
about 21-23 ms, combined about 64-68 ms. These are not displayed frame rates.
The earlier 220 measured trials did not qualify a combined kernel speedup.
Memory capacity, bandwidth, cache behavior, and dependent instructions are
different constraints. Timestamp measurements do not establish which hardware
resource causes the stall; a smaller allocation alone does not prove faster work.

Current resident accounting before diagnostic scratch is 11,877,814,048 bytes,
including 2,603,403,760 bytes of reusable exact regional sums. The diagnostic
adds 7,420,588 bytes of reusable scratch, not a dense volume. Track source,
index, outputs, scratch, and peak construction allocations separately.

Already implemented, not new proposals: signed entering/leaving detector
updates, previous-result/history bases, exact regional sums, minimum-plus-bitpack
index encoding, bounded scratch reuse, and sparse event blocks. In particular,
the encoder has both an early <=2-event fast path and a later byte-cost/slack
comparison for additional sparse events. Extending sparse coverage beyond two
events is not a new optimization.

## Recommended order

| Priority | Change | Reuse or work removed | Byte constraint |
|---|---|---|---|
| 1 | Better exact query planning over the existing sum hierarchy | Select a cheaper combination of stored sums and residual pixels instead of two independent majority votes | No additional image/volume cache; only small cost metadata |
| 2 | Block-relative source offsets | Reduce permanent metadata without touching decoded counts | About 247.7 MB theoretical saving across seven, before small alignment/terminal overhead |
| 3 | Gap-coded events and constant reductions | Skip zero-pair transitions or repeated constant contributions where profitable | Replace block representation, never retain both; remain within the total byte budget |
| 4 | Spend only part of verified savings on better reusable sums | Refine high-payoff regions and remove low-payoff redundancy | Net resident bytes must decrease; include index-build peak and time |
| Conditional | Decode once for several actually requested products | Share decoded counts among overlapping masks in the same request | Bounded extra outputs; no benefit assumed for a single ADF request |

This order deliberately starts by using existing computation more efficiently.
It does not first replace the whole codec or add a large cache.

## 1. Exact hierarchy planning

`PairedRuntimeTANSPolarPlan.makeUncached` currently chooses the majority
coefficient independently in each leaf, then in each parent. Its cost estimate
is `selectedFields + 4 * residualPixels`. That is a heuristic, not a measured
cost model. The table query and sparse/ANS residual costs vary substantially.

For one parent region, let `d[p]` be the requested signed detector-mask change,
`a` its parent coefficient and `b[l]` each leaf coefficient. The exact identity is:

```
delta image = a * parent sum
            + sum_l ((b[l] - a) * leaf sum[l])
            + sum_p ((d[p] - b[leaf(p)]) * count[p])
```

Every term is an exact integer contribution. Enumerating `a` and `b` from
{-1,0,1} includes the current solution; differences can be +/-2, as already
supported by the signed coefficient representation. Invalid pixels contribute
zero and must not influence count recovery incorrectly.

For each possible parent coefficient, choose each leaf coefficient minimizing:

```
parent query cost(a)
  + sum_l min_b [leaf query cost(b-a)
                 + sum_p_in_l residual cost(p, d[p]-b)]
```

This is a small exact search over the existing representation, not a learned
model or a general-purpose optimizer. Use the old plan and direct decoding as
explicit alternatives. A lower estimated cost is not proof of lower wall time.
Start with the existing uniform costs and tie cases; introduce source-specific
weights only if representative GPU summaries support them. A single float cost
per detector pixel and indexed field would be about 1.10 MB across seven
sources, not free; a coarser per-class model can cost much less.

The independent review found an important limit: with uniform field cost 1
and residual cost 4, choosing a non-modal leaf coefficient adds at least one
residual pixel, which costs more than the field it could remove. Joint planning
therefore mainly fixes ties, not the large residual workload. For example,
15 all-+1 leaves and one evenly split 0/+1 leaf can drop one correction field
without changing the residual count. Do not present this as a structural 4x
speedup. The meaningful hypothesis is source-aware costs based on actual
residual modes and field widths, validated on held-out requests.

Do not claim that this saves source memory. Its first purpose is better reuse
with no new stored scientific images. Any subsequent pruning/refinement of
the index is a separate measured change.

## 2. Compact offsets

There are 132,120,576 source streams across seven acquisitions. Each contains
512 scan values and currently uses a UInt32 offset plus a byte mode. The
encoder falls back to at most 1,024 payload bytes per full uint16 stream.

For a group of 32 streams, keep one UInt32 base and UInt16 relative starts.
The entire group spans at most 32,768 bytes, within UInt16. Lookup needs a
base and a relative start; the last stream's end comes from the next group
base or an explicit terminal entry. Keep mode bytes unchanged initially.

Analytical offset budget:

- Current starts: approximately 528.48 MB across seven.
- Relative starts: 264.24 MB; group bases: 16.52 MB.
- Saving: approximately 247.73 MB, before small terminal/alignment overhead.

This is a format calculation, not a measured implementation result. Validate
the maximum-size invariant at the producer boundary, cover zero-length streams
and group boundaries, and never reconstruct a permanent full UInt32 directory
alongside the compact one. DP and detector kernels need matching accessors.

## 3. Exact sparse gaps and constants

Current sparse mode 252 stores two-byte position/value events with values up
to 128, selected by byte cost and optional slack. A new gap-coded mode can
compress distances between nonzeros and preserve full uint16 amplitudes,
including 65535. Keep existing ANS/raw modes for unsuitable blocks.

The structural opportunity is fewer symbol transitions: reconstruct nonzero
events and omit proven zero contributions, rather than stepping through all
256 pairs. This is not permission to drop hot pixels or round counts. It must
also preserve selected-DP access, even if that requires decoding a bounded gap
prefix. Event scatters/atomics can dominate at higher density, so collect actual
residual-block density, amplitude, gap, and mode statistics before choosing a
gap code. Do not infer sparsity from an average compressed byte rate alone.

Constant blocks can reduce `constant * coefficient` once and reuse it across
scan positions instead of executing the pair decoder/reduction loop. Constant
and all-zero handling already exists in the representation; the opportunity is
how the detector kernel consumes it, and possibly repeated-packet metadata.
Applicability depends on the measured mode mix.

## 4. Reusable sums under a smaller total budget

The 2.603 GB index is reusable mathematical work, not a cache of fixed ADF
poses. Preserve that distinction. Changing a detector should combine exact
stored regions and decode only the uncovered correction, for arbitrary poses.

Consider selective finer leaves or an alternative small exact hierarchy only
after a query replay identifies where they eliminate the most residual work.
The measured campaign already uses 16-pixel leaves, the smallest supported
option; finer leaves require a new layout, not an existing setting change.
Any added region must pay for itself with removed redundancy or verified
metadata savings, while leaving net resident bytes lower. Different masks and
different acquisitions must be held out from the selection replay. Avoid an
index optimized only for the benchmark's one annular movement.

Minimum-based field packing is already implemented; proposing it again is not
a redesign. Full detector integral images or additional dense 4D copies are
outside the budget. A nonredundant sum basis may save storage but require more
terms per query; evaluate both rather than equating fewer fields with faster use.

One concrete nonredundant basis keeps each parent and 15 of its 16 children.
The omitted child equals the parent minus the other children, so counts remain
exact. This removes 5.88% of field streams, but not necessarily 5.88% of packed
bytes: each field has its own width. It can also turn a request for just the
omitted child from one field lookup into 16. Defer this representation until a
weighted planner can compare that penalty with source decoding. Test arbitrary
centers, radii, exclusions, and signed changes, not just the chosen drag trace.

## Designs to defer or reject

- A full decoded cache for the current 1,067 residual pixels costs about
  3.916 GB across seven sources. A 10 MB cache holds only about two detector
  pixels across all scans and sources. It cannot be called boundary reuse.
- Naively shortening streams from 512 to 128 scans adds about 1.982 GB of
  offset/mode entries, plus up to 0.793 GB of extra two-byte entropy headers
  in an all-entropy comparison, before other padding/rate changes.
- More interleaved ANS states can shorten each dependency chain but retain
  the same total symbol count and add stored states. The present kernel already
  processes independent detector streams per lane. New intra-stream interleaving
  is different, but neither a free change nor proof of 4x throughput.
- Larger lookup tables, loop unrolling, and a blanket extra raw buffer are not
  first choices after the recorded negative experiments.

Interleaved shared-stream ANS and GPU packed loads are established techniques,
but adapting them still needs explicit state/header accounting and a matching
encoder. See [Giesen, Interleaved entropy coders](https://arxiv.org/html/1402.3392v1).
That paper is not evidence of the proposed viewer's performance.

## First prototype gates

1. Collect small GPU summaries during preparation: mode counts, encoded bytes,
   nonzero/gap/amplitude distributions, indexed widths, and representative
   residual incidence. No dense readback or per-pointer synchronous profiling.
2. Compare old and proposed plans on the same masks with exact decomposition
   checks before changing storage. Record fields reused, residual streams
   decoded, estimated cost, planning time, and actual all-seven GPU/return time.
3. Implement offset compression independently and confirm actual allocated
   bytes, random DP retrieval, payload boundaries, and load/encode overhead.
4. Test one selected gap/constant representation, replacing old blocks under a
   hard total-byte limit. No simultaneous old/new resident outside separately
   recorded bounded conversion scratch.
5. Run at least 20 shuffled matched repetitions per candidate on small/large
   BF, ABF, and ADF changes, jumps, reversals, and held-out poses. Separate
   first-use/build costs from steady reuse. Keep full u16 and source validity.
6. Require full-map frozen-reference parity, independent count/DP checks and
   malformed/high-count fixtures. Native presentation testing follows a
   qualified backend change; 8.33 ms remains a target, not a promised result.

No candidate is promoted merely because it compresses more or wins one trace.
