# Reuse-first ANS prototype

Status: completed; exact memory saving, no qualified speedup. Both prototypes
remain default-off. Defaults and installed app are unchanged.

## Hypotheses

1. Source UInt32 offsets can be replaced during GPU consolidation with one
   UInt32 base per 32 streams and UInt16 relative starts, retaining exact
   random DP and detector access while lowering seven-source allocation by
   about 247.7 MB. No second full resident offset table is allowed.
2. Joint selection of parent and leaf coefficients can remove unnecessary
   sum queries at the same residual count, without another scientific-image
   cache. With the current uniform cost, large speedups are not expected.

The seven originals remain full uint16 512x512x192x192. No clipping, binning,
cropping, or source-count change. Existing validity exclusions stay unchanged.
ANS and exact regional sums remain the representation; this is not a dense
cache or fixed ADF image lookup.

## Measurement protocol

- CPU planner proof over exact signed mask reconstruction and existing fallback.
- Synthetic GPU compact-directory conversion and decoder fixtures, including
  uint16 high values, all modes, group boundaries, terminal and malformed data.
- Same executable control/compact/control full seven-source loads. Resident
  data stay allocated throughout each arm. OS page state is uncontrolled;
  these runs do not establish cold loading performance.
- Twenty interleaved measured trials per greedy/joint planner, after warmup,
  for isolated index, residual, and combined signed ADF changes.
- Twenty BF/ABF/ADF masks, including resizing and large center jumps, replayed
  against 140 frozen independently checked full-map hashes.
- Thirty-five selected scan positions per source, including record and stream
  boundaries, compared exactly across both offset layouts.
- Record resident bytes, sampled device allocation, and teardown. No UI FPS
  claim from backend timestamps. Full construction peak is not inferred from
  the ready-state sample.

Run `run.py --help` for paths. Recheck the retained evidence without GPU work:

```sh
python3 experiments/20260913-apple-m5-ans-reuse-first/summarize.py \
  experiments/20260913-apple-m5-ans-reuse-first/results \
  --reference experiments/20260912-apple-m5-ans-adf-optimization/polar16-radial1-aga.jsonl
```

The driver rereads original HDF5 files; its catalog directory only stores
existing small discovery metadata, not an encoded resident checkpoint.

## Scope filter from existing evidence

The earlier `polar16-radial1-aga.jsonl` mode census counted 290,816 streams per
source for a one-pixel ADF move. Each source had zero all-zero, constant, or raw
blocks, and only 35-88 sparse blocks. Most were entropy blocks. This is a
different transition from the larger stage-isolation move, but it argues
against spending this pass on constant-mode shortcuts. A gap-code redesign
still needs actual nonzero/gap statistics; mode census alone is insufficient.

## Result

### Retained memory

| Ready-state metric | Control A1 / A2 | Compact offsets | Change |
|---|---:|---:|---:|
| Accounted seven-source resident bytes | 11,877,814,048 | 11,630,087,982 | -247,726,066 (-2.09%) |
| Sampled Metal allocation bytes | 11,883,921,408 | 11,636,195,328 | -247,726,080 |
| Exact regional index bytes | 2,603,403,760 | 2,603,403,760 | unchanged |

The source identity arrays agree across all three arms and contain seven
distinct acquisitions. Each compact offset directory occupies 40,108,038 bytes
instead of 75,497,476 bytes. The GPU writes one UInt32 base per 32 streams and
UInt16 relative starts during consolidation, rather than retaining a second
full offset table. Payloads, modes, scientific counts, and index are unchanged.
After explicit release, all three arms report all resident storage released;
sampled remaining Metal allocation is 5,701,632 bytes. These samples do not
establish peak construction memory or zero temporary allocations.

### Difficult ADF transition

All-seven backend return medians in milliseconds, 20 measured trials per row.
The transition is `adf-center-8` to `adf-center-20`; each arm interleaves greedy
and joint plans using seed 1713, keeping all seven sources allocated.

| Offset arm | Planner | Index only | Residual only | Combined |
|---|---|---:|---:|---:|
| Control A1 | Greedy | 22.37 | 46.94 | 67.86 |
| Control A1 | Joint | 23.46 | 46.84 | 67.75 |
| Compact | Greedy | 21.84 | 48.30 | 68.73 |
| Compact | Joint | 23.25 | 48.23 | 71.11 |
| Control A2 | Greedy | 23.12 | 46.92 | 67.00 |
| Control A2 | Joint | 23.12 | 46.82 | 69.35 |

Compact/greedy combined p95 was 73.77 ms versus 72.53/71.35 ms in controls.
Compact offsets save capacity but did not pass the no-slowdown hypothesis:
combined median was 1.3-2.6% slower than the bracket controls. This is one
session, not evidence that those percentages are stable across machines or
sessions. The extra address lookup is a plausible cost, not a demonstrated
hardware bottleneck.

The joint planner uses 370 stored fields instead of 371, but still decodes
1,067 residual pixels per source. It introduces no extra scientific-image
cache. Uniform-cost joint planning mainly resolves ties; it did not remove
the dominant residual work and has no repeatable combined improvement here.
CPU proofs cover 316 exact decompositions, including signed +/-2 field
coefficients. The 300 random cases did not improve aggregate proxy cost.

### Exactness and limits

- 2,520 full detector-map observations agree with 140 independently checked
  frozen reference maps across 20 BF/ABF/ADF masks and all three offset arms.
- 245 unique `(source, scan)` diffraction positions, including boundaries,
  agree across all three arms (735 observations, not a full-volume audit).
- 882 exact source transitions in isolated-stage checks, including warmups.
- Independent synthetic GPU fixture: all 32,768 uint16 values match, including
  high counts, every block mode, packet boundaries, terminal offsets, signed
  detector sums, and index partials. Invalid block spans and record receipts
  are rejected.
- The 20-mask matrix has only two cycles per mode and is a correctness/replay
  smoke test, not a stable performance distribution. No native UI was driven;
  these backend timings are not displayed FPS and do not meet 120 FPS.
- Diagnostic preparation took 22.21 / 21.76 / 21.00 seconds for A1/B/A2.
  It includes building the regional index, not just HDF5 decompression. Page
  state was uncontrolled and system swap was already in use; no cold-I/O or
  load-speed improvement is claimed.

### Reproducibility and failures retained

The benchmark was built from dirty `cc3177d745f49a1811ce8cf80703b933302fad14`;
the manifest records executable, actual packaged shader, and a bounded final
source snapshot. Earlier unrelated work remains untouched. A tracked Git diff
hash alone is insufficient because these experimental sources are untracked.

Preflight found and fixed a throwing Metal API call, an unspecialized function
constant in the synthetic codec, and a Swift fixture type-checking issue.
Their failed logs remain alongside the successful final fixture log. A final
Python 3.9 `hashlib.file_digest` reporting failure happened after all GPU arms
completed and cleanly released. It did not interrupt any measurements. The
driver now uses streaming SHA-256, and the retained raw outputs were validated
again by the stricter summarizer against frozen hashes. No GPU rerun is implied
by that reporting repair.

Decision: retain both as opt-in experiments, not a production performance
change. The next useful work must reduce the 1,067 residual streams themselves:
measure actual residual gaps/modes and source-specific query costs before
choosing a new exact representation or additional reusable regions. Do not
spend this small memory saving on an unbounded decoded cache.

No promotion, release, push, or merge was performed.
