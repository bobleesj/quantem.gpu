# Seven ANS acquisitions: CUDA detector queries

Seven complete 512 × 512 × 192 × 192 uint16 acquisitions were loaded together
and queried at native resolution on an RTX PRO 6000 Blackwell Workstation
Edition. Their encoded allocations total 7.759 GiB; the logical input is
126 GiB. Each acquisition remains independent. Query workspaces are additional.

The residual decoder now groups up to sixteen scan-block warps per CUDA block,
instead of four. The number of decoded pixels and scan positions is unchanged.
Small chunks use fewer warps. The spatial-index kernel and stored format are
unchanged. The byte reader buffers aligned four-byte words entirely within
each stream, with scalar reads for unaligned prefixes and short tails. This
reduces byte-load dependencies without changing the encoded representation.

## Matched results

Baseline: `5c6a2ec6`, four residual warps. Candidate: the same source plus the
sixteen-warp launch, buffered byte reader and malformed-stream checks. The runner compiles both
kernel sources in one process and gives each variant its own query state.
Baseline, candidate and repeat-control queries are interleaved in seeded random
order at every mask. This matters: changing background GPU load produced false
gains in preliminary runs measured in separate phases.

The table reports GPU-event time, in milliseconds. Baseline is the median of
paired controls; speedup is the median paired ratio. Seven warm repetitions
follow the first measured trajectory. Other GPU jobs were active; these are
contended measurements, with variable clocks and caches.

| Acquisitions | Detector operation | Baseline | Candidate | Paired speedup |
|---|---|---:|---:|---:|
| 1 | Small BF | 0.512 | 0.573 | 0.90× |
| 1 | Large BF | 0.584 | 0.598 | 0.98× |
| 1 | Displaced DF | 1.471 | 0.795 | 1.86× |
| 1 | ADF | 1.296 | 0.844 | 1.55× |
| 7 | Small BF | 2.865 | 1.611 | 1.79× |
| 7 | Large BF | 3.371 | 3.049 | 1.12× |
| 7 | Displaced DF | 9.164 | 3.382 | 2.65× |
| 7 | ADF | 9.637 | 4.855 | 1.98× |
| 7 | ADF movement | 11.302 | 4.062 | 2.63× |

Large BF uses radius 70 px; DF uses radius 22 px at offset (25, 40) px;
ADF spans radii 40–80 px. The suite also includes translations, resizing,
large jumps, detector-edge intersections, empty masks and full masks.

The single-source small-BF case regresses by 0.061 ms; this change targets
large and joint detector queries. With seven sources, ADF takes 5.765 ms median
through the synchronous Python query; copying its exact integer images to the
host takes another 0.620 ms. Its worst sample is 5.146 ms of GPU time in this run. These results do not establish
120 Hz browser rendering. The complete suite separately records loading,
session preparation, GPU query, Python query and host-copy time. The repeated
HDF5 read/encode pass took 4.872 s; storage caches were not evicted.

## Profile and correctness

Nsight Compute attributes the original seven-source ADF cost to residual
decoding: 8.64 ms versus 0.436 ms for the spatial index. With larger blocks alone,
residual time is 4.49 ms and index time is 0.433 ms. Residual compute throughput
rises from 31% to 63%, with achieved occupancy near 90% in both profiles.
Long-scoreboard waiting falls from 28.2 to 9.4 cycles per instruction. This is
consistent with improved locality across scan blocks, rather than reaching the
device's DRAM bandwidth limit. Adding aligned byte buffering reduces residual
time further to 4.13 ms, with 77% compute throughput and 93% achieved occupancy.
The index remains 0.435 ms. Profiling used unfixed clocks and caches.

Every output pixel in all 22 mask states agrees exactly with the frozen
implementation, for both one and seven sources. Independent uint64 reductions
of decoded patterns check distant positions and codec boundaries. No values,
detector pixels or scan positions were omitted. Both paths use the same recorded
median hot-pixel correction.

The focused CUDA suite passes 56 tests, including uint8/uint16 counts,
uint64 sums above 2³², odd geometry, partial chunks, seven-source CoM and live
partial-scan queries. A new corruption test fails on the frozen baseline and
passes after the fix: sparse event ordering and byte extents, constant extents,
and empty-stream extents are checked before publishing detector or moment
results. Failed queries also invalidate dependent queued images. The public
CoM result now retains the acquisition axis instead of trying to reshape seven
images into one. Nine memory-sanitizer cases pass with zero errors:
all four payload alignments, mixed codecs, odd lengths, short tails, truncated
headers, raw pairs, renormalization bytes, escaped counts and trailing bytes.

## Reproduce

Use the same complete acquisitions for both variants, with a private output
directory. Extract the baseline kernel source from the revision above, then run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/benchmark_detectors.py \
  tilt-series/*_master.h5 --reference-source reference-streamed.cu \
  --reference-warps 4 --reps 7 --contention contended \
  --json-out detector-comparison.json
```

The JSON records source dtype, correction policy, output hashes, source-code
hashes, launch geometry, individual observations and p50/p95 timings. Raw
artifacts and acquisition identities stay in private experiment
`ans-detector-seven-20261025`. This qualifies the tested CUDA source checkout;
other CUDA architectures, MPS and browser paint rates were not benchmarked here.

The user API is unchanged:

```python
from quantem.gpu import detector, io

tilts = io.load(files, backend="cuda", stack=False)
session = detector.prepare(tilts)
images = session.masked_sum(mask, output="native")
patterns = session.frame(scan_index, output="native")
```

For seven inputs, `images` has shape `(7, 512, 512)` and `patterns` has shape
`(7, 192, 192)`. Both remain on the GPU; encoded is the default representation.
