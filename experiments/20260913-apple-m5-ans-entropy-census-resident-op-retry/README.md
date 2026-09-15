# Seven-source entropy-stream census — protocol retry

Status: completed census only. This is a memory-feasibility census, not a speed benchmark.
It repeats the resident-loop census after the first run exposed a Python
post-release call-site typo. The first attempt's complete raw protocol log and
failure record remain in `20260913-apple-m5-ans-entropy-census-resident-op`.

## Question

How many entropy-coded streams would one or three exact midpoint checkpoints
need to cover for the all-seven ADF delta and full detector support, and do
those estimates fit under the current sampled Metal allocation ceiling?

## Protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` `uint16` acquisitions.
- Leaf16/radial1 polar index; compact offsets enabled; no crop, bin, clip, or
  count conversion.
- Wait for the resident-ready event, send the dedicated `entropy_census`
  operation, validate all 14 per-mask/source records and mask hashes, then
  explicitly release all seven residents.
- Diagnostic mode counting uses a 128-lane threadgroup reduction. It does not
  compute detector maps and it is outside the ADF timing path.
- Report the aligned four-byte checkpoint estimate as the conservative size;
  the three-byte figure is only a codec-capacity lower bound requiring an exact
  normalized bit-reader restart.
- Allocation values are ready-state and sampled diagnostic values only; they
  are not a checkpoint-construction peak measurement.

## Interpretation

This run sizes a possible repeat-interaction cache. It does not measure a
speedup and does not establish that a cache fits during construction.

## Result

Complete, census only. The seven full sources had **11,630,087,982 resident
bytes** and **11,636,195,328 Metal bytes** at readiness. The diagnostic sampled
**11,636,342,784 Metal bytes**, leaving **247,578,624 bytes** below the fixed
Metal ceiling at that sample. All seven sources had distinct identity hashes,
and the release record confirms all seven were released.

| Requested stream set | Requested detector pixels | Entropy streams / all streams | One `UInt32` checkpoint | Three `UInt32` checkpoints | Headroom after one / three |
|---|---:|---:|---:|---:|---:|
| ADF 8→20 changed-mask upper set | 6,271 | 22,357,445 / 22,468,096 | 89,429,780 B | 268,289,340 B | +158,148,844 B / −20,710,716 B |
| All valid detector pixels | 36,864 | 131,061,170 / 132,106,240 | 524,244,680 B | 1,572,734,040 B | −276,666,056 B / −1,325,155,416 B |

The checkpoint figures are sizing estimates, not allocated buffers. The full
detector checkpoint clearly exceeds the current headroom. One checkpoint for
the entire 6,271-pixel ADF mask fits at this ready-state sample; three do not.
The current polar plan processes **1,067 residual pixels** for this ADF change,
so an intentionally conservative upper bound of one `UInt32` checkpoint for
every residual stream is 15,296,512 B; three are 45,889,536 B. That upper bound
uses all residual streams, even non-entropy modes, and still does not prove
construction-peak feasibility or estimate cache-management overhead.

No decoder update was timed, so this diagnostic demonstrates **no speedup**.
It only makes a bounded one-checkpoint repeat-update prototype plausible for
the current ADF residual set. It does not justify caching the full detector or
three checkpoints for the full ADF mask under the current budget.
