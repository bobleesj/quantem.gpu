# Exact residual-chunk entropy census

## Question

For the exact seven-source ADF center-8 → center-20 indexed transition, what
fraction of complete 64-stream residual work chunks use entropy modes only
(mode 64–95), such that a SIMD-uniform branch-specialized tANS loop could
possibly help?

## Fixed workload and restrictions

- Apple M5, 24 GB unified memory; seven distinct full
  `(512, 512, 192, 192)` `uint16` acquisitions.
- Exact signed ADF mask transition, no crop, binning, clipping, or count change.
- The diagnostic uses the production residual pixel plan and resident mode
  buffers. It uploads only the plan indices and reads back a small integer
  count; it must not read/copy the private mode array to CPU or allocate a
  detector-sized image/output buffer.
- Each source is measured serially and uses a transient 4,268-byte UInt32
  selection upload plus a 36-byte nine-counter readback buffer (4,304 requested
  bytes total, excluding driver overhead). This does not change persistent
  resident accounting; sampled Metal allocation is recorded separately.
- The candidate census reports both complete 64-stream decoder chunks (32 lanes
  × 2 streams/lane) and the individual 32-stream SIMD halves across all 512
  packets. Expected denominators for each source and all seven are:

  | Scope | Complete chunks per packet | Complete denominator per source | Complete denominator, 7 sources | Tail streams per packet | Tail denominator per source / 7 sources |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Decoder chunk (64 streams) | 16 | 8,192 | 57,344 | 43 | 512 / 3,584 partial chunks |
  | SIMD group (32 streams) | 33 | 16,896 | 118,272 | 11 | 512 / 3,584 partial chunks |

  Both partial tails are reported separately and excluded from the corresponding
  complete-chunk denominator. The runner asserts that per-source counts sum to
  each denominator and that aggregate entropy/mixed counts equal the sums of all
  seven per-source results.
- This is a kernel-selection diagnostic, not an image update or speed result.

## Decision gate

Do not implement a branch-specialized decoder unless the exact census shows a
substantial eligible fraction. A qualifying census is only a reason to run a
separate exact A/B/A performance experiment; it is not itself evidence of a
speedup. Preserve baseline resident bytes, full-map exact parity, malformed
stream behavior, and the high-count fixtures in that follow-up.

## Status

The seven-source GPU census completed on 2026-09-13. Of 57,344 complete
64-stream chunks, 51,437 (89.70%) contained only entropy-coded modes. At the
actual 32-lane SIMD decision scope, 110,722 of 118,272 complete groups (93.62%)
were entropy-only; the per-source range was 90.78–96.51%. Among the 3,584
partial 11-stream SIMD tails, 3,501 (97.68%) were entropy-only, but tails are
excluded from the full-chunk denominator.

The resident process loaded seven distinct sources in 21.70 seconds for this
diagnostic, observed 11,883,921,408 Metal-allocated bytes, verified that
persistent resident-byte counts and source identities were unchanged, and
released all seven residents. The census itself took 172 ms including command
completion and tiny count readback. These measurements are diagnostic overhead,
not ADF latency or a speedup result.

This mode uniformity is enough to justify a separate branch-specialized decoder
A/B/A test. It does not show that specialization will be faster; require exact
full-map parity, unchanged resident budget, malformed/high-count fixture
coverage, and a reproducible all-seven wall-time improvement before promoting
any kernel change.

The runner writes aggregate denominators, entropy/mixed counts, fractions,
partial-tail counts, resident-byte evidence, and per-source identities/counts to
the manifest after a successful run. If a runner attempt fails, it marks the
manifest `failed` and records the error type, sanitized message, and finish time
instead of leaving an attempted run looking merely planned or running.
