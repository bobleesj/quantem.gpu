# Seven-source entropy-stream census (resident-loop command)

Status: failed after data collection. The census emitted all 14 mode-count
records and the seven-source loop released cleanly, but a Python post-release
call-site typo raised `TypeError` before the runner could finalize its summary.
The raw evidence and failure record remain here; a corrected runner repeat is
tracked in `20260913-apple-m5-ans-entropy-census-resident-op-retry`.

## Question

How many entropy-coded streams would one or three exact midpoint checkpoints
need to cover for the all-seven ADF delta and the full detector support? Do
those byte estimates fit under the already established Metal allocation
ceiling at census time?

## Protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` `uint16` acquisitions.
- Leaf16/radial1 polar index; compact offsets enabled; no crop, bin, clip, or
  count conversion.
- The runner waits for resident readiness, sends `entropy_census`, validates
  one requested-mask hash across all sources, validates source-specific
  validity-mask exclusion, selected stream arithmetic, and all 14 records.
- The count uses a 128-lane threadgroup reduction and one global atomic per
  group. It computes no detector map and does not affect the timed ADF path.
- Preserve distinct source identities, resident allocation, sampled census
  allocation, and explicit release of every source. A 4-byte aligned
  `(decoder state, logical bit position)` estimate is the conservative sizing
  case; 3-byte packing is only a theoretical lower bound and requires a
  normalized restart reader.
- The diagnostic allocation sample is taken with the selected-pixel buffer,
  result buffer, pipeline, and encoded command present. It is not a full peak
  memory trace and excludes construction peak for any future checkpoint sidecar.

## Interpretation

This run can inform checkpoint-cache sizing only. It does not demonstrate
that checkpoint creation fits during construction, nor that checkpointed
decoding improves first-update or repeated-update latency. No speedup is
claimed from this census.

## Failed runner attempt

The raw JSONL contains the ready record, both masks for all seven source
identities, and `all_released: true`. The runner then called the event reader
with an obsolete argument list and raised
`unsupported operand type(s) for +: 'float' and 'str'` while waiting for the
already-issued release event. The captured counts are retained but not treated
as a successful finalized result; use the corrected retry manifest for the
validated summary.
