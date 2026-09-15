# Seven-source entropy-stream count

Status: failed (incomplete launch attempt). This is a memory-feasibility
census, not a speed benchmark. The first runner attempt was interrupted while
waiting for the resident-loop ready event; it produced no census records,
timings, or resident-release evidence. The corrected resident-op rerun is
tracked separately under `20260913-apple-m5-ans-entropy-census-resident-op-retry`.

## Question

How many resident streams would a checkpointed decoder need to cover, and what
are the exact 1-checkpoint and 3-checkpoint sidecar byte estimates under the
current compact-offset layout?

## Protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` `uint16` acquisitions.
- Leaf16/radial1 polar index, compact offsets enabled, no cropping/binning.
- Two masks: all valid detector pixels and the exact ADF 8→20 changed-pixel
  set. Every source reports selected/invalid pixels and total/entropy stream
  counts. Entropy is the existing mode range 64–95.
- The Metal diagnostic reduces 128 per-stream predicates in threadgroup memory
  and emits one atomic per threadgroup. It allocates only the existing selected
  pixel list and a 4-byte result counter; it does not produce detector maps.
- Preserve source identity, allocation, exact count totals, and all-seven
  release evidence. Checkpoint bytes are estimates, not allocations; no speed or
  construction-peak claim is made.

## Failed attempt

No stdout records were received before the operator stopped this attempt.
That is not evidence that the seven-source load failed or succeeded. No mode
counts or performance result were produced, and clean release was not
verified. See `results/failure.json` and the separate corrected-run record.
