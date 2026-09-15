# All-stream ANS mode census

Status: failed. This was a memory-feasibility census, not a speed benchmark.
Its global-atomic mode histogram serialized too heavily for the full detector
support and was stopped before it emitted usable counts or a release record.

## Why

The current center-20 update is about 59 ms p50. It repeatedly decodes a long
dependent tANS chain per selected detector pixel/scan packet. A parallel
checkpointed decoder may shorten that chain, but a checkpoint sidecar must fit
under the existing 11,883,921,408-byte sampled Metal-allocation ceiling. This
census measures how many streams are entropy-coded before designing that
sidecar. It uses the established compact-offset resident layout, whose earlier
measurement freed about 247.7 MB, but does not assume that entire amount is
available at construction peak.

## Protocol

- Apple M5, seven distinct original uint16 acquisitions, full
  `(512, 512, 192, 192)` shape and no crop/bin/clip.
- Leaf16/radial1 index and compact offsets; all valid detector pixels counted
  for each source; a separate ADF 8→20 changed-pixel mask is counted.
- The only additional allocations are the diagnostic histogram buffers and
  selected-pixel list. Counts are reduced on Metal to 256 mode bins/source;
  no detector image maps are computed.
- Preserve per-source modes, selected/invalid pixel counts, exact source IDs,
  resident/Metal allocation, and explicit release of all seven sources.

For each checkpoint cadence, report both 3-byte packed and 4-byte aligned
state+bit-position estimates. These are estimates only: the census does not
allocate checkpoints and does not demonstrate a safe construction peak.

## Failure

After about 108 seconds in the contended histogram, the run was interrupted
before `ans_opt_mode_counts`; there is no valid count or timing result and no
verified all-seven release. This rejected diagnostic is retained so the global
atomic approach is not repeated. A threadgroup-reduced counter was used in the
corrected follow-up census.
