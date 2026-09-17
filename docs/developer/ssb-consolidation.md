# SSB consolidation and verification

This consolidation preserves the merge topology of the SSB experiments. Public
history has been sanitized to remove acquisition identities and local paths.
The original history and its old-to-new commit mapping are retained privately.
Experiment commit references are mapped to their public equivalents. Recorded
artifact checksums describe the original measured artifacts, before redaction;
they are not checksums of redacted documentation or scripts.

## Scope

- Native Metal column-IFFT stores are coalesced without changing arithmetic.
- MPS scalar row-IFFT output uses the tiled layout expected by its consumer.
  This is **not an optimization of the fit objective**.
- Correctness fixes cover mean-DP ownership, small-detector index lengths,
  indexed HDF5 export, and allocation of exact pair packs beyond retained size
  classes. Logical pair boundaries and reduction order remain unchanged.
- Timeline recording is optional and unset by default. It does not alter the
  objective, but two clock reads per command buffer remain when recording is off.

## Acceptance boundary

The frozen losses `0.14511984586715698`, `0.13808111846446991`, and
`0.13864889740943909` must not be changed to make a test pass. They are not a
sufficient gate for accumulation-order changes: `planesPerRange=4` diverges by
one float32 ULP at fixture index 8 while those three values still pass.
Widen the gate to the recorded 16-fixture set before accepting any further
accumulation reordering. See the
[candidate-batching record](../../experiments/20260916-ssb-candidate-batch/REPORT.md).

The stricter oracle comparison also records pre-existing MPS findings. A native
Metal-only pass is not evidence that those MPS findings have been resolved.

## Reproducing checks

Set `QUANTEM_SSB_PARITY_SOURCE` to an authorized local ARINA acquisition when
exporting a new fixture. Set `QUANTEM_SSB_PARITY_RUNS` to a disposable working
directory. Do not overwrite an earlier experiment's arrays or reports.

```sh
bash scripts/check_metal_ssb.sh
bash scripts/check_ssb_parity.sh --build-metal --full --metal-only
bash scripts/check_ssb_fit_trajectory.sh --build-metal
```

These are backend workflow checks, not proof that an already-installed viewer
has been rebuilt against this revision. Timings require a separate controlled
before/after measurement on the same data and geometry.
