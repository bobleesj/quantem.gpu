# Disposable test data

## GPU-first numerical verification

- For GPU optimization, freeze the known-good GPU result and compare the
  candidate on the same backend, device, inputs, calibration, and precision.
  Preserve existing pins; never refresh them automatically on a mismatch.
- Require bit-exact parity for changes advertised as bit-exact. Cross-backend
  comparisons use their separately documented tolerances, not a new CPU gold.
- Do not launch full-size CPU reconstruction references during routine checks.
  CPU numerical checks, when useful, must be tiny and bounded. Large CPU-oracle
  runs require an explicit user request and the opt-in flag/environment switch.
  CPU orchestration, input validation, hashes, and comparisons remain permitted.
- Run `bash scripts/check_ssb_parity.sh --build-metal` for the frozen GPU fit
  gate. Run `bash scripts/check_metal_ssb.sh` for native reconstruction and
  saved-result workflow checks. These do not resolve recorded strict MPS oracle
  findings or replace the broader gate required before accumulation reordering.

## Generated artifacts

- Write every generated `.qem`, HDF5, raw or NumPy export to pytest's `tmp_path`.
  Do not save beside original acquisitions or into tracked fixture folders.
- The session fixture redirects Python and subprocess temporary files into
  disposable storage. Do not override `TMPDIR` to a persistent folder.
- Keep small timing/parity reports separately only when needed. Do not retain
  generated arrays for successful or failed tests by default.
- Standalone benchmarks need an explicit temporary-directory lifecycle; pytest
  cleanup does not cover a command launched outside pytest. For native UI runs,
  use Live4DSTEM's `scripts/run_test.py` supervisor and staged inputs.
- Never clean source datasets, intentional user saves, or Codex conversations.
