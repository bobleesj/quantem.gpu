# Disposable test data

## Compressed acquisition acceptance

- For acquisition IO changes, use `python scripts/check_ans_io.py` on the
  physical requested backend. Supply the local Zenodo collection via
  `--zenodo-root` or `QEM_ZENODO_ROOT` for the full real-data gate.
- Missing hardware/data, skipped/xfail tests, deselection and empty execution
  are not passes. Retain the generated table with unsupported/untested cells
  visible; do not substitute an older installed package or a different commit.
- Python MPS/Metal tests do not certify native Swift/Metal or application UI.
  CUDA requires its own physical-device run. Do not infer raw EMPAD2/G3 support
  from processed float exports, DM3 from DM4, or Velox events from EMD arrays.
- Keep reports outside the repository and use disposable exports. See
  `docs/maintainer/ans-io-acceptance.md` for the exact scope and commands.

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
