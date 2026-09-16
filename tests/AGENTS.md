# Disposable test data

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
