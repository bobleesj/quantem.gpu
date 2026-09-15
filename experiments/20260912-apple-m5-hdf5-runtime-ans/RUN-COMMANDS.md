# Runtime ANS commands

This run used the physical Apple M5 GPU through the MPS backend. The source
identity in `result.json` is the SHA-256 of the ordered member-content SHA-256
values; private filenames are intentionally not retained.

Focused parity gate:

```bash
PYTHONPATH=src pytest -q \
  tests/hardware/mps/test_streamed_h5_ans.py \
  tests/hardware/mps/test_ans_file_workflow.py \
  tests/hardware/mps/test_mps_ans_counts.py \
  tests/contracts/io/test_ans_detector_adapter.py
```

Registry and source checks:

```bash
python -m ruff check \
  src/quantem/gpu/io/_streamed.py \
  src/quantem/gpu/io/load.py \
  src/quantem/gpu/io/backends/mps/_streamed.py \
  tests/hardware/mps/test_streamed_h5_ans.py
git diff --check
python scripts/check_profile_registry.py
```

The full-size timing harnesses were inline Python programs using `io.load` on
the seven-file HDF5 acceptance folder, `MPSStreamedSeries`, `perf_counter`, and
NumPy percentile calculations. No environment tuning flag, crop, bin, dtype
narrowing, saved ANS source, or precomputed 2D image cache was used.
