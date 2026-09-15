# Native runtime ANS commands

Build and run the physical-Metal benchmark:

```bash
swift build -c release --product metal-runtime-ans-benchmark
.build/release/metal-runtime-ans-benchmark \
  /path/to/tilt-series-seven-native-v1 \
  /tmp/qgpu-runtime-ans-index
```

The sustained result used the default 60 interaction updates. The benchmark
loads ordinary indexed HDF5 input directly into exact in-memory runtime rANS;
it does not read or write a prepared ANS archive.

Focused parity and source checks:

```bash
/path/to/python -m pytest -q \
  tests/hardware/mps/test_streamed_h5_ans.py \
  tests/hardware/mps/test_mps_ans_counts.py \
  tests/hardware/mps/test_ans_file_workflow.py \
  tests/contracts/io/test_ans_detector_adapter.py
git diff --check
python scripts/check_profile_registry.py
```

The load timing is not described as cold I/O because operating-system source
page state was not controlled. No crop, binning, clipping, dtype narrowing,
saved two-dimensional product, or prepared ANS source was used.
