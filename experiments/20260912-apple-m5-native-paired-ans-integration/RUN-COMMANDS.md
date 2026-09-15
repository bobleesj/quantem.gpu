# Native paired-ANS integration commands

Build and run sampled backend parity on the physical Apple GPU:

```bash
swift build -c release --disable-sandbox \
  --product metal-paired-runtime-tans-benchmark
.build/release/metal-paired-runtime-tans-benchmark \
  INPUT_HDF5 INDEX_DIRECTORY
```

Run the independent CPU codec oracle:

```bash
python -m pytest -q \
  experiments/20260912-metal-paired-runtime-ans-prototype/test_paired_runtime_ans_oracle.py
```

Build the sibling native application and run the hook-driven seven-acquisition
journey. The application used a local package dependency for this unpublished
integration candidate.

```bash
cd ../Live4DSTEM
swift build -c release --disable-sandbox
DIAG=1 python3 Tests/NativeUI/drive_folder.py \
  --exe .build/release/Live4DSTEM \
  --folder ACCEPTANCE_FOLDER \
  --out OUTPUT_DIRECTORY \
  --count 7 --compare
```

No environment tuning flag, crop, bin, dtype narrowing, saved ANS file, or
precomputed image cache was used. The file-system page state was not controlled,
so the recorded loading observations are not cold-I/O measurements.
