# Float32 .qem across native Metal, Python MPS and CUDA

The `float32-bit-lanes-rans-v1` profile now reopens as encoded GPU storage in
Python MPS and CUDA. This is new backend support, not a replacement compression
format. Native Metal already reads and writes this profile.

## Scientific and memory contract

- Original IEEE bits, including signed zero, NaN payloads and subnormals, survive
  GPU loading and source-independent `.qem` re-export. No float-to-count cast.
- Metadata and embedded source documents survive copying. Raw reads preserve
  measurements; scientific display products apply a saved mean-dark plane once.
- Point DPs, binary detector masks, mean/selected DPs and mean-subtracted CoM
  operate on the accelerator. Empty/invalid CoM frames stay NaN, not fake zeros.
- Decoded windows are at most 512 frames / 32 MiB. Additional working tensors
  are bounded too; 32 MiB is not a total-process or total-GPU-memory claim.
- No full decoded acquisition, CPU science fallback, scan bin or crop.
- Detector geometry is currently 128 by 128 and storage dtype float32. This
  qualification does not extend to float64, arbitrary detector dimensions,
  raw float-source ingestion by Python, SSB, or native presentation cadence.

## Reproduction

Use the source commit in [manifest.json](manifest.json). CUDA uses CuPy 14.0.1
on RTX PRO 6000 Blackwell; MPS uses PyTorch 2.9.0 and native Metal kernels on
Apple M5. The full input is public Zenodo 15987625 WSe2, scan 128 by 128,
detector 128 by 128, float32. Its SHA-256 is recorded in the manifest.

Build `EMPADSourceParity` from `tests/hardware/metal/swift_original_packing`
in release mode. In an owned temporary directory, export a native `.qem` and
frozen GPU products using the public raw input:

```bash
EMPAD_TEST_METAL=1 EMPAD_TEST_BUDGET=5368709120 \
EMPAD_TEST_SAVE_QEM="$scratch/public-float.qem" QGPU_ORIGINAL_READ_AHEAD=0 \
  "$parity_executable" "$public_raw" "$scratch/native-reference.bin" 0,8191,16383

PYTHONPATH=src python tests/hardware/float_qem_compare.py \
  "$scratch/public-float.qem" "$scratch/native-reference.bin" "$report" --backend mps
# Transfer the disposable .qem and native-reference.bin* files to the CUDA host;
# repeat with --backend cuda. Keep reports; remove generated arrays afterwards.

QEM_TEST_BACKEND=mps PYTHONPATH=src pytest -q tests/hardware/test_float_qem_resident.py
QEM_TEST_BACKEND=cuda PYTHONPATH=src pytest -q tests/hardware/test_float_qem_resident.py
```

Each backend compares every decoded float word, in bounded windows, with the
native export's logical SHA-256. Three explicit point DPs also match the native
GPU oracle. Four virtual-detector images (BF, ABF, ADF and full detector) must
match native Metal bit-for-bit. Mean DP uses rtol 2e-5 / atol 2e-4; CoM uses
rtol 2e-4 / atol 3e-4 pixels. These tolerances were set before measurements.
The maximum tested decoded buffer is 16 MiB, because this native file uses
256-frame chunks. A 513-frame decoded request is rejected before allocation.

Tiny synthetic fixtures cover entropy/literal/constant/zero streams, special
IEEE words, cancellation, excluded NaNs, background correction, metadata,
corrupted-upload cleanup, released owners and cross-device ownership. They are
the only CPU numerical references. Full-sized comparisons use GPU references;
CPU SHA-256 and host comparisons are verification, not reconstruction.

## Timing interpretation

Final measured source: `de942589a5cfad8df9d2132d90d9f4a1f75b0d27`.
Each load number is one fresh-process run; DP numbers are medians of 15 warm
queries. The encoded body is 1,084,227,840 bytes (not a large compression saving).

| Operation | Runtime | Statistic | Time | Device tested | Date tested |
| --- | --- | --- | ---: | --- | --- |
| Verified encoded reopen | CUDA | one run | 0.652 s | RTX PRO 6000 Blackwell | 2026-09-20 |
| Verified encoded reopen | Python MPS | one run | 0.872 s | Apple M5 | 2026-09-20 |
| Resident point DP | CUDA | median | 0.029 ms | RTX PRO 6000 Blackwell | 2026-09-20 |
| Resident point DP | Python MPS | median | 0.241 ms | Apple M5 | 2026-09-20 |

All 268,435,456 decoded float words match logical SHA-256
`459f2b401d862f79856949900e4602a374b937f87a900518802874b538aede69` on both backends.
All four detector images match native Metal exactly. Mean-DP maximum absolute
difference is 0.0234375 (3.50e-7 relative to the peak), within the predeclared
elementwise tolerances. CoM maximum absolute differences are below 2.3e-5 pixels.

[results.json](results.json) retains samples, parity differences and timing
boundaries. Loading is wall time from `io.load` through accelerator completion
in a fresh Python process, with OS file pages uncontrolled and generally warm
after export/transfer. It is **not cold SSD throughput or app first-presentation**.
Point queries are warm resident operations, synchronize both sides of the timer,
and exclude conversion to a host array. Detector timings include the host image.

The point A/B/A uses the unchanged integer-lane decoder as a reference and
requires identical hashes. The detector A/B/A is an earlier full-window helper
versus compensated direct-lane reduction; it is **not a bit-identical speedup
over a previously shipped float CUDA reader**, which did not exist. Native
Metal's compensated result is the frozen scientific oracle. The helper uses a
different accumulation order. Broad CUDA masks can be slower with compensation;
that cost is retained for correct native parity, not hidden behind a faster
uncompensated path.

MPS latency varied considerably during desktop activity. Only backend
interoperability is qualified here, not stable 120 Hz performance. The first
mean-DP and CoM reductions still need performance work; mean DP is cached.
In the final run these took 1.75 s and 3.26 s on MPS, versus 10.5 ms and 17.3 ms
on CUDA. They are not included in the encoded-reopen time above. CUDA broad-mask
reductions took approximately 13.5–14.5 ms, exceeding an 8.33 ms frame budget.
No native UI end-to-end, notarization or release signoff is implied.

Generated .qem/reference arrays were removed locally and on the isolated CUDA
validation host. Original acquisitions and the remote working checkout were
not changed. Only small public-data reports remain.
