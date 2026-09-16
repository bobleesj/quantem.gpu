# Native Metal SSB scan sizes

Native 128x128 and 256x256 reconstruction and full-IFFT phase-variance fitting
are implemented alongside the existing 512x512 engine. No input is padded,
cropped, binned, narrowed, or reduced to fewer BF terms.

This is **backend verification**, not an application release or a real-data
120 FPS claim. The native application's existing 512-only selection gate and
image dimensions still require separate integration and headed testing.

## Numerical checks

`tests/metal/export_ssb_scan_size_reference.py` exports deterministic uint8
counts and independent full-plane CuPy complex-probe results. It also checks
the production CUDA engine's complex object and `reconstruct_with_loss` against
that equation. CUDA revision: `3b61354ef9ad7e3c8d059e43ca3cf08106da3dbe`.

`scripts/check_metal_ssb_scan_sizes.sh <reference-directory>` checks:

- 128, 256, and 512 native scan dimensions, all 95 BF pixels in the control's disk;
- zero aberrations, C10/C12/phi12, and rotated geometry;
- cached, streamed, and hybrid execution, alternating object and loss calls;
- complex-object relative L2 below `1e-4`, maximum wrapped phase error below
  `2e-4` radians, and loss absolute error below `1e-6 + 1e-5 * abs(reference)`;
- exact saved-run round trips, native-size calibration, equal-count uint16
  and uint32 input, higher-order cache/stream parity, and 200-trial small-size fits.

The observed worst errors in the independent rerun were approximately
`6.84e-7` object relative L2, `1.39e-6` radians phase, and `1.20e-7` loss.
The original `scripts/check_metal_ssb.sh` also passes unchanged, including
large count scaling, every higher order, and a 200-trial fit plus refinement.

## Failure investigation retained

1. Clean `e0a0abc` reproduced the old 512 cached/streamed loss failure exactly:
   `0.17328005` versus `0.1730958`, relative error `0.0010632381`, gate `5e-5`.
   The corrected-spectrum anti-Hermitian identity is not valid on the signed
   Nyquist axes. Two small 1-D inverse transforms now restore their complex
   residual before phase accumulation. The same unchanged gate now measures
   approximately `5.17e-7` relative error. No frozen result or tolerance changed.
2. An initial zero-aberration numerical control used a rational detector center
   `(7.4, 7.7)`, making many frequencies exactly perpendicular to BF coordinates.
   Gamma normalization is ill-conditioned there: independent aperture rounding
   produced object relative error `1.21e-4`. Both CUDA reconstruction paths
   agreed with each other (`1.61e-7`) but failed that equation gate. The retained
   cross-backend control uses a nondegenerate subpixel center `(7.41327, 7.73113)`.
   This does not establish cross-backend parity at the degenerate normalization
   singularity. It is not a precision-tolerance relaxation.
3. The first CUDA 256 comparison accidentally called the sparse optimizer loss:
   `0.0573312` versus full-IFFT loss `0.155959`. The corrected test explicitly
   calls `reconstruct_with_loss`; these are different scientific quantities.

## Full-aperture scaling, not real acquisition performance

Apple M5, 24 GB, 2026-09-15. Deterministic native uint8 scans, 192x192 detector
geometry, **8,937 selected and active BF terms**, full cached complex64 Fourier
evidence. Three warmups and 20 repetitions, changing C10. Timing is a completed
resident operation; file loading, app rendering, and optimizer search are not
included. Before and after use the same `performance.swift` harness and counts.

| Scan | Operation | Version | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---:|---:|---:|
| 128x128 | Objective | Initial generic FFT | 86.81 | 86.47 | 88.34 |
| 128x128 | Objective | Fused mixed radix | 36.76 | 36.65 | 39.19 |
| 256x256 | Objective | Initial generic FFT | 403.15 | 403.44 | 406.24 |
| 256x256 | Objective | Fused mixed radix | 156.62 | 157.23 | 161.01 |
| 128x128 | Object redraw | Initial generic FFT | 8.53 | 8.48 | 9.15 |
| 128x128 | Object redraw | Fused mixed radix | 8.46 | 8.48 | 8.55 |
| 256x256 | Object redraw | Initial generic FFT | 22.29 | 22.18 | 23.14 |
| 256x256 | Object redraw | Fused mixed radix | 24.78 | 24.81 | 25.89 |
| 512x512 | Objective | Initial comparison | 257.99 | 256.32 | 283.19 |
| 512x512 | Objective | Later comparison | 295.53 | 292.93 | 319.68 |

The smaller-size objective improved about 2.36x and 2.57x by p50. Object redraw
was not accelerated; the later 256 redraw and unchanged 512 control were slower.
These sequential runs are not sufficient to separate thermal/system variation
from a small redraw regression, so no redraw speedup or non-regression is claimed.
Both 512 rows already include the Nyquist fix; they do not measure its overhead.

| Scan | Fourier cache (GB) | Sampled Metal allocation (GB) | Prepare (s), later run |
|---|---:|---:|---:|
| 128x128 | 0.595 | 0.770 | 0.215 |
| 256x256 | 2.361 | 3.022 | 0.526 |
| 512x512 | 9.408 | 11.976 | 3.176 |

Allocation is sampled device allocation, **not peak process memory**. Preparation
starts from in-memory counts, not HDF5. Full-aperture 200-trial/refinement time
and end-to-end app FPS remain unmeasured. See `baseline.json`, `fused.json`, and
`manifest.json` for complete metrics and provenance.

All recorded artifact sizes and SHA-256 hashes were verified. The publication
uses an anonymous development-host label and redacts a local path in the parity
build log; measured results are unchanged. The original log is retained privately.
The shared research-manifest helper assumes a different authority host and does
not validate this native-development record's host field.

## Reproduce

On CUDA, export into a new directory; the exporter refuses to overwrite one:

```bash
CUDA_VISIBLE_DEVICES=1 python tests/metal/export_ssb_scan_size_reference.py /tmp/ssb-reference
```

Copy the reference directory to the Mac, then:

```bash
bash scripts/check_metal_ssb_scan_sizes.sh /tmp/ssb-reference
bash scripts/check_metal_ssb.sh
swift build -c release --product metal-ssb-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -I "$build_dir/Modules" \
  experiments/20260915-metal-ssb-native-sizes/performance.swift \
  "$build_dir"/MetalSSBKernels.build/*.o -parse-as-library \
  -o /tmp/ssb-native-size-performance
/tmp/ssb-native-size-performance /tmp/ssb-native-size-performance.json
```

The scaling benchmark is a separate synthetic stress check. Do not substitute
it for native-size real-acquisition validation or application UI testing.
