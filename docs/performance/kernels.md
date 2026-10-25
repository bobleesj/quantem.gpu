# GPU kernel registry

One row per kernel version. A result records the kernel that produced it
(`MetalKernelIdentity`: name, version, source digest, pipeline variant, and a
fingerprint over both), so any saved phase, export, or benchmark line can be
traced to a row here.

## Rules

- **One production kernel per job.** When a faster version is validated, it
  replaces the older one; the older one is retired, not kept as a silent
  fallback. Any case the new kernel cannot serve is named below and in code.
- **Version on change.** Editing a kernel's `.metal` source or its numerical
  contract requires a new version here and in `SSBKernelRegistry`.
  `SSBKernelRegistryTests` fails when a shipped source digest does not match
  its registered version, or when this page lacks the version or digest.
- **Active means measured.** A version becomes active only with parity tests
  passing and both benchmark and in-app headed timings recorded with date,
  device, and dataset. Retired versions stay listed with their last numbers.

Timings are median / p90 ms per reconstruction. Device for all rows: MacBook
Pro, Apple M5 Max, 40-core GPU, 128 GB, macOS 26.4. Data: the first acquisition,
512 × 512 × 192 × 192, 8,806 bright-field pixels, 300 keV, 30 mrad; 256²
rows use the same acquisition's full detector at a 256² scan
(`docs/performance/ssb/reports/2026-10-24-native-metal-ssb-redraw.md`).

## Native SSB reconstruction (`MetalSSBKernels`)

| Kernel | Status | Source digest | Serves | Validated | Fifth order, 512², 3× | Low order, 512², 3× | Fifth order, 256², 3× | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-redraw` v0.2 | **active** | `b20d7914811d` | `phase_of_mean`, `complex_wave`, 1–4× | 2026-10-24 | benchmark 33.7 / 33.9; in app 30.3–33.0 (median ≈ 31.3, ≈ 32 FPS) | benchmark 31.2 / 31.5; in app 32.5–34.9 | benchmark 9.9 / 10.0 | `SSBRedrawReconstructionTests` (per-term and double-precision fifth-order references), `SSBComplexWaveTests` | — | — |
| `ssb-per-term` v0.1 | **retired** (test reference; explicit 8× only) | `1c390cc7b833` | 1–4× before 2026-10-24; 8× now | before 2026-10-24 | benchmark 214.9 / 223.8; in app ≈ 320 | benchmark 60.4 / 64.3 | benchmark 67.9 / 69.3 | reference for v0.2 | `ssb-redraw` v0.2 | 2026-10-24 |
| `ssb-mean-phase` v0.1 | **active** (`mean_phase`, C10/C12 only) | `4aa7c1da6391` | `mean_phase` | before 2026-10-24 | n/a (no higher orders) | not remeasured | not remeasured | existing joint tests | — | — |

The first render after a fit costs about 100–130 ms in the app (pipeline
variant compilation and warm-up); the steady drag rate above excludes it.
In-app timings are `MetalSSBJointPhase.wallSeconds` on an otherwise idle GPU
with one app instance on the built-in display.

## Fit objectives (`MetalSSBJointModel` agreement)

GPU ms per candidate objective (median of three batches; command-buffer GPU time, which the wall time of the
call exceeds by at most 0.2 ms) for the fifth-order objective at a fit's kept rotation. Batch: one candidate,
four seeded random candidates, the 57-point Newton diagonal stencil, 64 random candidates. Data: film
`scan_06` (8,805 BF pixels, 44,247 band frequencies), a second 512² acquisition (9,296 BF, 44,255) and a
third (8,928 BF, 77,842), all 300 keV, 30 mrad. Benchmark: `metal-ssb-fit-benchmark --objective-bench`.

| Kernel | Status | Source digest | Serves | Validated | film 1 / 4 / 57 / 64 | Second | Third | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-fifth-agreement` v0.2 | **active** | `f2f700147e47` | `fifthOrderAgreements` (fifth-order fit) | 2026-10-25 | 11.2 / 6.5 / 6.4 / 7.5 | 11.9 / 7.2 / 6.3 / 7.4 | 18.6 / 11.5 / 10.3 / 12.9 | `SSBFifthOrderNumericsTests` (per-term kernel and float64 objective); measured: within 3.1e-7 of v0.1 on all three acquisitions | — | — |
| `ssb-fifth-agreement` v0.1 (`fifth_agreement` in `ssb-per-term`) | **retired** (test reference) | `1c390cc7b833` | before 2026-10-25 | before 2026-10-25 | 24.4 / 19.6 / 20.8 / 21.2 | 25.4 / 21.4 / 23.3 / 22.9 | 41.3 / 36.6 / 37.9 / 37.5 | reference for v0.2 | `ssb-fifth-agreement` v0.2 | 2026-10-25 |

v0.2 evaluates the same per-pair float32 terms. Only chi(k -/+ q) is reassociated: a threadgroup owns band
frequencies of at most two band rows and collapses the bivariate chi polynomial once per BF pixel and row, so
each sideband is one degree-6 Horner evaluation. The aperture edge, alpha^2 and depth-sinc arguments are
bitwise those of v0.1 on every (k, q) pair of film (3.9e8 pairs; edge expression only in the one-pixel ring,
divisions through the correctly rounded reciprocal with a Markstein correction); `fast::sin` / `fast::sincos`
equal `precise::` with fast math off except for 1-ulp differences in about one argument in 1e7. Candidates
with equal depth and tilt share their depth-sinc weights. On the synthetic test scene both kernels are about
1e-6 from the float64 objective and v0.2 is no less accurate.

