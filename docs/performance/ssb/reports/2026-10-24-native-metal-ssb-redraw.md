# Native Swift/Metal SSB redraw while dragging aberration controls, 2026-10-24

## Question

Can the native `MetalSSBJointModel.reconstruct` re-render a fitted SSB phase at
30 frames per second (≤ 33 ms per frame, median and p90) while a user drags an
aberration control, after a fifth-order fit, at 3× output sampling, with every
BF pixel, all 14 aberration terms, sample tilt and depth spread, float32 and
complex64, and the default `phase_of_mean` estimator?

## Workload and boundary

- Apple M5 Max, 40-core GPU, 128 GB unified memory; the benchmark checks the
  Metal device name before timing. No thermal or performance warning was
  recorded. No other GPU job ran during the timings.
- Geometry of a 512 × 512 × 192 × 192 acquisition at 300 kV, 30 mrad and a
  0.373 Å scan step: 8,804 BF pixels (radius 52.95 px). The 256 × 256 rows use
  the same detector and step. Counts are deterministic synthetic values; the
  work depends on the geometry, grid and fitted terms, not on count values.
- Low order: C10, C12, φ12 with tilt (1.2, −0.8) mrad and 15 nm depth spread.
  Fifth order: the same plus all 12 higher-order terms nonzero (C21 … C56).
- One observation = one `reconstruct` call with a changed C10, synchronized:
  correction, mirror, origin, inverse FFT and phase into a host-visible buffer.
  Engine preparation and the first 3 calls are excluded; 20 warm observations.
  App display, colormap and FFT panel are excluded.
- Runner: `swift run -c release metal-ssb-redraw-benchmark --scans 256,512`
  (`native/swift/Benchmarks/MetalSSBRedrawBenchmark`). Before = the same runner
  built against `f5599f73`; after = `518c6761`. Both ran in one quiet session.

## Results

Per-frame time, median / p90, ms.

| Scan | Order | Output | Before | After | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| 256² | low | 1× | 13.5 / 14.0 | 5.8 / 5.9 | 2.3× |
| 256² | low | 2× | 25.3 / 25.7 | 8.7 / 8.8 | 2.9× |
| 256² | low | 3× | 24.3 / 24.8 | 8.9 / 9.1 | 2.7× |
| 256² | low | 4× | 24.3 / 24.9 | 9.4 / 9.5 | 2.6× |
| 256² | fifth | 1× | 29.1 / 29.3 | 6.4 / 7.1 | 4.6× |
| 256² | fifth | 2× | 62.8 / 63.7 | 9.9 / 10.1 | 6.4× |
| 256² | fifth | 3× | 67.9 / 69.3 | 9.9 / 10.0 | 6.9× |
| 256² | fifth | 4× | 68.1 / 68.9 | 10.3 / 10.4 | 6.6× |
| 512² | low | 1× | 29.9 / 30.6 | 20.5 / 21.2 | 1.5× |
| 512² | low | 2× | 65.3 / 67.0 | 30.5 / 31.6 | 2.1× |
| 512² | low | 3× | 60.4 / 64.3 | 31.2 / 31.5 | 1.9× |
| 512² | low | 4× | 60.6 / 61.6 | 32.2 / 33.0 | 1.9× |
| 512² | fifth | 1× | 101.5 / 102.9 | 21.8 / 22.7 | 4.7× |
| 512² | fifth | 2× | 203.0 / 212.3 | 34.6 / 35.1 | 5.9× |
| 512² | fifth | 3× | 214.9 / 223.8 | 33.7 / 33.9 | 6.4× |
| 512² | fifth | 4× | 224.6 / 227.2 | 35.7 / 36.3 | 6.3× |

The table's "after" column ran directly after the ~10-minute "before" sweep.
The target case run alone, twice: 28.8 ms median / 29.8–30.0 ms p90 (34.7 FPS).
300 consecutive frames: 29.2 ms median / 32.2 ms p90. The 30 FPS target at
512² fifth order 3× is therefore met for a drag, with about 1 ms p90 margin;
after many minutes of continuous load it sits at the 33 ms line. 2× and 4×
at 512² fifth order are 1–3 ms above it.

60 FPS at 512² 1× is not reachable on this machine with every BF pixel in
complex64: the 9.3 GB Fourier cache G(q, k) must be read each frame, and
measured read bandwidth is 450–510 GB/s (≥ 19 ms). Fifth-order 1× reaches
45.9 FPS. At 256², every case exceeds 97 FPS.

## What changed (`ssb_redraw.metal`)

1. Anti-Hermitian half: Γ(−q, k) = −conj Γ(q, k) also for odd orders, because
   the minus sideband uses χ(k − q); only source rows 0 … N/2 are evaluated and
   mirrored, with the output Nyquist column evaluated directly.
2. One thread per source frequency visits its output aliases inside each
   32-pixel batch, so aliases reread that batch of G from cache. Per-alias
   threads streamed G from DRAM once per alias (~17 GB per frame at 3×).
3. χ as a row-collapsed polynomial: per BF pixel and output row, χ is a degree
   ≤ 6 polynomial in the output column; the threadgroup computes the
   coefficients once, and each pair costs one Horner evaluation. Bivariate
   coefficients are built on the host in double precision from the packed
   float32 terms.
4. Conservative per-SIMD-group compaction: each group keeps only the BF pixels
   whose minus or plus sideband can open in its column interval (94% lane
   use), split into minus-only, plus-only and both. With one open sideband,
   conj Γ / |Γ| = ±exp(iΔχ): aperture and depth magnitudes cancel. Only the
   sinc sign is used, with a guarded fallback to the general formula near the
   1e-8 normalization floor.
5. `fast::sincos`, which is bit-identical to `precise::sincos` for |x| ≤ 200
   rad on this GPU, branch-free aperture weights, and padded float4 rows.

8× keeps the earlier per-term kernel.

## Parity

- Against the per-term kernel (`SSBRedrawReconstructionTests`): max wrapped
  phase error 5.2e-8 rad, RMS 1.5e-8 at 128²; 1.4e-7 / 5.6e-8 at 256². This
  covers 1–4×, cached and streamed spectra, low and fifth order, with and
  without depth. The stated bound is 1e-5 / 1e-6.
- Against an independent double-precision fifth-order definition: ≤ 1.1e-7
  rad, bound 2e-4.
- Full-size benchmark phases, after vs before: ≤ 3.9e-6 rad at 512², ≤ 8.5e-7
  at 256², correlation ≥ 0.99999972. These synthetic phases have small
  contrast (std 1e-3 rad).
- The new `complexWave` estimator agrees with Python MPS
  `phase_estimator="complex_wave"`: 2.5e-8 rad in phase, 3.1e-7 of the mean
  amplitude, on a committed synthetic fixture.

## Rejected candidates

These variants were measured and not kept:

- Accumulators for all F² aliases in registers: correct but slower (53–57 ms).
- G staged in threadgroup memory: 52–55 ms. Threadgroup memory shares the
  on-chip pool with registers and cache on this GPU.
- Several columns per thread, two pixels per iteration, software prefetch,
  rint-based sign: neutral or slower.

A bug found during development: a barrier inside non-uniform control flow
produced a nondeterministic race; the shipped kernel's barriers are uniform.

## Limits

- The app's on-screen render time and displayed FPS were not measured: control
  of the app was declined for the headed check, so the headed check remains
  open. Kernel time is not displayed frame rate.
- Synthetic counts at the real geometry; the real acquisition's BF count can
  differ slightly (8,807 vs 8,804).
- One Apple GPU; other Apple GPUs are not qualified.

Private evidence: experiment `20261024-metal-ssb-interactive` (raw JSON, logs,
phase dumps). Kernel SHA-256 (`ssb_redraw.metal`):
`b20d7914811d0fb2941d59907ba6e15607829f0988b08498f262d9942a3a5e34`.
