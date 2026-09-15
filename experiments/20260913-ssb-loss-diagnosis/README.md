# Cached versus streamed SSB objective diagnosis

## Cause established on the failing fixture

The cached objective uses a half-plane transform, assumes the corrected
non-DC spectrum is anti-Hermitian, and discards the imaginary part after
transforming the corresponding Hermitian signal. The streamed objective
instead inverse-transforms the full corrected complex plane and evaluates
`atan2(imaginary, real)` without that projection.

Discrete even-sized FFT Nyquist endpoints do not obey the continuous-frequency
sign reversal used by the correction formula. The half-plane projection
therefore drops a nonzero component. This is not ordinary float32 roundoff.

The independent NumPy complex128 reference reproduces the discrepancy on the
existing three-BF, full 512-square integer test fixture:

| Method | Loss at zero DC |
| --- | ---: |
| Full-plane reference | 0.173095850663 |
| Streamed Metal, preceding test | 0.1730958 |
| Half-plane/projected reference | 0.173279975680 |
| Cached Metal, preceding test | 0.17328005 |

Reference relative difference is 0.10637%. Removing both Nyquist lines in a
diagnostic copy makes full and projected reference losses agree to 4.83e-16
relative error, with maximum complex-object difference 1.34e-13. This
intervention identifies the boundary contribution; it is NOT an acceptable
production fix because it discards frequencies. Positive DC can suppress the
relative loss effect: at DC=100,000,000 the same fixture differs by 0.002326%.

The independent calculation is in `reference.py`, with retained output in
`reference-results.json`. It does not call the Metal correction implementation.
No production scientific behavior was changed during diagnosis.

## Speed mechanism and measurement scope

Cached mode retains Fourier evidence and uses the fused half-plane objective.
Streamed mode obtains packed detector columns, converts counts, recomputes
forward FFTs, corrects full complex planes, and runs full inverse FFTs every
objective. The speed comparison therefore includes both transform reuse and
the different objective implementation; it cannot isolate their contributions.

Full-data measurements use the same original-file probe, full 8,937 BF terms,
512×512 scan and 192×192 detector, three C10 values and six timed warm calls.
Initial repetitions are excluded; startup and loading are separate. One GPU
workload runs at a time. The zero-budget arm uses
`SSB_PROBE_CACHE_BUDGET_BYTES=0`; default uses the admitted Fourier cache.

Probe source SHA-256:
`d025599167fa03c51a60a7e29f4934375db97b63dc6958fbced98ba70dd484ac`.
Executable SHA-256:
`d90a55311649693780cb2bdd9e79777a3dc914cbdb5f9a9ff419dbb260d6a56e`.

## Corrective direction, not implemented here

Full-data first matched pair (six warm objective calls):

| Mode | GPU objective mean/p50/p95 | Wall objective mean/p50/p95 | Sampled allocation |
| --- | --- | --- | --- |
| Cached | 374.52 / 333.47 / 576.50 ms | 377.12 / 334.83 / 577.92 ms | 12,573,016,064 B |
| Streamed | 4162.07 / 4171.67 / 4229.43 ms | 4171.81 / 4181.78 / 4239.18 ms | 3,089,956,864 B |

Object redraw GPU p50/p95 was 98.44/117.59 ms cached and
2351.64/5711.51 ms streamed. These are different operations from the objective.
Original-file loading was 2.799 s cached-arm / 4.690 s streamed-arm; cache
preparation was 2.936 s / effectively zero. This is one noisy local matched
pair, not a stable hardware-limit or cold-I/O benchmark. Allocation is not peak.

The bracketing cached repeat measured GPU objective p50/p95 297.32/353.26 ms
and wall p50/p95 298.55/356.95 ms, with unchanged allocation. Thus the measured
wall-median gap is approximately 12.5-14.0x for this session, not a precise
portable speedup claim. Repeat report SHA-256:
`a9e5f020dd1f95425c78e4ee9143fd0f991bfff5e2a6385a6989cb167dba6406`.

Real-data loss disagreement (cached versus streamed): C10=0 has relative
error 4.4344e-5; C10=55 has 3.4637e-5; C10=155.96977 has 5.7935e-5.
The last exceeds the existing 5e-5 tolerance: losses 0.136068195104599 and
0.13606031239032745. No independent full-data CPU oracle was run; the
independent oracle covers the smaller diagnostic fixture above.

Raw reports: `local-evidence://ssb-loss-diagnosis-20260913/cached/report.json`
(SHA-256 `c6679e8be8b4bcbc3f6bc7312558709a03b8453f4eae090e856844d3fb423cbf`)
and `local-evidence://ssb-loss-diagnosis-20260913/streamed/report.json`
(SHA-256 `4bd8a4aeab37370bcf79062a4a63269525397e6acd7afba5550759f3d956184b`).

Preserve caching and the fast interior computation, but account exactly for
the exceptional endpoint contribution before evaluating phase. Validate both
paths against a full-plane reference for zero, positive and negative DC,
Nyquist-rich inputs, and real acquisitions. Do not loosen tolerances or
silently remove frequencies. The independent fixture identifies which
implementation matches the full-plane formula; it is not a universal
scientific signoff for all SSB cases.
