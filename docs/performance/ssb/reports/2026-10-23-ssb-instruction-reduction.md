# Native 512 SSB instruction-reduction experiments

## Scope and frozen reference

Full 512 × 512 × 192 × 192 acquisition, all 9,275 active bright-field pixels,
native output, fifth-order optics, tilt and depth. Counts, calibration,
preprocessing and precision are unchanged. This measures coherent redraw with
`phase_of_mean`, without fitting or loss evaluation. The source is revision
`e8750f3e3652` plus the previously validated coefficient struct and native-512
launch layout. No production compute source changed during this investigation.
Private evidence: `20261023-ssb-instruction-reduction`.

## Where the instructions go

A source-correlated Nsight Compute capture of a middle correction launch on an
RTX PRO 6000 Blackwell Workstation Edition counted 6,688,889,305 executed
instructions. Deduplicating instruction addresses across source-line mappings
attributes approximately:

| Work | Executed instructions |
| --- | ---: |
| Fifth-order aberration polynomial | 34.43% |
| Depth weighting and sinc evaluation | 23.82% |
| Probe aperture/geometry | 10.82% |
| Compensated accumulation | 3.01% |
| Shared source attribution | 2.53% |
| Other work, including loads, normalization and control | 25.38% |

These are instruction-count fractions, not independently measured wall-time
fractions. The separate hardware-counter study found 91–92% issue utilization
and 31–32% DRAM utilization. Source-line totals must not be summed naively:
some machine instructions are attributed to multiple lines.

## Matched experiments

Each row compares against a contemporaneous reference in the same process.
The input, full BF set, model and output grid stay fixed. Warm medians exclude
initial rounds; host copies and parity comparisons are outside timing.

| Candidate | Reference | Candidate | Decision |
| --- | ---: | ---: | --- |
| Restrict-qualified pointers | 21.91 ms | 21.90 ms | Exact, no meaningful gain |
| Explicit polynomial loop unrolling | 21.91 ms | 21.90 ms | Exact, no meaningful gain |
| Four-way BF-loop unrolling | 21.91 ms | 22.44 ms | Exact, slower |
| Skip phase calculation behind individually closed apertures | 21.91 ms | 21.31 ms | Not bit-identical; rejected |
| Pair positive/negative frequencies with separate corrections | 21.69 ms | 20.84 ms | Not bit-identical; experimental only |
| Pair frequencies and reuse polynomial/depth intermediates | 21.50 ms | 15.61 ms | 27% faster, fails numerical acceptance |
| Certified quotient fast path with exact-division fallback | 21.93 ms | 24.63 ms | Exact on tested sweeps, slower |
| Refined certified quotient | 21.93 ms | 24.94 ms | Exact on tested sweeps, slower |
| Prepare per-BF probe geometry/phase once per redraw | 21.78 ms | 21.99 ms | Exact, slower |
| Prepared probe with direct reads and no shared-memory staging | 21.76 ms | 21.98 ms | Exact, slower |

## Why pairing is promising but not accepted

For a frequency pair, the two aberration evaluations at `k - q` and `k + q`
exchange roles. The depth weights similarly exchange under the symmetric
model. A prototype reuses these intermediates, calculates both output values,
retains the BF accumulation order, and treats Nyquist rows separately. It
needs no acquisition-sized cache and no reduction of data or precision.

The naive reuse changes floating-point results. Explicit rounding in the
normalization helped the baseline C10 sweep: its eight checks passed the
existing `atol=1e-6, rtol=1e-5` comparison. It did not generalize. A further
21 cases varied C10, depth, tilt and the higher-order coefficients; only six
passed, with a maximum observed phase difference of 3.46e-6 radians. The
reference was not replaced and tolerances were not loosened. Compiler
contraction/order and reuse of intermediates are suspected contributors, but
this experiment does not isolate every discrepancy to a specific instruction.

## Counter confirmation of the faster structure

The explicit-normalization paired prototype reduced executed instructions in
the profiled middle launch from 6.689 billion to 4.402 billion (34.2%). Its
launch duration was 3.12 ms versus the reference's 4.27 ms under profiling.
Issue utilization fell to 73.25%, achieved occupancy to 46.76%, and DRAM
throughput rose to 43.71%; spilling remained zero. This supports the proposed
work-reduction mechanism. It does not override the failed numerical checks or
turn profiler timings into notebook throughput.

## Conclusion

The instruction-heavy work is now identified and a substantially faster
execution structure has been demonstrated. It is not ready to enable. The
next blocker is preserving the numerical contract across joint-parameter
settings, rather than occupancy or large caches. All candidate overrides were
process-local and restored; no production kernel or public API was changed.

No new MPS, notebook image-update rate or 60 fps claim is made. Profiler replay
and unlocked-clock observations are separate from paired warm timings. The
exact variants that were neutral or slower are retained as rejected evidence.
