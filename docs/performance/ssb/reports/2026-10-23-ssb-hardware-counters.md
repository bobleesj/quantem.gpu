# Native 512 SSB hardware-counter diagnosis

## Question and workload

Is full-data coherent SSB redraw limited by hardware throughput, rather than
transfer, launch overhead or an inverse FFT? This profiles native 512 × 512
output from a 512 × 512 × 192 × 192 acquisition with all 9,275 active
bright-field pixels. Fifth-order optics, sample tilt, depth, calibration,
precision and preprocessing remain unchanged. The scientific quantity is
`phase_of_mean` without loss evaluation; it is not an optimization/refit.

The inspected revision is `e8750f3e3652` with the previously validated coefficient
struct and native-512 launch layout. The imported source was fingerprinted.
Private evidence is experiment `20261023-ssb-hardware-limit`.

## Direct measurement

Nsight Compute 2025.4.1 profiled the RTX PRO 6000 Blackwell Workstation Edition.
One middle correction launch received the full section set; a separate capture
measured all five correction launches. Each capture followed three warm redraws.
The resulting phase remained bit-identical to its uninstrumented reference.

| Counter | Across the five launches |
| --- | ---: |
| Instruction issue slots busy | 91.19–91.90% |
| DRAM throughput relative to sustained peak | 31.22–31.83% |
| Achieved occupancy | 64.98–65.68% |
| Registers per thread | 53 |
| Local-memory spilling requests | 0 |

Theoretical occupancy is 66.67%, limited by registers. The middle launch's
highest-utilized execution pipeline was FMA at about 59%; FP32 arithmetic
reached about 40% of peak. The 92% figure is instruction issue utilization,
not 92% of peak floating-point arithmetic or of all GPU resources.

## Interpretation and limits

This implementation is primarily instruction-throughput limited, with healthy
scheduling and no register spilling. DRAM bandwidth is not the measured limiting
resource. Increasing occupancy alone has little unused issue capacity to fill.
Substantial improvement requires reducing executed work or changing the
instruction mix while preserving the scientific contract. This does not prove
an algorithmic lower bound or establish that 60 or 120 fps is impossible.

GPU clocks and caches were not forced; counter values come from multipass
replay on a GPU that also serves the desktop. All five launches agree, but this
is not a controlled-clock theoretical roofline bound. Profiler replay wall time
is excluded from ordinary redraw timing. Neither browser FPS nor MPS behavior
was measured in this experiment. Generic profiler suggestions to fuse
operations are hypotheses: they may change floating-point rounding and require
independent parity validation.

## Register-pressure control

A separate, unprofiled alternating test changed only the compiler register cap.
Sixteen rounds per configuration varied C10 over eight values; medians below
exclude the first two rounds. All data and scientific parameters were retained.

| Register allocation | Warm redraw median | Maximum phase difference |
| --- | ---: | ---: |
| Default, 53 registers | 22.03 ms | 0 |
| Cap 40 | 23.19 ms | 1.03e-6 rad |
| Cap 48 | 22.48 ms | 1.03e-6 rad |
| Cap 64 | 22.09 ms | 0 |

The 40- and 48-register variants compiled with 32 and 8 bytes of local memory
per thread, respectively; the default had none. They were slower and not
bit-identical, so neither is accepted. The 64-register variant was bit-identical
but showed no gain. No production kernel was changed in this investigation.

Metric interpretation follows the [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
