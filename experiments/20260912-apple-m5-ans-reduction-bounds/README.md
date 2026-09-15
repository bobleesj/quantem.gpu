# Exact reduction and bounded submission

Test two independent factors, retaining seven full uint16 acquisitions:

1. FC11 assigns each scan output to one SIMD lane's 16 private accumulators.
   Dense contributions no longer issue 512 lane-zero threadgroup atomics per
   selected-stream chunk. Sparse contributions retain the existing 8 KiB
   threadgroup atomic array, merged at publication. This may increase register
   use or spill; no such hardware counter is available here. Default is off.
2. Benchmark-only coordinated batches contain 1, 2, 4 or 7 sources, with a wait
   between groups. This tests contention versus parallelism, not a fused kernel.
   The ordinary concurrent-call control remains `batch=false`.

Both preserve every output element. No extra resident or device scratch buffer
is added. The same masks and frozen full-map oracle apply to every arm. First
cycles are warmup. Returned-image copies, scheduling and waits are inside the
all-seven timer; input loading, hashes and UI presentation are outside.

## Results

Eleven arms passed 6,860 independent full-map hashes. FC11 is slower: indexed
ADF center-1 measured 23.920 / 28.048 / 21.701 ms in off/on/off order. The raw
center-1 case rose from 23.129 to 30.432 ms. Do not attribute this to register
spills without counters; that remains an explanation to investigate.

Bounded coordinated batches of 1/2/4/7 sources measured ADF center-1 at
28.624 / 25.200 / 25.528 / 24.556 ms. Ordinary concurrent calls afterward
measured 22.978 ms. Smaller batches did not establish a win. No variant promoted.
Every arm retained 11,877,814,048 bytes; Metal current allocation was
11,883,921,408 bytes. There was no resident growth from these variants.

Post-run system observation: swap used 4,514 MiB and system-wide free memory
35%. A one-second idle vm_stat interval showed 20 swap-ins and zero swap-outs.
These are system-wide observations, not attributable peak/per-run measurements;
they do not establish that the process caused swap or that no paging occurred.

## Memory and theoretical limits

Seven 512x512 UInt32 output images contain 7,340,032 bytes. A logical complete
read/write is 14,680,064 bytes. Dividing that by Apple's advertised M5
[153 GB/s bandwidth](https://www.apple.com/macbook-pro/specs/) gives about
0.096 ms **for output traffic alone under an ideal bandwidth assumption**.
This is not a kernel latency floor: input bytes, lookup tables, dependent ANS
state transitions, reduction instructions, caches and scheduling are omitted.
It does not establish an achievable whole-query rate or prove 120 FPS.

The device exposes only GPUTimestamp, and Instruments/xctrace is unavailable.
Do not describe derived traffic arithmetic as measured occupancy or cache misses.

## Bounded reuse, separate from fresh computation

An additional prior-image buffer and mask would need 7,598,080 bytes across
seven residents (about 7.25 MiB), not another 4D volume. It could accelerate
immediate A/B backtracking through exact buffer reuse. It cannot be reported
as faster new ADF computation and is not included in these kernel timings.
