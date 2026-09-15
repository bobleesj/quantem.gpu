# Small pair-loop unrolling

No factor qualified. FC19 factors 2/4/8 unroll the existing pair loop without
new retained sums or resident buffers. All per-pair reductions remain exact.
Default factor remains 1.

| Factor | Residual mean / median (ms) | Combined mean / median (ms) | Combined p95 (ms) |
|---|---:|---:|---:|
| 1 |42.96 /41.77|64.77 /64.06|73.43|
| 2 |46.28 /47.99|67.88 /70.03|76.34|
| 4 |44.08 /45.01|66.52 /66.42|76.45|
| 8 |44.75 /46.27|66.74 /66.76|77.78|

20 measured trials per factor, one warmup each, shuffled round order. Same
seven full uint16 sources and exact signed ADF transition, retained ANS and
index allocations, reusable scratch. Combined paired ratios to factor 1:
1.039 /1.054 /1.000 for factors 2/4/8. No reliable combined gain.

All 588 source transitions passed exact stage parity, including warmups.
All new variants also pass the small signed high-count/malformed-stream fixture.
The final ordinary configuration ran all 20 detector masks twice across seven
sources: 280 complete maps matched 140 previously frozen independent references.
Those checks verify output correctness, not an ordinary UI performance gate.

Memory accounting including scratch is 11,885,234,636 bytes for every factor.
Sampled Metal allocations range from 11,883,921,408 to 11,891,261,440 bytes,
including the transition from before to after diagnostic scratch preparation.
These are snapshots, not process peak memory. Kernel register use/occupancy
were not measured. Unrolling may increase instruction/register pressure, but
the timing result alone cannot establish that as the cause.

Timing is all-seven backend return, not displayed FPS or loading time.
Run the preceding campaign's `run.py --pair-unroll` to reproduce. No push,
merge, installed app or production kernel default changed.
