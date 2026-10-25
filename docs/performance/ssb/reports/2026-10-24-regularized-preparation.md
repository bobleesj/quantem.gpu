# Where the 9 s of regularized preparation goes (2026-10-24)

## Question

`DirectPtycho.run(fit, method="phase" | "complex", upsampling_factor="auto")` takes about 9 s the first time and about 1 ms when only a strength changes. Switching phase -> complex -> phase pays the 9 s each time. Which stage costs the time, which parts depend on the optics, and what would it take to keep both methods prepared?

## Setup

- Frame 151 of a MoS2 bilayer time series, full 512 x 512 scan, 192 x 192 detector, 300 kV, 30 mrad, 0.373 A scan step, rotation 176.3 deg, the saved broad fit (14 aberration terms, tilt (5.92, -0.07) mrad, depth 12.4 nm).
- `upsampling_factor="auto"` resolves to 3. Bright-field pixels: 9298. `G_qk` is (9298, 512, 257) complex64, 9.8 GB.
- GPU 0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition (188 SMs), live-env, CuPy, CUDA device confirmed by name. GPU 0 was idle (0 % utilization) during the runs.
- Scripts (scratchpad `prep-profile/`): `api_table.py` (public API timings), `stages.py` (a copy of `PreparedDirect.prepare` with a device synchronize around each stage), `stages_variants.py` (same copy with the chunk size and the Gram implementation as parameters), `gram_bench.py` (Gram alternatives on the real transfer kernel, plus `torch.profiler` kernel names). No source file was edited.
- `ncu` and `nsys` are not installed on the test machine; kernel identity comes from `torch.profiler` (CUPTI), which records CuPy kernels too.

## Code path

`DirectPtycho.run` -> `direct_workflow.run` -> `direct_workflow.solve` -> `PreparedDirect.prepare` (`ssb/cuda/direct.py`) on a cache miss, then `PreparedDirect.solve(strengths)`.

`prepare` loops in Python over alias counts (3, 4, 5, 6 here) and, inside each, over chunks of 128 scan frequencies: 130560 frequencies -> 1022 chunks. Per chunk it gathers `G_qk` at the frequencies, evaluates the transfer `_transfer` (an ElementwiseKernel over (frequency, alias, BF pixel)), casts it to complex128, forms the Gram `T^H W T` and right side `T^H W g` with batched `@`, and for `phase` runs a batched `eigh` so that any strength is a diagonal evaluation later. `complex` keeps the Gram and solves `(G + diag(penalty)) x = rhs` at each strength.

The cache is a single slot, `session._prepared_direct = (key, prepared)`. The key holds `id(engine)`, `id(engine.G_qk)`, rotation, every aberration coefficient, tilt, depth, upsampling factor, method, `smoothness_inv_A` and `detector_weighting`; strengths are not in it. A method switch therefore changes the key and evicts the other model. `set_rotation` and `close` clear the slot.

## Raw numbers

Public API, frame 151 after its broad fit (`compute_loss=False`). The first column was measured earlier on 2026-10-24; the next two are this run's repeats.

| Call | earlier | repeat 0 | repeat 1 |
|---|---|---|---|
| ssb | 0.06 s (0.19 first) | 0.206 s | 0.056 s |
| phase, first (prepare) | 9.1 s | 9.138 s | 0.001 s (slot already held phase at these optics) |
| phase, strength only | 0.001 s | 0.001 s | 0.001 s |
| complex, first (prepare) | 9.4 s | 9.379 s | 9.371 s |
| complex, amplitude strength only | 0.004 s | 0.004 s | 0.003 s |
| phase again after complex | 9.0 s | 9.017 s | 8.863 s |

Stage split of one warm preparation (synchronized per stage, so the sum is slightly above the unsynchronized 9.1 s):

| Stage | phase (s) | complex (s) | Depends on optics |
|---|---|---|---|
| Gram `T^H W T`, batched complex128 `@` | **7.998** | **8.111** | yes |
| transfer to complex128, weighting | 0.168 | 0.661 | yes |
| transfer kernel `_transfer` | 0.386 | 0.221 | yes |
| batched `eigh` (phase only) | 0.201 | - | yes |
| right side `T^H W g` | 0.159 | 0.184 | yes |
| alias indexing (`cp.nonzero`, host sync per chunk) | 0.133 | 0.128 | no |
| prior `(1 + q^2/s^2)^2` | 0.121 | 0.118 | no |
| gather measured `G_qk` | 0.102 | 0.097 | no |
| eigenbasis projection | 0.069 | - | yes |
| signal power | 0.059 | 0.059 | no |
| setup and concatenation | 0.004 | 0.003 | no |
| **total** | **9.415** | **9.616** | |

Alias-count histogram: 3: 3770, 4: 106424, 5: 19304, 6: 1062 frequencies. The cold first call in a fresh process measured 16.4 s with the same split, the extra time being CuPy kernel compilation and cuBLAS/cuSOLVER setup.

Kept device memory per prepared model: phase 91.7 MB, complex 170.4 MB (beside the 9.8 GB `G_qk`).

Gram kernel (torch.profiler, phase, chunk 128): 99.5 % of the time is one kernel, `cutlass_80_tensorop_z884gemm`, 10.25 ms per chunk; the remaining 0.5 % is the complex128 copy of the strided operand.

Gram alternatives on the real transfer (alias count 4, scaled to all 130560 frequencies; relative error is the largest entry difference over the largest entry of the shipped Gram):

| Gram implementation | phase, dim 4 | complex, dim 8 | rel. error |
|---|---|---|---|
| shipped: complex128 batched `@`, 128 per batch | 11.8 s | 10.8 s | reference |
| shipped `@`, 1024 per batch | 4.5 s | 4.8 s | same arithmetic |
| shipped `@`, 4096 per batch | 4.1 s | 4.4 s | same arithmetic |
| complex64 batched `@` | 0.25-0.31 s | 0.41-0.54 s | 4e-6 to 1e-5 |
| one block per (frequency, a, b), complex64 input, float64 accumulation | 0.23-0.30 s | 0.87-1.19 s | 2e-14 to 3e-14 |

End to end with the copied preparation (median of 3, synchronized stages):

| Variant | phase | complex |
|---|---|---|
| shipped (chunk 128, zgemm) | 9.30 s | 9.58 s |
| chunk 1024, zgemm | 3.79 s | 4.30 s |
| chunk 128, float64-accumulating Gram kernel | 1.48 s | 2.56 s |
| chunk 1024, float64-accumulating Gram kernel | 0.91 s | 2.11 s |
| chunk 4096, float64-accumulating Gram kernel | 0.86 s | 2.09 s |

## Conclusion

1. **The Gram is 8.0 of the 9.1-9.4 s, and it is neither compute nor bandwidth bound: it is a badly shaped GEMM.** Each frequency's Gram is a 4 x 4 (phase) or 8 x 8 (complex) product with a reduction over 9298 BF pixels. cuBLAS dispatches the FP64 tensor-core CUTLASS kernel `z884gemm`, whose tiles are far larger than 4 x 4, so almost all of each tile is padding, and the batch of 128 gives only 128 thread blocks on 188 SMs, each walking K = 9298 serially. The useful work is about 1.6e11 FP64 flops for phase (6e11 for complex) and about 39 GB of complex64 transfer to read, roughly 0.1-0.3 s at this card's FP64 rate and memory bandwidth. A plain reduction kernel reaches 0.23 s on the same inputs, a 35x speedup, at 2e-14 relative difference. The per-chunk Python loop and host syncs cost about 0.6 s in total (1022 chunks), visible once the Gram is fixed; the dense `eigh` is 0.2 s; there is no host to device copy of data.
2. **Everything that dominates depends on the optics.** The transfer, and with it the Gram, the right side, the eigendecomposition and the basis, depend on every aberration coefficient, the tilt and the depth; the cache key includes all of them. Only the dose, weights, frequency and alias indexing, active mask, prior, `G_qk` gather and signal power (about 0.4 s of the 9.4 s) depend on the data alone. A C10 or C12/phi12 change (the streaming case) must redo 95 % of the preparation. For a pure quadratic aberration the k-dependent phases cancel only in the diagonal phase terms; off-diagonal alias pairs and the 14-term fits keep a k-dependent phase, so there is no closed-form update of the Gram in C10 or C12.
3. **Keeping both methods prepared is cheap in memory: two independent objects, 91.7 + 170.4 = 262 MB**, under 3 % of `G_qk`. The cache today is one slot whose key includes the method, which is why phase -> complex -> phase prepares three times. Two slots (one per method, same optics key) fix it. The complex Gram contains the phase Gram as its lower-right block and the complex right side contains the phase right side, so phase could also be derived from a prepared complex model with only the `eigh` (about 0.2 s); that shares parts but is not bit-identical to an independent phase preparation, because cuBLAS sums an 8 x 8 tile in a different order than a 4 x 4 tile.

## Optimization candidates (not implemented)

- **A. Replace the batched complex128 `@` with a reduction kernel** that reads the complex64 transfer once and accumulates in float64 (one block per frequency, each thread holding the dim x dim upper triangle for its BF pixels, then a block reduction). Measured with a naive one-block-per-entry version: phase 9.3 -> 1.5 s at chunk 128, 0.86 s at chunk 4096; complex 9.6 -> 2.1 s. A per-frequency kernel that loads each transfer row once, and fusing the transfer evaluation into it (skipping the 39-77 GB complex64 transfer and its complex128 copy, 0.6 s for complex), should bring both near 0.5 s. The result differs from the shipped Gram at 2e-14 to 3e-14 relative (summation order), so the parity test needs that written cause; the shipped arithmetic itself is not exact to that level either, being a cuBLAS tensor-core reduction.
- **B. Larger chunks with the shipped arithmetic.** Raising the 128-frequency chunk to 1024 keeps the same zgemm and cuts phase 9.3 -> 3.8 s and complex 9.6 -> 4.3 s with no new kernel; at 1024 the complex chunk's complex128 transfer is about 1.2 GB. Combined with A it removes most of the 0.6 s of Python loop and host-sync overhead.

With A and B, a per-frame rebuild after a C10/C12 change would cost about 0.9 s (phase) and 2.1 s (complex) as measured, versus 0.06 s for SSB, so regularized methods on every live frame remain about 15x to 35x the SSB cost even after the fix.

## Rejected ideas

- **complex64 `@` for the Gram.** It is the fastest variant (0.25 s phase) but the relative error is 4e-6 to 1e-5; the Gram feeds `eigh` and `solve` at strengths down to 0.01 of the scale, so a float32 reduction over 9298 terms is the wrong place to lose precision. The float64-accumulating kernel is nearly as fast at 2e-14.
- **Reusing the preparation across C10/C12.** Not possible without recomputing the transfer and its Gram (point 2); only about 0.4 s is data-only.
- **Blaming the eigendecomposition or the per-frequency loop.** `eigh` is 0.2 s and the loop overhead about 0.6 s; neither explains the 9 s.

# Preparation in one kernel per alias count: 9.1 s to 35 ms (2026-10-24, later)

## Question

Can the regularized preparation run at an interactive rate, so that a defocus or astigmatism slider in ShowPtycho shows phase-only or amplitude + phase at full 3x Auto sampling within about 33 ms (30 fps), at float32-level precision, and can phase and complex both stay prepared at one optics?

## Answer

Yes for phase (35 ms through `run`, 34-35 ms preparation alone), close for complex (51 ms through `run`, 46-47 ms preparation). Both 180x to 270x faster than before. Phase -> complex -> phase at one optics now prepares each once; the third call takes 1 ms and returns the first call's wave bit for bit.

## Setup

As above: frame 151, 512 x 512 scan, 9298 BF pixels, factor 3, the saved broad fit; GPU 0 (RTX PRO 6000 Workstation, 188 SMs), live-env. Other agents shared GPU 0 during the day, so every timing run first waited for 5 s below 10 % utilization (`prep-fast/wait_idle.sh`). Scripts in the scratchpad `prep-fast/`: `reg_time2.py` (the public-API table, `reg_time.py` plus two optics-change rows), `time_prep.py` (`PreparedDirect.prepare` alone, and with solve and phase image), `compare.py` (old against new on every prepared array), `precision.py`, `ablate*.py`, `peak.py`, `peak2.py`, `images.py`. "Before" ran the committed `direct.py` and `direct_workflow.py` on an otherwise identical source tree.

## Raw numbers

Public API (`ptycho.run(fit, upsampling_factor="auto", compute_loss=False, ...)`), median of 3 processes:

| Call | before | after |
|---|---|---|
| ssb (first in process) | 0.203 s | 0.209 s |
| ssb again | 0.056 s | 0.063 s |
| phase, first (prepare) | 9.122 s | 0.120 s |
| phase, strength only | 0.001 s | 0.001 s |
| complex, first (prepare) | 9.356 s | 0.077 s |
| complex, amplitude strength only | 0.003 s | 0.004 s |
| phase again after complex | 9.032 s | 0.001 s |
| phase, C10 + 0.5 nm (new optics) | 9.134 s | 0.035 s |
| complex, C10 + 0.5 nm (new optics) | 9.385 s | 0.051 s |

The first phase call includes loading the kernels for each alias count (CuPy disk cache) and the data-only layout (about 85 ms, once per process and data). The optics-change rows are the slider case.

Preparation alone (`time_prep.py`, 7 calls per process, 3 processes): phase 34.4-35.2 ms minimum, 35.2 ms median; complex 45.8-48.1 ms minimum, 47.4 ms median. With solve and phase image: phase 35.4-35.6 ms, complex 48.9-49.9 ms.

Measured throughput of this card (`peak.py`, `peak2.py`, independent FMA chains): float64 0.886 T FMA/s, float32 59 T FMA/s (1/67), float -> double conversion as costly as a float64 add, copy 1.47 TB/s.

Gram accumulation choices inside the same kernel design (an earlier shared-memory version of the kernel; `precision.py`, minimum of 7, errors against plain float64 sums of exact float32 products):

| Accumulation | phase prep | complex prep | Gram | right side | phase image |
|---|---|---|---|---|---|
| float64 sums, exact products | 122 ms | 327 ms | reference | reference | reference |
| float32-pair (two-sum) sums | 56 ms | 102 ms | 3.5e-14 | 4.0e-14 | 5e-8 |
| float32 products, float64 sums | 61 ms | 141 ms | 1.4e-9 | 2.5e-8 | 2.7e-7 |
| float32 products and sums | 53 ms | 86 ms | 1.9e-7 | 2.6e-7 | 2.7e-7 |

Float64 alone needs about 58 ms (phase) and 138 ms (complex) of float64 operations for the products that remain after skipping zero transfer, before any transfer is evaluated, so the 100 ms floor (and 33 ms) is impossible for complex with float64 arithmetic on this card. With float32-level precision accepted for this CUDA path, the shipped kernel sums in float32 per lane (about 300 terms) and merges the 32 lanes in float64.

Ablations of the shipped kernel (`ablate10.py`, minimum of 5, preparation):

| Variant | phase | complex |
|---|---|---|
| shipped | 35.1 ms | 45.6 ms |
| sums of diagonal entries only (transfer kept) | 31.0 ms | 30.8 ms |
| transfer replaced by a cheap function of k and q (all sums kept) | 14.3 ms | 21.5 ms |
| G_qk gather replaced by a constant | 29.0 ms | 41.5 ms |

Register budget (`rr.py`, `lb.py`): complex at the compiler's choice (255 registers, 8 warps per SM) 61 ms; capped at 168 registers (12 warps) 45 ms; phase capped at 96 registers 34 ms instead of 35-36 ms.

Agreement with the previous implementation, test field: Gram 1.9e-7 to 3.1e-7 and right side 6e-8 to 3.7e-7 of the largest entry, eigenvalues 1.7e-7 to 3.0e-7, signal power 7e-9, phase images 2.5e-7 to 3.0e-7 of the largest phase (8.9e-8 rad), log-amplitude 3.5e-7 absolute. `before_after_phase.png` and `before_after_complex.png` (scratchpad) show the full field and a 256 x 256 crop before, after and the difference: identical by eye; the phase difference is noise at 1e-7 rad, the log-amplitude difference is isolated float32 ulps at |w| = 1.

Kept device memory: phase 91.7 MB, complex 170.4 MB per prepared method, plus 15.7 MB of data-only layout on the engine.

## What changed

1. **One kernel per alias count, one warp per frequency** (`direct_normal_equations<count, complex, parts>`). Lanes walk the BF pixels; each lane evaluates the transfer of its pixel for every alias (the expressions of `weak_object._transfer`) and adds every entry a <= b of the augmented Gram [T G]^H W [T G] to sums in registers. The Gram, the right side T^H W G and the signal power come out of one reduction; there is no chunk loop, no complex128 copy of the transfer and no host synchronization per chunk.
2. **Exact zeros skipped.** A transfer term is zero outside the lens where the two apertures overlap; its chi, sinc and phase factor are not evaluated there.
3. **k-only terms once.** chi(k), the aperture at k and the detector row are computed once per preparation (`direct_pixel_terms`), and the BF pixels run in 4 x 8 detector tiles so the lanes of a warp branch alike (about 10 %).
4. **Data-only layout kept on the engine** (`_layout`): dose, weights, scale, aliases per frequency, prior and pixel order, with every host round trip, are computed once per data, sampling and prior; an optics change runs only kernels.
5. **Phase eigenbasis in one kernel** (`direct_eigenbasis<n>`, one thread per frequency, cyclic complex Jacobi): 1.3 ms instead of 33 ms for cuSOLVER's batched `syevj`. Its loops are fully unrolled: with runtime indices into the local arrays the optimized NVRTC build returned wrong eigenvalues for n >= 5 (correct under `-G`).
6. **One prepared system per method** (`direct_workflow.solve`): `_prepared_direct` is `(optics key, {method: prepared})`, so phase and complex at the same optics are both kept; any optics, sampling or prior change drops both.

## Conclusion

1. The 9 s was a badly shaped GEMM, then host loops; with both gone the preparation is bound by evaluating the float32 transfer at about 5e9 (pixel, alias) pairs: replacing it with a trivial function takes phase from 35 to 14 ms and complex from 46 to 22 ms. The off-diagonal sums cost 4 ms (phase) and 15 ms (complex), the strided G_qk gather 4-6 ms.
2. Float64 arithmetic cannot reach 100 ms for complex on this card (138 ms of float64 operations alone, measured rate); float32 sums with a float64 merge agree with the previous complex128 path to a few float32 ulps, the same size as two float32 evaluations of the transfer differ (point 4).
3. Complex is register bound: its 45 running sums per lane fill the register file; capping registers at 168 for 12 resident warps gave 61 -> 45 ms.
4. Bit-level agreement with the previous preparation is not possible from a fused kernel: identical transfer source compiled into a different kernel differs by one float32 ulp in 35-43 % of values (FMA contraction), which alone puts the Gram 2-3e-8 apart. The new test (`tests/hardware/cuda/test_ssb_prepared_methods.py`) keeps the previous preparation verbatim as a reference and checks Gram, right side and eigenvalues at 1e-6 of the largest entry, power at 1e-7, images at 2e-6 of their largest value (log|w| at 4 float32 ulps of 1).

## Rejected ideas

- **Materializing the transfer with the existing ElementwiseKernel, then reducing.** Keeps the transfer bit identical, but writes and reads 40-80 GB per preparation; 0.09 s of transfer evaluation alone at 1.5 TB/s.
- **Shared-memory staging of a tile of V per block, warps over entries.** Per-tile stores, ballots and barriers cost more than the products (phase 55 ms, complex 100 ms at best); the register-only warp design replaced it.
- **Skipping products whose 32-pixel column is all zero.** Saves no time once V lives in registers; the per-entry branch costs as much as the four FMAs.
- **Several frequencies per warp to share pixel loads and G sectors.** Doubles the running sums; register pressure made phase 43-107 ms.
- **Sharing a pixel tile between the warps of a block through shared memory.** Barriers across warps with different aperture-overlap work: phase 47-56 ms, complex 115 ms.
- **Splitting the entries of a complex system over several blocks.** Each block evaluates the transfer again: 70-124 ms.
- **Fast intrinsics (`__sinf`, `__cosf`).** Saved only 1-6 ms in an earlier version, at about 1e-6 absolute error in the transfer.
- **Prefetching the next pixel's G_qk one step ahead.** No change.
