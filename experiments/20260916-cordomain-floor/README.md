# Cordomain ticket: can the loss intermediate round trip be eliminated?

Status: **premise refuted at source level; no code change landed.** The branch
`ssb-cordomain` is base `597b566` plus this record and the probe below.

## Q1. What domain is `half_g` in, and which encode step produced it?

`half_g` is the **2D forward FFT of the detector plane, hermitian-halved along
the last axis** — i.e. the scan-frequency domain.

- Produced by `extract_hermitian_half` (`ssb.metal:378-394`), which copies the
  `col < 257` part of `full_g` plane by plane.
- `full_g` is written by `encodeForwardFFT`
  (`MetalSSBKernels.swift:1297-1314`): `convertPipeline` turns the raw packed
  counts into `float2`, then `encodeFFT` (`MetalSSBKernels.swift:1323-1420`)
  runs row FFT -> transpose -> row FFT -> transpose, i.e. a full 2D forward FFT.
- So `half_g[bf][row][col]` is `F[u_row, u_col]` with `u_col` restricted to the
  hermitian half `0...256` (`halfPlane = size * (size / 2 + 1)`,
  `MetalSSBKernels.swift:295`, `halfBytes` at `:604`).

## Q2. Is the correction a diagonal multiply in the 2D scan-frequency domain?

**Yes.** `ssb_corrected_value_precomputed_chi` (`ssb.metal:853-897`) is a pure
pointwise complex multiply `value * conj(gamma)/|gamma|` whose factors depend
only on `(row, col)` and the per-BF geometry:

- `chi_trig[row * n + col]` — a 512x512 sincos table;
- `q_row[row]`, `q_col[col]` — the frequency axes;
- `cross_trig` — a separable per-BF row/column pair, recombined by complex
  multiply (`ssb.metal:876-881`);
- `aperture_at(qx ± bg.x, qy ± bg.y)` — the two shifted aperture disks.

Therefore, for a fixed bright-field term, the whole loss is exactly

    loss = mean_pixel( Var_bf( atan2( IFFT2[ diag(t) * half_g ] ) ) )

with `t` the pointwise correction. `IFFT2[diag(t) * Gfreq]` **is** a single
inverse transform — the ticket's premise is correct so far.

It does not collapse to one pass, because the two kernels are the two mandatory
stages of that one 2D IFFT, not two separate transforms:

- `ssb_correct_half_column_ifft512_hermitian` (`ssb.metal:1129`) = the part of
  the 2D IFFT that transforms the **column axis** (input column at a time,
  correcting and preparing the hermitian completion);
- `ssb_ifft512_rows_hermitian_phase_moments` (`ssb.metal:1280`) = the part that
  transforms the **row axis** and then consumes the result as phase.

The intermediate is the half-plane between those stages. Its size is fixed by
the transform, and it is already the minimal hermitian representation
(512 x 257 complex64 = 1,052,672 B per BF). Fusing the stages needs the whole
plane on chip: 2,097,152 B against `maxThreadgroupMemoryLength` 32,768 B.

### Which term breaks the single-transform formulation?

`diag(t)` acts in the scan-frequency domain. To fold a correction into the
*other* stage's input you must express it in that stage's domain, i.e. as an
operator on the other transform — which is a dense convolver, not a diagonal.
Two of the three factors do factor cleanly and are already exploited:

- the quadratic `chi` splits into a pure row part, a pure column part and a
  mixed term — separable row x column;
- the mixed term is already precomputed as exactly that separable pair
  (`cross_trig`, `ssb.metal:876-881`).

The obstruction is the **aperture mask**, which is the only non-separable
factor:

    A(qx, qy) = R_-(qx - bx, qy - by) + R_+(qx + bx, qy + by)

with `R` the unit-disk indicator. Each disk centre couples `qx` and `qy`, so
`A` is not a product of a row function and a column function and cannot be
hoisted into either 1D stage. This is why two transforms are required.

## Q3. Measured result

No reduced-traffic formulation exists to implement, so the arm was not built.
The floor was re-verified independently in this worktree instead.

### Full fit, my own worktree, base `597b566`

    fit 65.36 s (trials 56.32 s, refine 8.08 s, 27 evaluations)
    loss 0.13769799470901489        <-- frozen reference, bit-exact
    host load 1.87 before / 2.66 after

Loss and optimum reproduce exactly; only the wall time differs, consistent with
the ~10-15% host-load sensitivity.

### Per-pass split (existing ablation arms, same binary/session, load 2.4-3.3)

| arm | p50 gpu ms |
|---|---|
| full loss (control) | **249.0** |
| skip row-axis pass (`SSB_PROFILE_SKIP_ROWS=1`) | 149.7 |
| skip column-axis pass (`SSB_PROFILE_SKIP_COLUMN=1`) | 198.2 |
| skip Nyquist boundary pass (`SSB_PROFILE_SKIP_NYQUIST=1`) | 344.8 (noise-dominated, load rose mid-run) |

Differencing against the clean control (249.0 ms):

| component | ms / evaluation | share of measured cost |
|---|---|---|
| row-axis pass + phase moments | 99.3 | 40% |
| column-axis pass + correction + blocked store | 50.8 | 20% |
| residual: Nyquist pass + encode + command-buffer cost | ~99 | 40% |

Two reservations on this split, both material:

- The residual line is a difference of differences (249.0 - 99.3 - 50.8) and is
  not a measured stage. It contains the Nyquist pass (measured separately at
  5-9 ms) plus all per-command-buffer encode and synchronization cost. It is
  not evidence of a large recoverable block, only that the two ablated passes
  do not account for the whole control.
- The Nyquist arm above ran at a higher host load than the control (its p50 is
  *above* the control's max), so no per-pass number is quoted from it.

The column-axis pass runs at 138.8 GB/s against this device's measured
~139 GB/s mixed-stream ceiling, so it is finished. The row-axis pass is the
largest single stage at 99.3 ms, but it is **not** cleanly attributed: a
transcendental-bound explanation (one `atan2` plus one square per output pixel
per BF) fits better than a device_memory-bound one, yet no ablation here separates its
`atan2`/ALU work from its memory work, and the `SKIP_ROWS` arm removes the pass
entirely rather than cheapening it. Attribution is left open rather than
asserted. Either way, cheapening `atan2` requires an approximation that changes
values and is out of scope under the no-precision-loss constraint.

## Conclusion

`28.223 GB / evaluation` (`G` read 9.41 + intermediate write 9.41 +
intermediate read 9.41) is already the minimal traffic this objective admits.
The ticket's target does not exist: the "intermediate round trip" is the
mandatory half-plane between the two stages of a single 2D inverse transform,
and both of those stages do necessary, non-duplicated work.

- Precision: unchanged, bit-exact (`0.13769799470901489`).
- 30 s: not reachable by traffic reduction. The per-evaluation work is
  irreducible at this precision; the only remaining lever of that size is a
  different objective (accumulate-first), which moves the pinned value by O(1).
