# Candidate batching: rejected, and the parity gate is too narrow

Revision: `597b5664f22347771f6c385390602f9750fd9d75` (origin/main)
Worktree: `~/perf-lab/ssb-audit/candbatch`, branch `ssb-candbatch`.

Data: ARINA `arina-fixture-a_master.h5`,
512² scan, 8937 executed brightfield terms, uint32 counts, 300 kV, 30 mrad,
0.264 Å, complex64/float32 pipeline. All GPU runs through `gpurun`
(`GPU_RUN_LABEL=candbatch`). Host load 1.97–2.14 throughout.

## Hypothesis

One ranged sweep over the cached brightfield half planes costs 266 ms and is
pure memory traffic. If `k` candidates shared that sweep, per-candidate cost
should fall by roughly `k` until the correction tables dominate. The sweep
already reads the same `G(k)` planes for every candidate; only the phase ramp
and the accumulators are candidate-specific.

## Result: rejected. No sharing, and batching is slower per candidate.

`perEvalSeconds` for the batched arm is `elapsed / chunk.count`
(`batchprobe.swift`, `runBatched()`), i.e. already per candidate. It is
therefore comparable to the sequential arm directly and must not be divided
by `k` a second time.

| mode | k | order | ranged passes | encoders | sequential p50 | batched p50 | per-candidate ratio |
|---|---|---|---|---|---|---|---|
| points | 1 | groupedPasses | 1118 | 3356 | 265.5 | 262.3 | 1.01× |
| points | 1 | interleaved | 1118 | 3356 | 266.7 | 261.0 | 1.02× |
| points | 2 | groupedPasses | 2236 | 6712 | 272.1 | 288.7 | 0.94× |
| points | 2 | interleaved | 2236 | 6712 | 271.5 | 285.1 | 0.95× |
| points | 4 | groupedPasses | 4472 | 13424 | 268.0 | 289.6 | 0.93× |
| points | 4 | interleaved | 4472 | 13424 | 268.3 | 302.6 | 0.89× |
| points | 8 | groupedPasses | 8944 | 26848 | 268.3 | 302.0 | 0.89× |
| points | 8 | interleaved | 8944 | 26848 | 266.1 | 337.0 | 0.79× |

Three reps per arm, 16 fixtures, arm order rotated per rep so neither arm owns
a fixed slot.

`ranged_pass_count` and `encoder_count` scale exactly with `k`
(1118 → 8944 passes, 3356 → 26848 encoders). Both `groupedPasses` (every
candidate's column pass, then every candidate's row pass, so the cached planes
are re-read in the shortest window) and `interleaved` reproduce the whole sweep
once per candidate. **`G(k)` is not shared and there is no reuse to exploit.**

Rejected on two counts: it does not reduce traffic, and it makes the wall clock
worse per candidate at every `k > 1` (up to 27% at `k = 8`, interleaved).

## Bit parity

`bit_identical` is true for every `planesPerRange` in {2, 8 (default), 16} at
every `k`, verified against the sequential loss bit patterns over 16 fixtures.
The design keeps per-candidate correction tables, Nyquist scratch and phase
accumulators, and visits the same brightfield ranges in the same order, so
`atan2` sees the same arguments and the phase sums see the same association.

`planesPerRange = 4` is **not** bit-identical. `pb4-k1.json`, `pb4-k2.json` and
`pb4-k4.json` all diverge at fixture index 8 in all three reps:

| | value |
|---|---|
| sequential | `0.1392689347267151` |
| batched | `0.13926894962787628` |

That is one float32 ulp, from a different phase-sum association order. The
`planesPerRange = 4` arm is rejected.

## Finding: the frozen three-fixture parity gate cannot see this

The standard acceptance gate compares three frozen losses:

| c10 | loss |
|---|---|
| 0 | `0.14511984586715698` |
| 55 | `0.13808111846446991` |
| 155.96977 | `0.13864889740943909` |

None of those is the fixture that diverges at index 8 (`0.1392689347267151`).
A change that alters the phase-sum association order — exactly the class of
change that a batching, tiling or fusion optimization introduces — can pass the
frozen gate while shipping a one-ulp difference.

**Action required before any further accumulation-order change is accepted:**
widen the gate to the 16-fixture set used here, or add a fixture whose loss
pattern differs from the frozen three. Until then the gate is not a parity
proof for this class of change.
