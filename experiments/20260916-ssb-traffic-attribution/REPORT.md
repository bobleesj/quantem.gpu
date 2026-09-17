# Per-kernel cost attribution for one production SSB evaluation

Revision: `597b5664f22347771f6c385390602f9750fd9d75` (origin/main) with
`traffic-gates.patch` applied to the working tree (experimental instrumentation,
not production; the patch wraps existing encoders and adds no arithmetic).

Data: ARINA `arina-fixture-a_master.h5`
(sha256 `57cb9490…cf95b`), 512² scan, 8937-term brightfield disk, uint32 counts,
complex64/float32 pipeline, 300 kV, 30 mrad, 0.264 Å scan step.

## Method

Three independent instruments, each with its own failure mode:

1. **Interleaved fractional ablation (`probe_traffic.swift`).** One process, one
   `prepare`, then `full`, `*_half` (pass encoded on every second cached batch
   dispatch), `*_none`, and `floor` arms cycled round-robin for N reps. Every arm
   runs the production kernels; arms differ only in how many dispatches encode a
   pass. Load drift is common-mode, so within-rep paired deltas are the estimator.
   Gate: the `full` arm must reproduce the frozen probe losses bit-for-bit.
2. **Pattern-pricing skeleton (`ghost_traffic.swift`).** A standalone Metal probe
   with the production dispatch geometry (1118 dispatches of 257 x 8 threadgroups
   x 64 threads, 8 slots/thread), the production bit-reversed 4 KB-run read, the
   production `STORE_BLOCKED` store, a 9.42 GB read window and the production
   8.4 MB circular intermediate — with no FFT and no corrections.
3. **In-situ footprint and host-load controls.** `SSB_PHASE_BATCH` 8 vs 32
   (intermediate 8.4 MB vs 33.7 MB) and 8 CPU hogs, both against the same arms.

Every GPU run went through `gpurun` with `GPU_RUN_LABEL=traffic`. Sessions:
`r1` (1 rep), `r2` (2), `r6` (6), `pb` (4), `load` (4, 8 CPU hogs) →
17 paired samples per main arm. Losses were bit-identical to the frozen pins in
all five sessions (`c10` 0 → `0.14511984586715698`, 55 → `0.13808111846446991`,
155.96977 → `0.13864889740943909`).

## Result: split of one 266 ms evaluation (p50, clean sessions 262.9–272.1)

| component | ms | share | basis |
|---|---|---|---|
| column pass — G(k) read, 9.42 GB, production pattern | 80 | 30% | ghost_read 117.7 GB/s |
| column pass — intermediate store (marginal, sharing with the read) | 68 | 26% | 147.8 − 80; alone 59 ms @159.5 GB/s |
| row pass — intermediate read, 9.42 GB, contiguous | 61 | 23% | ghost_row_read 153 GB/s |
| row pass — `atan2` over 2.34e9 values, Nyquist reads, moments | 61 | 23% | 121.7 − 61 |
| Nyquist correction pass | 10.4 | 4% | paired ablation, ±1.0 |
| framework: fills, clears, host loss reduction, finalize | 8.2 | 3% | `floor` arm |
| per-eval chi/cross trig rebuild | ~0 | 0% | `trig_none` −0.1 ± 3.9 |
| inter-pass overlap | −22 | −8% | additivity defect |
| **total** | **266** | | |

Paired marginal costs (mean of within-rep deltas, n = 17):

| arm | mean Δ vs full | median | stdev | implied full cost |
|---|---|---|---|---|
| `col_none` | −147.8 | −148.0 | 3.1 | 147.8 ms |
| `col_half` | −72.0 | −72.7 | 2.9 | 144.0 ms |
| `row_none` | −121.7 | −121.9 | 4.3 | 121.7 ms |
| `row_half` | −62.9 | −62.3 | 3.7 | 125.9 ms |
| `nyq_none` | −10.4 | −9.7 | 4.0 | 10.4 ms |
| `nyq_half` | −5.1 | −5.0 | 2.5 | 10.1 ms |
| `trig_none` | −0.1 | −0.9 | 3.9 | ~0 |
| `floor` | −259.3 | −258.7 | 3.7 | framework 8.2 ms |

Half-fraction arms are linear within 3%, so each pass composes with its own work
fraction; the non-additivity is *between* passes (Σ costs 279.9 ms vs 258 ms of
executed work). Host-side encoding is 1.6–1.9 ms per evaluation
(`wall − gpu`), so the main thread is not a factor.

## Nyquist: the +90 ms observation was contamination, not work

The `SSB_PROFILE_*` gate wraps exactly the `nyquistCorrectionPipeline` encoder;
it does skip the work (loss moves to `0.13864953815937042` at c10 = 155.96977,
consistent with reading the previous candidate's stale correction buffer). Its
real cost is **10.4 ms/eval** (7.3–10.4 across sessions, ±1.0 paired).

Why the earlier 343.0/345.7 ms reading is not a property of the correction:

* that sweep ran 17:06:30–17:07:02 — six arms in one 32-second window, and
  `ablate.sh` calls the probe binary directly, not through `gpurun`;
* the *identical* `full` arm measured 275–362 ms (median 297) at 17:06:30 and
  247–253 ms (median 248) at 17:07:02: ±20% drift on unchanged code inside the
  same sweep;
* the MPS agent (`mps/experiments/20260916-ssb-mps-hotpath/`) and the parity
  agent (`parity/build/ssb-parity-check`, MPS kernel caches) were built and run
  in that exact window — a second GPU client on the same device;
* the samples' internal trends fit a transient load that started mid-sweep
  (`ablate-full` rising 275→362) and ended before the final arm (`ablate-full-2`
  flat at 248–253), with the no-Nyquist arm caught in the middle;
* re-measured 17 times under the lock, paired and in-process, the effect is
  +10.4 ms with a 4 ms stdev.

The same contamination explains why the old ablations do not compose
(134 + 199 > 253–290): they were single-shot, unlocked, and taken while the
window drifted by ±20%. They should not be used for cost attribution.

## Is the evaluation device_memory-bandwidth-bound? No.

1. The 28.2 GB nominal traffic moves at 106 GB/s, against a **measured** ceiling
   in the same session of 143.4 GB/s (streaming read) and 143.9 GB/s (streaming
   write). The machine is not at its device_memory limit.
2. **The 18.84 GB intermediate round trip is not device_memory traffic.** 4× the
   footprint (`SSB_PHASE_BATCH` 8 → 32, i.e. 8.4 → 33.7 MB, past any plausible
   cache) changes the evaluation by **−0.2 ms** at bit-identical losses (n = 4
   paired). Its cost is on-chip path throughput: 159.5 GB/s store, ~153 GB/s
   load, invariant to footprint.
3. The column pass runs at 18.84 GB / 147.8 ms = **127 GB/s**, i.e. at or above
   the ghost's own mixed read+write ceiling (122–128 GB/s) for the identical
   pattern. Its FFT arithmetic and its 28 GB/eval of logical correction-table
   reads are fully hidden behind that data movement.
4. The row pass runs at 9.42 GB / 121.7 ms = 77 GB/s, so ~61 ms of it is
   instruction work (`atan2` on 262144 pixels x 8937 terms plus the per-element
   Nyquist reads), not memory.

So the evaluation is ~80% memory-*path* bound (device_memory plus on-chip movement, at
~88–100% of the achievable rate for the actual pattern mix) and ~20%
instruction-throughput bound in the row pass. The earlier "104 vs 121 GB/s" gap
resolves into: (a) the 121 GB/s reference is a read-only streaming number, and a
read+write mix at this pattern tops out at 122–128 GB/s; (b) 61 ms of the
evaluation is not memory at all.

## Ranked, lossless, measured headroom

| change | measured effect | status |
|---|---|---|
| Fewer TPE trials (protocol, not kernel): 100 + NM = 34.3 s, 25 + warm NM = 22.2 s vs 200 = 62.3 s | −45% / −64% fit wall, bit-identical at 100 | already measured; needs a decision |
| Coalesced (unblocked) intermediate store | blocked 151.7–154.0 vs coalesced 146.5–147.0 ms in the skeleton; ~1 ms in the isolated in-situ store test | worth ~0.4–2.6% |
| Staged/coalesced G read | read alone 80 → 66 ms (117.7 → 143 GB/s), but ~0 in the mixed kernel; staged+coalesced skeleton 148.8–149.2 ms vs 151.7–154.0 | worth ~1–2% |
| Shrinking or re-representing G(k) | G is already the r2c/hermitian floor (1.05 MB per term, the same as a plain real plane); no lossless form is smaller | refuted |
| Removing the intermediate round trip | needs 2 097 152 B on chip vs 32 768 B threadgroup memory | refuted |
| Cheaper phase than `atan2` | changes numerics; strict float32 parity is a hard requirement | out of scope |

**120 ms per evaluation is not reachable on this device.** Moving the
unavoidable 28.2 GB at the measured streaming ceiling would take 196 ms with
zero arithmetic and zero overlap loss; the measured floor with the row pass's
instruction work included is ~245–257 ms, and the evaluation is at 266 ms.
The only 1.8–2.8× available is buying fewer evaluations (protocol) or a faster
GPU (CUDA/RTX track).

## Rejected ideas this data refutes

* "The blocked store costs 4x" (160 vs 640 GB/s microbenchmark): not in situ.
  In the production configuration the intermediate is an 8.4 MB circular buffer;
  blocked and coalesced stores both run at ~160 GB/s and a 4x larger footprint is
  free. The earlier number priced a large-footprint store pattern.
* "Skipping the Nyquist correction makes the evaluation ~90 ms slower": an
  artifact of an unlocked, single-shot sweep during concurrent GPU work.
* "Host load moves p50 by 10–15%": 8 CPU hogs move the full arm by +7 ms
  (266 → 273, +2.6%); the column, row and floor arms do not move. The large
  swings need a competing *GPU* client, which is why `gpurun` matters.

## Limitations

* Single device (Apple M5, 24 GB), single dataset, single control point per arm.
* The read window in the ghost is 1.18 GB re-read 8x rather than the production
  9.42 GB read once; both are device_memory-served, but row-buffer locality may differ.
* The 2.34e9-value `atan2` attribution is a residual, not a direct count.
* The `pb32` footprint test conflates footprint with dispatch count (1118 -> 280);
  both are neutral here, but it cannot separate them.
* All arms were measured with the machine under a GPU lock shared with a
  coalesced-store agent, i.e. occasionally waiting for the lock; load1 is
  recorded per sample (1.42–4.85 across sessions).
