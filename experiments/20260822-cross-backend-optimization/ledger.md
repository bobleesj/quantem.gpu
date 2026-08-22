# Cross-backend optimization ledger

Started: `2026-08-22T02:35:24-07:00`

This ledger is append-only for the active eight-hour campaign. A result is not
accepted until its exact input, source revision, host/device, cache state,
timing boundary, run-level distribution, memory accounting, and parity output
are retained. Failed, neutral, blocked, and superseded attempts remain visible.

## Frozen checkpoint

| Field | Value |
|---|---|
| Source authority | Phil |
| Repository | `quantem.gpu` |
| Branch | `mps-load-sub500ms` |
| HEAD | `6df7237e7b90e4e6e2ee122f1082785a00ab844a` |
| Code parent | `f0f39c9158e4d8a2bd73c427cda91c25e3eddfc2` |
| Worktree state | clean |
| Dirty diff SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Frozen fixture | `512 x 512 x 192 x 192`, `uint16`, full scan, no crop |
| Accepted detector bins | explicit exact sums at 1, 2, 4, and 8 only |

The frozen preceding Apple result is full-output byte exact and reports
warm-process MPS load p50 values of 0.523, 0.498, 0.421, and 0.417 seconds for
detector bins 1, 2, 4, and 8. These are not cold-source or application E2E
claims.

## Initial physical-host audit

| Time | Host | Device | State | Decision |
|---|---|---|---|---|
| 02:32 PDT | Phil | Apple M5 Max, 40-core GPU, 128 GB unified memory | AC power; no throttled pages; pre-existing desktop/virtualization load retained | Available for isolated MPS/native profiling |
| 02:32 PDT | Rodman | Apple M5, 10-core GPU, 24 GB unified memory | AC power; pre-existing 1.17 GiB swap and service relay retained | Available only with measured admission/memory guard; no app policy or GUI takeover |
| 02:32 PDT | MJGOAT GPU 0 | RTX PRO 6000 Blackwell, 96 GB | 100% utilization; unrelated process owns about 76.8 GiB | Blocked; audit retained evidence only; do not schedule GPU work |
| 02:32 PDT | MJGOAT GPU 1 | RTX PRO 6000 Blackwell Max-Q, 96 GB | 100% utilization; unrelated process owns about 54.3 GiB | Blocked; audit retained evidence only; do not schedule GPU work |

MJGOAT's ordinary checkout is on `main` at
`4f89e08c31ea394098a971750500b9b8cf8fb7d7` with two unrelated dirty files.
That worktree is preserved exactly and is not an editable campaign surface.

## Priority evidence gaps at start

| Cell | Start state | Owner rule |
|---|---|---|
| Swift/Metal decode, bin, provenance | evidence gap | Phil or Rodman native isolated worktree |
| Swift/Metal detector integer products | evidence gap | exact integer reference required |
| CUDA prepared screening products | evidence gap | MJGOAT only after an uncontended GPU is proven |
| Swift/Metal CoM, rotation, and iDPC | evidence gap | frozen row/column and float-parity contracts |
| Hardware-WebGPU CoM, rotation, and iDPC | evidence gap | genuine hardware adapter; software smoke is not evidence |
| MPS 200-trial plus Nelder-Mead SSB calibration | evidence gap | Phil physical MPS, deterministic seed/history |

## Event log

| Time | Trial | State | Observation / next action |
|---|---|---|---|
| 02:35 PDT | campaign-checkpoint | accepted | Frozen clean authority and host ownership before benchmark execution. |
| 03:31 PDT | phil-webgpu-dpc | evidence-gap | Physical Apple Metal-3 loaded the full `512 x 512 x 192 x 192 uint16` detector-bin-1 source with exact counts and DPC. Prepared-index/source-pages-uncontrolled load profile was 1.294 s; warm product p50 was 0.7 ms row DPC, 0.6 ms column DPC, and 1.1 ms iDPC. The iDPC gate remained open: 4,332/262,144 values exceeded the frozen `1e-5` tolerance, maximum absolute error `5.3406e-5`. No tolerance was relaxed. Clean evidence revision `f218e70ea50273240b6186637a8a574b6608ab6c`. |
| 03:31 PDT | phil-mps-ssb-calibration | blocked-provenance | The retained detector-bin-2 BF companion was refuted by a fresh exact load: 596,089,086/596,377,600 integer values differed. The harness failed closed before optimization; zero of 200 TPE trials ran. Clean evidence revision `f218e70ea50273240b6186637a8a574b6608ab6c`. |
| 03:34 PDT | rodman-native-metal | accepted-bounded | Rodman clean revision `0b305f69fc1039933463986a0f74609e22d4dd35`: full-native detector-bin-1 was admitted once at 3.827780 s wall / 0.881714 s GPU with exact full-volume and product parity, but increased swap from 1,193.25 to 2,436.44 MiB, so repetition was correctly stopped. Exact detector-bin-2 reused-destination p50/p95/max was 0.683492/0.690893/0.690893 s; bin-4 was unexpectedly slower at 0.813422/0.829663/0.829663 s. Bin-8 is unsupported and fails closed. These are prepared-index, source-page-unspecified package measurements, not cold-source or app E2E. |
| 03:38 PDT | phil-mps-metal-system-trace | accepted-diagnostic | A valid Metal System Trace of exact full-native bin-1 isolated 0.242712 s of GPU interval union and 19,937,378,304 B of driver wire events. Allocation API time was negligible; first-use shared-memory wiring and queue latency overlap the remaining wall. This explains why arithmetic-only tuning did not move the roughly 0.5 s package wall. Evidence revision `d58ac86`; summary SHA-256 `deef3d2e8e71f289d660424033fdd617efb05b55193fc7c3192c8629b2614d2f`. |
| 03:49 PDT | phil-private-residency-pair | promising-not-yet-accepted | One fresh-process shared/private pair at revision `531e5001f5a0a886c058f772ef42d770f000890b` preserved the complete 18 GiB SHA-256 and all seven product hashes. Shared/private wall was 0.516492/0.309717 s; GPU was 0.245242/0.253917 s; process peak RSS was 19.478/0.685 GB. Readback/hash was excluded from wall. This is a single prepared-index/source-pages-unspecified pair only; a bounded ABBA distribution is running before acceptance. Comparison SHA-256 `35870e5aa410db6ea31058dd04847a444fccba5c093a2977ec6bea826730b556`. |
| 03:50 PDT | mjgoat-cuda-ownership | blocked-resource | Both CUDA devices remain owned by unrelated processes. Retained historical screening timing has no reusable result/stdout artifact for the current revision, and the frozen candidate lacks H2D/decode-kernel/synchronization/D2H/mask-validation/cache-write timing splits. No CUDA GPU work was launched and no unrelated owner was displaced. |
| 04:01 PDT | phil-private-residency-abba | accepted | Clean revision `afd5e06b2afa2c9c30e69ff5dc8d7825c28068f1` preserved the complete 18 GiB volume hash and seven exact product hashes in 12/12 ABBA loads. Shared/private p50 was 0.530581/0.313870 s (1.690x, -216.711 ms); GPU p50 0.245472/0.254773 s. Private residency reduced process RSS p50 from 19.393 GB to 0.685 GB while Metal accounting rose from 19.337 GB to 19.941 GB; swap was unchanged. This is warm-page-state package evidence, not cold HDF5 or app E2E. |
| 04:14 PDT | phil-mps-bin4-specialization | accepted | Clean source `0a0d60eef298190597778dbf11bb5b79f97a3615`, evidence `6ed319b80387290d135123f26addc90eb16ec22a`: a dedicated 4 KiB exact bin-4 tile reduced fresh-process warm-page-state p50 from 0.421165 to 0.383772 s (8.88%; n=6/arm). The complete 1,207,959,552-byte output and all retained hashes were byte exact; full suite 489 passed / 87 skipped / 0 failed. |
| 04:18 PDT | mjgoat-cuda-owner-guard | accepted-guard, hardware-blocked | Clean guard revision `05d6bbf7ca367f47bb65a88c87e3ad568563d646` passed seven ownership cases. GPU 1 briefly appeared free, but unrelated PID 24670 reacquired it before the authorized smoke; preflight exited 75 before CUDA context or output creation. GPU 0 remained occupied. Candidate `7902c68` stays hardware-unqualified; 78 portable and 19 parity/API tests pass with hardware cases skipped. |
| 04:38 PDT | phil-mps-exact-range | accepted | Clean source `8045d4e708d717e7cdb6ac6ae853a0b47d1fae36`, evidence `72df01c`: fused bin 2/4/8, scalar `uint16`/`uint32`, selective, and fast-sidecar detector sums now fail closed before publishing unrepresentable counts; explicit `uint8` clipping remains declared. Full suite 501 passed / 87 skipped / 0 failed. Bin-4 ABBA measured a bounded 1.387 ms p50 safety cost (0.36%) with exact hashes and unchanged memory geometry. |
| 04:39 PDT | rodman-bin2-barrier-followup | preserved-provisional | A two-barrier exact bin-2 follow-up passed 115 tests / 8 skips and strict formatting, but Rodman remained contaminated by Screen Sharing/system GPU activity. No timing or speed claim was made; clean accepted `34782081` remains the integration boundary and the provisional patch is sealed as SHA-256 `3324de94...`. |

## Current measured matrix

These rows use different devices and timing boundaries and are intentionally not
ranked as though they were interchangeable. A dash means the required evidence
does not yet exist.

| Module | Platform / computer | Geometry | Source state | Samples | Wall p50 | GPU p50 | Peak memory | Scientific gate |
|---|---|---|---|---:|---:|---:|---:|---|
| Load + exact products | Swift/Metal / Rodman M5 24 GB | bin 1, full native | prepared index; pages unspecified | 1 | 3.827780 s | 0.881714 s | 19.940737 GB Metal; 20.033504 GB footprint; 0.669876 GB RSS; swap +1,243.19 MiB | exact; repetition stopped for pressure |
| Load + exact products | Swift/Metal / Rodman M5 24 GB | bin 2, `512 x 512 x 96 x 96 uint16` | prepared index; pages unspecified | 6 reused | 0.683492 s | 0.663008 s | 5.445222 GB Metal; 5.563729 GB footprint; 0.694780 GB RSS; swap +0 | exact |
| Load + exact products | Swift/Metal / Rodman M5 24 GB | bin 4, `512 x 512 x 48 x 48 uint16` | prepared index; pages unspecified | 6 reused | 0.813422 s | 0.799494 s | 1.827635 GB Metal; 2.071970 GB footprint; 0.714670 GB RSS; swap +0 | exact; superseded below by fused follow-up |
| Load + exact products | Swift/Metal / Rodman M5 24 GB | bin 4, `512 x 512 x 48 x 48 uint16` | prepared index; pages unspecified | 14 ABBA | 0.631540 s | — | policy/product memory unchanged; exact evidence sealed | exact; 23.07% reduction; clean `34782081` |
| Full-native decode + products | Python MPS / Phil M5 Max 128 GB | bin 1, full native private Metal | warm OS pages; fresh-process ABBA | 6/arm | 0.313870 s | 0.254773 s | 19.941 GB Metal; 0.685 GB RSS; swap +0 | complete 18 GiB plus seven products exact |
| Exact detector-bin load | Python MPS / Phil M5 Max 128 GB | bin 4, `512 x 512 x 48 x 48 uint16` | warm OS pages; fresh-process ABBA | 6/arm | 0.383772 s | — | 1.208 GB logical; 2.484 GB Metal; ~0.614 GB RSS | complete output byte exact |
| CoM/DPC/iDPC | WebGPU / Phil Chrome Apple Metal-3 | bin 1, full native | prepared index; pages uncontrolled | 1 full profile + 7 warm products | 1.294 s load profile | warm products only | 6.791889 GB Chrome RSS; swap +0 | counts and DPC exact; iDPC fails frozen tolerance |
| Screening | CUDA / MJGOAT | full native | — | 0 current | — | — | — | blocked by unrelated GPU owners |
