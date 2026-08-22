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
