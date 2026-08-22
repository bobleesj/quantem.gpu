# Continuous profiling

QuantEM.GPU treats profiling as a scientific experiment. Continuous profiling
does not mean running every private fixture on every pull request. It means that
every layer has a defined cadence, every measurement has a comparison key, and
no timing can outrun its parity evidence.

```text
code change
    |
    v
PR smoke: contracts, small parity, compilation
    |
    v
scheduled physical profile: frozen device + frozen configuration
    |
    v
same-key comparison: parity first, then p50/p95/max and memory
    |
    v
manual signoff: real full source + minimum-memory + application wall clock
    |
    v
promote one atomic row to the dashboard
```

The machine-readable schedule is
[`benchmarks/profile_matrix.json`](https://github.com/bobleesj/quantem.gpu/blob/main/benchmarks/profile_matrix.json).
It contains one cell for every capability/backend pair in the
[parity matrix](parity.md), including unsupported cells and known evidence
gaps. The registry is validated in PR CI; physical timing remains on the named
hardware owner.

The schedule is intentionally coarse. Exact computer, scan, detector, bin,
dtype, cache state, timing boundary, memory, and parity gates live in the
[benchmark coverage registry](coverage.md). That registry includes unrun rows
and maps every open gate to a repository-owned command. Query it instead of
copying an old experiment command:

```bash
python scripts/benchmark_registry.py next --limit 10
python scripts/benchmark_registry.py show GATE_ID
python scripts/benchmark_registry.py command GATE_ID
```

## Four execution tiers

| Tier | Runs when | What it proves | What it cannot prove |
|---|---|---|---|
| **PR smoke** | Every pull request | Imports, API contracts, package contents, source compilation, small synthetic parity, and explicit unsupported behavior | Hardware speed, minimum-memory support, true cold storage, or application wall time |
| **Scheduled profile** | Weekly when the owning physical runner is available, and after a relevant IO/kernel change | Drift on a frozen source, device, runtime, cache state, precision, and timing boundary | Release readiness on a different device or fixture |
| **Release signoff** | Manually before promoting a performance claim or release | Real full-source parity, cold/warm/prepared states, minimum-memory behavior, and application first-usable-product wall time | A universal ranking across unrelated hardware or fixtures |
| **Diagnostic** | As needed for a falsifiable hypothesis | Stage attribution using traces, counters, or isolated variants | An accepted dashboard baseline until parity and signoff pass |

PR CI never fails because one shared runner was temporarily slower. It fails on
scientific parity, broken manifests, missing cells, packaging, compilation, and
contract regressions. Timing becomes a blocking gate only after at least five
accepted sessions establish a stable same-key device baseline.

## Hardware ownership

| Platform | Profiling owner | Scheduled responsibility | Minimum-device signoff |
|---|---|---|---|
| CPU reference | MacBook Pro (M5 Max, 128 GB) portable CPU | Independent exact adjudication; never a silent fallback | Not applicable |
| CUDA | Linux CUDA workstation (dual 96 GB Blackwell GPUs) physical CUDA | Load/decode, resident products, screening, SSB reconstruction, and deterministic calibration | Physical or equivalently constrained 6 GiB dedicated-VRAM run, including other card occupants |
| Python MPS | MacBook Pro (M5 Max, 128 GB) physical MPS | Load/decode, products, screening, SSB, process footprint, pressure, and swap | Physical 8 GB unified-memory run |
| Native Swift/Metal | MacBook Pro (M5 Max, 128 GB) physical native toolchain | Swift tests, Metal compilation, QH5/native IO, FFT/display, SSB reconstruction/loss/calibration, and package-owned numerical kernels | Physical 8 GB unified-memory run plus application signoff where the claim is wall-to-wall |
| WebGPU | MacBook Pro (M5 Max, 128 GB) hardware browser | Browser load/decode, products, DPC/iDPC, SSB reconstruction, adapter limits, browser-tree memory, and device loss | Physical 8 GB laptop run; software/fallback adapters never qualify |

The physical 8 GB Apple gate is coordinated: only one memory/GPU campaign may
own the device at a time. A larger Apple device, Chrome RSS estimate, prepared
cache, or calculated payload remains a pre-check rather than signoff.

## The comparison key

Two timings are comparable only when all comparison-key fields agree:

- protocol version and platform/module cell;
- source identity, shape, dtype, selected region, and bad-pixel policy;
- scan/detector crop and bin;
- mask/calibration signature, objective, and numerical precision;
- cache state and wall-clock boundary;
- physical device identity and runtime/driver versions; and
- exact source revision.

Changing one field creates a new row. Cross-key comparisons may be discussed as
separate experiments, but they cannot trigger a regression or replace the
current baseline. In particular, detector bin 4, a scan crop, a saved-result
reopen, and a native-detector source are four different scientific states.

## Run lifecycle

Before a run starts:

1. Choose an exact gate with `benchmark_registry.py next` and inspect its
   runbook with `benchmark_registry.py command GATE_ID`.
2. Write a falsifiable question.
3. Add a `running` row to
   [`experiments/RUNS.md`](https://github.com/bobleesj/quantem.gpu/blob/main/experiments/RUNS.md).
4. Create `experiments/<id>/manifest.json` with the source revision, dirty diff,
   input identities, parameters, host/device, cache state, and planned outputs.
5. Check accelerator ownership and run one smoke configuration.
6. Stop immediately if the smoke violates parity, memory admission, or device
   ownership.

When the run ends:

1. Set status to `ok`, `failed`, `refuted`, or `superseded`; never delete a
   failed hypothesis.
2. Retain run-level timings, memory samples, output/parity hashes, profiler
   artifacts, and exact commands.
3. Record p50, p95, and maximum without hiding initialization outliers.
4. Update the detailed evidence ledger first.
5. Promote a dashboard row only if the scientific and physical-device gates
   pass.

Validate the plan and registry locally with:

```bash
python scripts/check_profile_registry.py
python scripts/benchmark_registry.py validate
PYTHONPATH=src python -m pytest -q \
  tests/test_profile_registry.py \
  tests/test_benchmark_registry.py
```

## Regression decisions

The comparison result has four possible states:

| State | Meaning | Action |
|---|---|---|
| **Pass** | Parity passed and the same-key distribution stayed within its established device envelope | Retain the run; promote only if it is a signoff configuration |
| **Investigate** | Parity passed but p50, tail, or memory changed outside the established envelope | Repeat twice on the same source/device, inspect ownership and counters, and keep both records |
| **Block** | Scientific parity, provenance, memory admission, device ownership, or deterministic-repeatability gate failed | Stop the matrix; diagnose before further profiling |
| **No baseline** | The cell is new or has fewer than five accepted sessions | Record it in report-only mode; do not invent a universal percentage threshold |

A threshold belongs to one exact comparison key. Global rules such as “fail any
10% slowdown” are deliberately avoided because device scheduling, browser
versions, storage state, and unified-memory pressure have different variance.

## Current automation boundary

The registry validator and portable parity suite run in PR CI now. The physical
CUDA, MPS, Swift/Metal, WebGPU, and minimum-memory campaigns are intentionally
not assigned to generic GitHub-hosted runners. They become scheduled jobs only
after each specified physical runner has stable device ownership, fixture access,
artifact retention, and a fail-closed preflight.

This prevents an unavailable runner, software WebGPU adapter, occupied GPU, or
missing private fixture from producing a false green performance claim.
