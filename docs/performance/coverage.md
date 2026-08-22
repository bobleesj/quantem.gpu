# Benchmark coverage and runbooks

This page is the operational index for QuantEM.GPU performance work. It keeps
required configurations visible even when they have never run, imports every
retained atomic measurement from the current evidence index, and maps each open
gate to a repository-owned runbook.

The registry separates three concepts:

- a **coverage gate** defines the exact module, platform, computer, geometry,
  dtype, cache state, timing boundary, parity, and memory evidence required;
- a **retained measurement** records what actually ran, including p50, p95,
  maximum, resident bytes, measured peaks, swap, device, date, and revision;
- a **runbook** gives a stable command, preflight, required environment, and
  artifacts for repeating or closing one gate.

```{admonition} Status is not support by implication
:class: important
A pending row is planned work. A partial row retains useful evidence but does
not satisfy the full gate. A portable parity command never becomes physical
timing evidence, and a prepared reopen never becomes a cold-source claim.
```

## Agent entry point

Start by validating the registry, then ask it for the next open gate. Do not
guess a command from an old experiment page.

```bash
python scripts/benchmark_registry.py validate
python scripts/benchmark_registry.py next --limit 10
python scripts/benchmark_registry.py next --computer "MacBook Air (M2, 8 GB)"
python scripts/benchmark_registry.py next --platform "Python MPS"
python scripts/benchmark_registry.py show io.mps.apple-m5-max-128gb.bin2.cold-original
python scripts/benchmark_registry.py command io.mps.apple-m5-max-128gb.bin2.cold-original
```

Use `--performance-entrypoint-only` to hide gates whose current repository
entry point is parity-only:

```bash
python scripts/benchmark_registry.py next --performance-entrypoint-only
```

Before a physical run, create the experiment manifest and `experiments/RUNS.md`
row described in [Continuous profiling](continuous-profiling.md). The command
output intentionally does not launch hardware on its own: device ownership,
fixture identity, cold-source control, and output locations must be resolved
first.

## Promotion rule

A measured row needs all of the following under one comparison key:

1. the exact source and implementation revisions;
2. computer, accelerator, runtime, and device ownership;
3. source and selected scan geometry, detector geometry, bin, crop, and dtype;
4. explicit cold, warm, prepared, resident, or saved-result state;
5. run-level records with p50, p95, maximum, and the wall-clock boundary;
6. logical resident bytes, accelerator allocation/peak, process or browser-tree
   peak, total-device peak when available, pressure, and swap;
7. independent scientific parity and output fingerprints; and
8. a retained manifest, raw artifact, and terminal `RUNS.md` status.

If a field is unavailable, the row remains partial or pending. Calculated
payloads are useful planning data but never replace measured peaks.

```{include} ../_generated/benchmark_coverage.md
```
