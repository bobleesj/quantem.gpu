# Evidence-backed readiness

The following view is generated from the existing capability matrix, profiling
matrix, and benchmark registry. Implementation, retained evidence, and exact
performance records remain separate. Query or regenerate it with:

```bash
python scripts/backend_status.py check
python scripts/backend_status.py summary
python scripts/backend_status.py json --backend vulkan
python scripts/backend_status.py render
```

A profiling cell may be marked `ready` only after its `retained_evidence` list
links structured run-evidence JSON using repository-relative `path` and
`sha256` fields. The run artifact must contain:

- integer `schema_version: 1` and protocol `quantem-gpu-cell-evidence/v1`;
- `cell_id`, `backend`, and `runner` matching the profiling cell exactly;
- `status: passed` and the full 40-character `source_revision`;
- a `fixture` object with an immutable `id` and 64-character `sha256`;
- a `result` object naming retained output by repository-relative `path` and
  `sha256`; the output must exist and its digest must match; and
- `outcomes` mapping each required gate name to `passed`. Required gates are
  the cell's `pr_gate` plus every outcome in the profiling matrix's
  `required_evidence.required_outcomes`, currently `scientific-parity` and
  `real-data-e2e`.

The profile matrix owns that evidence policy; this artifact is a retained run,
not another registry. Existing generic experiment manifests remain retained,
but `completed` alone cannot be promoted: some completed runs contain failed
scientific gates. They need a reviewed cell-specific run record before use in
signoff. A correctly hashed test source file is rejected as run evidence.

Human scientific adjudication is still required: a structurally valid record
does not prove its measurements are truthful or cover the entire cell scope.
Review its fixture, output, exact revision, runner, and gate outcomes against
the scientific contract. Failed and historical experiments remain retained in
the experiment and benchmark ledgers; do not relabel them as passed signoff
evidence. Compilation or a test's mere existence never substitutes for a run.

When this check was introduced, 31 legacy `ready` declarations had no
cell-scoped immutable signoff evidence linked. They were corrected to
`evidence-gap`, with an explanation in each profiling cell. This is an
evidence-accounting correction, not a backend regression: existing test gates,
implementation levels, frozen results, and historical benchmark measurements
were preserved. Broader package signoff requires the full cell scope; a
retained result for one shape, dtype, or device does not establish it.

Historical timings and exact comparison conditions remain in the
[benchmark ledger](coverage.md). See the [profiling protocol](continuous-profiling.md)
for required runners and the [scientific parity contract](parity.md) for
numerical acceptance.

```{include} ../_generated/backend_readiness.md
```
