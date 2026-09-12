# Persistent scheduler prototype

This experiment is not qualified for production. The default SSB engine keeps
the tested blocked four-row, two-pass schedule. Its unfinished persistent
scheduler is preserved in `pipelined-prototype.patch`, including diagnostics.

From the repository root, on an experiment branch with a clean tree:

```sh
git apply --check --unidiff-zero experiments/20260911-ssb-fft-cache-resident/pipelined-prototype.patch
git apply --unidiff-zero experiments/20260911-ssb-fft-cache-resident/pipelined-prototype.patch
```

Later source changes may require rebasing this patch. Before promotion, verify
timeout recovery for every cache chunk, exact numerical parity, memory bounds,
and adjacent A/B performance measurements. The existing microbenchmarks do not
establish those gates.
