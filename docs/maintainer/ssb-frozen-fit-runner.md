# Replaying the frozen SSB fit

Keep the original exported fixture outside the repository. Its case hash must
match the existing pin; do not regenerate the pin to accept a different input.
Set `QUANTEM_SSB_PARITY_RUNS` to the folder containing `strict-parity/` and
`QUANTEM_SSB_PARITY_PYTHON` to the interpreter with the MPS dependencies.

```sh
bash scripts/check_ssb_parity.sh --build-metal --full
```

The default checks the 128-square frozen fit. `--full` additionally repeats the
512-square native fit and records its current MPS reference. The 512 case has no
frozen trajectory pin; its production consistency and repeat determinism are
checked. This is GPU acceptance, not the separately opt-in CPU-oracle study.

Each invocation creates a new timestamped output directory. Set
`QUANTEM_SSB_PARITY_OUTPUT` to choose a new directory explicitly; existing output
directories are never overwritten. A per-user file lock serializes these gate
invocations. `QUANTEM_SSB_PARITY_LOCK` selects a shared lock location, and an
explicit `QUANTEM_SSB_PARITY_GPURUN` may wrap execution in a site's GPU scheduler.
Coordinate other applications separately: the file lock cannot reserve hardware
against unrelated processes.

Archived calibration names are translated only inside a temporary test view,
which links to the original count payload. Neither the fixture nor the pin is
edited. Current calibration exports are read directly. Scan sampling remains a
row/column pair, and all geometry comes from the fixture, not local raw-data
discovery or recalibration.
