# Working in this repository

This repository is public. Everything committed here, and everything in its
history, can be read by anyone. These rules apply to coding agents and humans.

## Where code belongs

- **quantem.gpu provides algorithms and backends only**: DirectPtycho, `series`, MAPED and the other
  reconstructions; GPU backends such as `quantem.gpu.streaming.LiveEncoder` (ANS encoding of incoming batches),
  `io.load` / `io.save` and the kernels. It holds no streaming code: no detector connections, receivers,
  hand-off queues, listener threads or acquisition lifecycle (start, wait, close, keep last N).
- **Receivers and the acquisition lifecycle live in live4d** (Holoscan RDMA/ZMQ receive), which calls
  `LiveEncoder` for each batch.

## Never commit

- Raw experiment or benchmark logs, manifests, JSONL, profiler dumps or run
  folders. They go to the private evidence archive (`quantem.gpu-experiments`,
  same `experiments/<id>/` layout). Write the dated summary under `docs/`
  (question, setup, numbers, conclusion) and refer to the experiment by its id.
- Datasets or detector files, other than the small synthetic fixtures under
  `tests/data/` and the native test fixtures under
  `native/swift/Tests/**/Fixtures/`.
- Images, figures or notebook outputs of any data other than the public gold
  nanoparticle dataset. Gold is the only dataset that may appear in this
  repository, in `docs/_static/`, in notebook outputs and in the docs. A figure
  of any other acquisition, however anonymous, is not committed; keep it in the
  private evidence archive and describe the result in words and numbers.
- Local paths (home directories), machine host names, or tailnet addresses.
- Names of collaborators, companies or people, other than the authors in
  `CITATION.cff`. Name data by its public dataset id.
- Credentials or tokens.
- Files over 5 MB, other than the vendored HDF5 library
  (`native/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a`).

`tests/infrastructure/test_public_repository.py` enforces these rules. It is
fast and offline, and it must pass before every push:

```bash
python -m pytest -q tests/infrastructure/test_public_repository.py
```

If unsure whether something is private, leave it out and ask.

Test-specific rules are in `tests/AGENTS.md`.

## Display API naming

Follow the shared standard in quantem.widget's
`docs/maintainer/display-api.md` when adding or changing display APIs. Use named
`size` presets (default `"medium"`) for each image panel and plain `rows` /
`columns` for gallery layout. Keep renderer dimensions out of beginner calls;
reuse the visualization layer. Preserve published keywords with a documented,
tested compatibility migration and verify signatures before changing tutorials.
Display options must not change scientific data or reconstruction quality.

## Benchmarks

For performance work, timing claims and release qualification, use the
repository's [gpu-benchmark skill](.agents/skills/gpu-benchmark/SKILL.md).
Keep reusable runners and reviewed summaries here, with raw evidence in the
private archive. Prefer the installed `quantem-gpu benchmark-ssb` command for
full-field and regional joint SSB fits.

## Tests

`pytest` runs the fast tier (about a minute); `pytest -m ""` runs everything,
including tests marked `slow`. Run the full set before a release.
