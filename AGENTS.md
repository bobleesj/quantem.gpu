# Working in this repository

This repository is public. Everything committed here, and everything in its
history, can be read by anyone. These rules apply to coding agents and humans.

## Never commit

- Raw experiment or benchmark logs, manifests, JSONL, profiler dumps or run
  folders. They go to the private evidence archive (`quantem.gpu-experiments`,
  same `experiments/<id>/` layout). Write the dated summary under `docs/`
  (question, setup, numbers, conclusion) and refer to the experiment by its id.
- Datasets or detector files, other than the small synthetic fixtures under
  `tests/data/`, `tests/detector/data/` and the native test fixtures under
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

## Experiment evidence

`scripts/check_profile_registry.py`, `scripts/benchmark_registry.py` and the
tests that read retained experiments check the archive when it is available:
set `QUANTEM_GPU_EXPERIMENTS` to a checkout of it, or clone it beside this
repository as `../quantem.gpu-experiments`. Without it they validate everything
else and say that experiment evidence was not checked.

Test-specific rules are in `tests/AGENTS.md`.

## Tests

`pytest` runs the fast tier (about a minute); `pytest -m ""` runs everything,
including tests marked `slow`. Run the full set before a release.
