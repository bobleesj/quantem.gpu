# Contributing to quantem.gpu

`quantem.gpu` owns reusable accelerated IO, scientific math, GPU kernels,
resource estimation, and provenance contracts. Application UI, view state,
cache policy, and reconstruction orchestration belong in consuming projects.

## Set up a development checkout

Use Python 3.11 or newer:

```bash
python -m pip install -e ".[dev,docs]"
```

On Apple platforms, install the MPS dependencies when the change exercises the
Python MPS backend:

```bash
python -m pip install -e ".[dev,docs,mps]"
```

CUDA tests require a compatible NVIDIA driver and the `cuda` extra. Native
Swift/Metal changes require Xcode command-line tools.

## Preserve the scientific contract

Every backend uses `(row, col)` coordinates and the same source geometry,
mask, bin, dtype, precision, and result meaning. A contribution must not
silently:

- crop scan or detector coverage;
- bin detector or scan axes;
- narrow integer or floating-point precision;
- substitute CPU work for an unsupported accelerated operation; or
- present a cached, cropped, or binned result as native source data.

Record every intentional crop, bin, dtype conversion, mask, approximation, and
cache state in result provenance. See the
[scientific contract](docs/concepts/scientific-contract.md) and
[cross-backend parity rules](docs/performance/parity.md).

## Add or change a backend

Organize code by scientific domain first. Keep backend implementations private
behind the domain's public API and update
[`tests/parity/backend_matrix.json`](tests/parity/backend_matrix.json).

Follow the [backend contribution guide](docs/developer/adding-backend.md). A
backend change normally includes:

1. synthetic tests for shapes, dtypes, masks, and incomplete edge bins;
2. frozen cross-backend parity evidence;
3. physical-device evidence for the backend being changed;
4. peak host and accelerator memory; and
5. cold-source, warm-source, and saved-result timing labeled separately.

Do not widen a frozen tolerance or change a reference artifact solely to make a
new implementation pass.

## Run the checks

Run the focused test first, then the relevant suite:

```bash
PYTHONPATH=src python -m pytest -q
python scripts/check_profile_registry.py
python scripts/benchmark_registry.py validate
python scripts/check_docs_links.py
jupyter-book build docs
python scripts/check_docs_links.py --html-root docs/_build/html
python scripts/check_docs_nav_toggle.py docs/_build/html
```

For Swift and Metal changes, also run:

```bash
swift test
xcrun swift-format lint --strict --recursive \
  Package.swift \
  src/quantem/gpu/swift/Sources \
  src/quantem/gpu/swift/Tests \
  src/quantem/gpu/swift/Benchmarks
```

Hardware-gated tests may skip on a general CI runner. A skip is not backend
signoff; attach the corresponding CUDA, MPS/Metal, Swift/Metal, or browser
evidence to the pull request.

Before a new physical benchmark, choose the exact gate and repository-owned
entry point from the [coverage registry](docs/performance/coverage.md):

```bash
python scripts/benchmark_registry.py next --limit 10
python scripts/benchmark_registry.py command GATE_ID
```

Add the experiment manifest and `experiments/RUNS.md` row before launch. Keep
all repetitions and outliers. Update `benchmarks/benchmark_registry.json`, then
run `python scripts/benchmark_registry.py render`; do not edit
`docs/_generated/benchmark_coverage.md` by hand.

## Documentation and style

- Follow the [scientific writing, notation, and units](docs/developer/writing.md)
  guide and
  [`ophusgroup/dev` Appendix D](https://github.com/ophusgroup/dev#appendix-d-coding-standards).
- Use NumPy-style docstrings: state the scientific problem first, then the
  design required to interpret the result.
- State units for every physical parameter and return value. Use unit-bearing
  names when an unlabeled scalar could change scientific meaning.
- Define every mathematical symbol, unit, shape, normalization, and
  calibration source when it first appears.
- Use `(row, col)` in public APIs, metadata, plots, and error messages.
- Include corrective next steps in user-facing errors.
- Include a scientist-facing `Examples` section for public Python APIs.
- Keep generated `docs/_build/` output out of commits.
- Put public workflow guidance in `docs/`, contributor policy here, and
  implementation evidence in the owning backend's maintainer documentation.

Use modern Python type hints with built-in generics and `X | Y` unions.

## Pull requests

Keep each pull request focused on one feature, fix, backend migration, or
documentation goal. Include:

- the scientific behavior being preserved or changed;
- exact commands and environments used for verification;
- parity metrics and output artifacts;
- memory and timing evidence when performance is claimed; and
- limitations, skipped hardware gates, and follow-up work.

Do not commit private fixture paths, credentials, raw research data, generated
build output, or scratch benchmark files.
