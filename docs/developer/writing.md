# Scientific writing, notation, and units

QuantEM.GPU follows the coding and documentation conventions in
[`ophusgroup/dev` Appendix D](https://github.com/ophusgroup/dev#appendix-d-coding-standards).
Write first for scientists using Python notebooks and scripts. Lead with the
shortest public workflow and the resulting scientific image or array. Present
Apple Silicon MPS setup before NVIDIA CUDA setup. Keep kernel implementation,
native application integration, and benchmark machinery in later sections.

Python is the user interface; MPS and CUDA are execution backends. Document one
public API where supported, and state backend-specific limits explicitly. On
implementation pages, define the scientific contract, map it to source paths,
and explain how to verify it.

## Documentation structure

Use four visible sections: **Start here**, **Python workflows**, **API reference**,
and **Developer guide**. Show installation for MPS before CUDA. Keep native APIs,
file specifications, performance evidence, and historical records nested beneath
the developer guide; expand only the branch a reader opens.

A workflow answers one scientist's task: minimal code, a real image or result,
then optional controls. A reference page owns signatures, units, return values,
errors, and limitations. A developer page owns equations, layouts, source maps,
and verification. Link between those owners instead of copying their inventories.
Retain short README examples and use its existing figures when they illustrate
the same calls. Keep experimental and historical instructions out of first-run
installation. Preserve their dated evidence in the archive.

## Audiences and page types

Use the page type that matches the reader's question:

| Reader question | Page type | Required content |
|---|---|---|
| How do I load data and get an image? | Python workflow | minimal calls, automatic defaults, calibration, output, optional overrides |
| How do I run it on my computer? | Installation | Python first, Apple Silicon MPS then NVIDIA CUDA, verified dependencies |
| What does this operation compute? | Scientific kernel | equations, coordinates, shapes/dtypes/units, optimization model, source map, parity gates |
| How do I implement it on my device? | Kernel implementation | runtime boundary, sources, memory/execution model, build, profiling, acceptance |
| What may my code call and rely on? | API contract | typed inputs/outputs/errors, provenance, minimal example, ownership |
| Is this result correct and faster? | Evidence | revision, device, source and plan, cache state, memory, statistic, parity artifact |

Primary documentation must not depend on a consumer UI framework. Application
developers need integration ownership—what this package computes and what the
client schedules or presents—not instructions for a particular screen or
viewer. Product-specific details belong in that product's documentation unless
they are necessary historical provenance.

## Scientist-facing examples

Start with the current short workflow: `io.load(path)`, array indexing,
`detector.bf(data)` / `detector.adf(data)`, and
`ssb = SSB(data, ...)`, `aberrations = ssb.find_aberrations()`, then
`ssb.reconstruct(aberrations)`. Call the fitted result `aberrations`. Show automatic defaults
first, then pixel-radius and center overrides. Keep `detector.prepare` in
advanced integration examples that need its exact/native output contract.

Use `Dataset4dstemGPU` for the acquisition handle. Describe what returns a
Torch tensor and what returns a NumPy product. Show `.sampling`, `.units`,
`.origin` and `.metadata`; missing physical calibration stays explicit.

For static `show_2d` examples, set the notebook's colormap once:

```python
from functools import partial
from quantem.core.visualization import show_2d

show_2d = partial(show_2d, cmap="inferno")
```

Do not repeat the import later and reset this local default. Explain display
normalization and shared versus independent contrast. Add scale bars when
calibration is known. Retain the short Gold workflows in the README and link
to them from detailed guides. Keep historical measurements dated and separate
from current API instructions.

## Docstrings

Use NumPy-style docstrings for public Python APIs. The first line is a concise
summary. The following paragraph explains why the operation exists and any
scientific choice the caller must understand.

Every public parameter and return value states:

- what the quantity represents;
- its units, or that it is dimensionless;
- its coordinate order and array shape when relevant; and
- its default or provenance source when relevant.

Use `name : type` in parameter sections, mark parameters with defaults as
`optional`, and include at least one `>>>` example showing the most common
scientist-facing call. Do not repeat type hints in prose or document private
implementation details.

## Coordinates and shapes

All public image and scan coordinates use `(row, col)` order with the explicit
mathematical equivalence `(row, column) ≡ (r, c)`. Row/$r$ is the slow,
vertical axis and column/$c$ is the fast, horizontal axis. Write shapes in the
same order:

```text
(scan_rows, scan_cols, detector_rows, detector_cols)
```

Use `row` and `col` in public names, metadata, and error messages. In equations,
use $\mathbf R=(R_r,R_c)$ for real-space probe/scan coordinates and
$\mathbf k=(k_r,k_c)$ for detector scattering coordinates.
Some plotting libraries request the horizontal coordinate before the vertical
coordinate. Document that adapter boundary without changing the scientific
row/column array order.

## Quantities and units

Put a space between a numerical value and its unit: `200 kV`, `5 nm`, `12 ms`,
and `6 GiB`. Unit symbols are not pluralized and do not end with a period.

Use a unit-bearing public name when an unlabeled scalar would be ambiguous or
could silently change scientific meaning, for example `rotation_angle_deg`,
`scan_pixel_size_nm`, or `timeout_s`. Otherwise, state the unit in the
docstring, result metadata, and provenance. Distinguish:

- decimal storage and transfer quantities (`MB`, `GB`, `GB/s`);
- binary memory quantities (`MiB`, `GiB`);
- elapsed time (`ms`, `s`);
- detector or scan indices (`px`) from calibrated distances (`nm`) or angles
  (`mrad`); and
- count-valued quantities (`counts`) from dimensionless ratios, masks, and
  normalized coordinates, and from calibrated physical quantities.

Do not attach a physical unit to a result until the required calibration has
been applied. Record both the numerical value and unit in metadata; do not make
the unit inferable only from prose or a plot label.

## Mathematical notation

Introduce an equation by stating the scientific quantity it computes. Define
every new symbol immediately after the equation, including its domain, shape,
unit, normalization, coordinate order, and calibration source.

Use consistent roles:

- italic lowercase letters for scalars;
- bold uppercase $\mathbf R$ for real-space probe position, bold lowercase
  $\mathbf k$ for detector reciprocal coordinates, and
  $\boldsymbol{\nu}$ for scan spatial-frequency coordinates;
- uppercase letters for arrays, transforms, or operators when appropriate;
- roman text for named operators, such as $\operatorname{argmin}$; and
- semantic subscripts, such as $k_{\min}$, instead of unexplained indices.

Current explanatory prose and equations use $\mathbf k$ for detector
coordinates and $\boldsymbol{\nu}$ for scan frequency. Historical evidence may
retain literal implementation identifiers such as `G_qk` when renaming them
would falsify a benchmark artifact, source symbol, or archived command. Label
those as legacy identifiers rather than treating them as current notation.

In MyST Markdown, write inline math as `$k = 2\pi/\lambda$` and display math in
`$$` blocks. In Python docstrings, use reStructuredText ``:math:`` for inline
math and ``.. math::`` for display equations. Do not use code formatting as a
substitute for mathematical notation.

Equations preserve the repository's scientific contract: they state any crop,
bin, mask, dtype conversion, approximation, normalization, or calibration that
changes the result. Code identifiers may follow an equation, but they do not
replace the mathematical definition.

Every scientific-kernel page also includes an **Optimization model** section.
Describe reusable dataflow choices—residency, fusion, traversal count, buffer
reuse, queue overlap, synchronization, and readback—without making an
unmeasured speed claim or hard-coding one backend topology into the science.

## Scientific prose and evidence

State facts, assumptions, limits, and measured values. Avoid evaluative terms
such as “fast,” “large,” or “accurate” without a number and protocol. A
performance claim includes the device, source state, shape, dtype, crop, bin,
cache condition, repetitions, statistic, and memory measurement. A parity
claim includes the reference, metric, tolerance, and result.

### Benchmark device labels

Identify a benchmark computer by reproducible hardware, never by a local host
nickname. The **Computer** field uses the product class, chip or accelerator,
and installed memory, such as `MacBook Pro (M5 Max, 128 GB)` or
`Linux CUDA workstation (dual 96 GB Blackwell GPUs)`. Put the exact model
identifier, GPU variant and index, driver, browser adapter, and runtime version
in **Device tested**. This keeps tables comparable when machines are renamed
and makes every row understandable outside the development lab.

Use [benchmark methodology](../performance/methodology.md) and
[cross-backend parity](../performance/parity.md) for the required evidence.

## Tables: one cell, one value

Use long-form tables for capabilities and benchmarks. One row represents one
exact configuration or one exact measurement; one cell contains one field.
Repeat the row when the platform, scan size, detector size, bin, dtype, cache
state, fixture, statistic, or timing boundary changes.

For example, detector bins 2, 4, and 8 at `512x512` are three rows, not
`2/4/8` in one cell with `1.199/1.212/1.106 s` in another. Split source,
decode/working, accumulation, and resident dtype into separate columns. Split
the timing statistic from the numerical time, and the memory kind from the
memory value.

Do not infer a Cartesian product from independent tests. Evidence that a
runtime supports a `512x512` scan and separate evidence that it implements bin
8 do not prove the joint `512x512`, bin-8 configuration. Add that exact row as
**Pending** until its joint parity and physical timing are retained. This
long-form form is deliberately repetitive: it is sortable, machine-checkable,
and safe to extend without rewriting table structure.

Use two levels of detail. A human-facing current-measurement table ends with
**Device tested**, **Date tested**, and **Revision**. Do not add opaque evidence
IDs to that overview; keep the source revision as one separate field and link
the authoritative benchmark ledger once in nearby prose. The detailed ledger
keeps the evidence ID, exact revision, command, distribution, memory record,
calibration, and parity artifact.
