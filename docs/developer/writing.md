# Scientific writing, notation, and units

QuantEM.GPU follows the coding and documentation conventions in
[`ophusgroup/dev` Appendix D](https://github.com/ophusgroup/dev#appendix-d-coding-standards).
Write for scientific API authors, kernel/runtime implementers, and application
integrators as well as scientists calling the API. State the scientific
operation first, define its stable contract, map it to real source paths, and
explain how to verify an implementation.

## Audiences and page types

Use the page type that matches the developer's question:

| Reader question | Page type | Required content |
|---|---|---|
| What does this operation compute? | Scientific kernel | equations, coordinates, shapes/dtypes/units, optimization model, source map, parity gates |
| How do I implement it on my device? | Backend implementation | runtime boundary, sources, memory/execution model, build, profiling, acceptance |
| What may my code call and rely on? | API contract | typed inputs/outputs/errors, provenance, minimal example, ownership |
| Is this result correct and faster? | Evidence | revision, device, source and plan, cache state, memory, statistic, parity artifact |

Primary documentation must not depend on a consumer UI framework. Application
developers need integration ownership—what this package computes and what the
client schedules or presents—not instructions for a particular screen or
viewer. Product-specific details belong in that product's documentation unless
they are necessary historical provenance.

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
mathematical equivalence `(row, column) ≡ (y, x)`. Row/$y$ is the slow,
vertical axis and column/$x$ is the fast, horizontal axis. Write shapes in the
same order:

```text
(scan_rows, scan_cols, detector_rows, detector_cols)
```

Use `row` and `col` in public names, metadata, and error messages. In equations,
use $r_y,r_x$ for scan coordinates and $q_y,q_x$ for detector coordinates.
Plotting libraries may request `(x, y) = (column, row)`; document that adapter
boundary without changing the scientific array order.

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
- bold lowercase letters for vectors, such as $\mathbf r$ and $\mathbf q$;
- uppercase letters for arrays, transforms, or operators when appropriate;
- roman text for named operators, such as $\operatorname{argmin}$; and
- semantic subscripts, such as $q_{\min}$, instead of unexplained indices.

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

Use [benchmark methodology](../performance/methodology.md) and
[cross-backend parity](../performance/parity.md) for the required evidence.
