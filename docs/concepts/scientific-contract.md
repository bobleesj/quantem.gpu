# Scientific contract

Accelerator parity begins with meaning, not speed. Every backend must preserve
the same source geometry, coordinate convention, arithmetic, and provenance
before its performance can be compared.

## Coordinates and regions

Spatial coordinates use `(row, column)` throughout Python, Swift, Metal, CUDA,
and WebGPU. Scan and detector regions are half-open:

```text
(row_start, row_stop, column_start, column_stop)
```

Regions are scientific choices. They are never introduced automatically to
make a dataset fit or a benchmark pass.

## Binning

Scan and detector bins are explicit positive integers. Integer detector binning
sums counts and widens the accumulation dtype before overflow. Partial edge
bins are retained using ceiling output dimensions.

Automatic resource policy may recommend or select detector binning only when
the consuming application makes that choice visible and records:

- the requested and selected bin;
- source and output detector shape;
- source, accumulation, and output dtype; and
- the memory reason for the choice.

A binned result must never be labeled or cached as native detector resolution.

## Precision and masking

Native detector counts remain integer evidence. A bad-pixel mask is applied in
the same order on every backend and is part of source identity. Converting to
`uint8` is lossless only when an exact value-range audit proves every corrected
count fits. Reconstruction workflows retain the precision required by their
objective.

## Provenance

Every reusable result or parity bundle records at least:

- source identity and source revision;
- source scan/detector shape and dtype;
- scan and detector region;
- scan and detector bin;
- bad-pixel policy and mask identity;
- output shape, output dtype, and accumulation dtype;
- backend, device, and kernel revision; and
- whether the evidence is native, cropped, binned, cached, or reconstructed.

## Failure behavior

An unsupported accelerated path fails with a corrective error. Production APIs
do not silently move scientific work to CPU, crop a scan, bin a detector,
reduce a mask, or change precision. The CPU implementation is available only
when explicitly selected as a reference.

See [Cross-backend parity](../performance/parity.md) for numerical gates and
[Repository architecture](../maintainer/backend-layout-and-parity.md) for the
implementation boundary.
