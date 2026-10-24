# Performance and parity

Choose the scientific operation first. Each folder contains a maintained entry
point and, where available, dated reports. Backends are compared within the
operation so the same scientific workload stays together.

## Browse by topic

| Operation | Scope |
|---|---|
| [IO and compression](io/index.md) | Loading, decoding, encoding, conversion, saving and memory |
| [SSB](ssb/index.md) | Calibration, joint fitting, reconstruction and local-region measurements |
| [Virtual detectors](virtual-detectors/index.md) | BF, DF, ADF, arbitrary masks and diffraction reductions |
| [CoM, DPC and iDPC](dpc/index.md) | Detector moments, rotation and phase integration |
| [Rendering and transfer](rendering/index.md) | FFT display, contrast, GPU/host transfer and actual image updates |
| [Cross-backend parity](parity/index.md) | Shared exactness contracts, floating-point gates and unresolved cases |

## How to read and reproduce results

Start with [methodology](methodology.md) for cache states, timing boundaries,
memory and parity requirements, then the [testing guide](../developer/testing.md)
for maintained commands. The [results index](results.md) retains individual
measurements and the [change ledger](changes.md) records chronological decisions.

A report date identifies a measurement, not the latest released implementation.
Keep its shape, dtype, mask, precision, backend, device, source revision and
cache state together. Kernel duration, completed reconstruction, transfer and
new scientific image paints are different quantities. Missing evidence remains
unqualified; older or rejected results are not silently replaced.

## Adding a benchmark

Put each new report under the owning operation's `reports/` folder, using
`YYYY-MM-DD-description.md`. Link it from that operation's report index and
register its interpretation in [results](results.md) and [changes](changes.md).
Use one canonical report for a cross-operation experiment and link to it from
other relevant topics. Create a report folder when there is an actual report.

Keep raw logs, source identities and non-public images in the private evidence
archive; publish anonymous workloads, measured numbers and limitations. General
API and kernel documentation remains in its existing guide, not copied here.

Previous published report URLs redirect to their new locations, including
section fragments. The [optimization ledger](../maintainer/backend-optimization-matrix.md)
and [historical archive](../maintainer/history/index.md) retain broader records.
