# IO and compression

Loading, decoding, encoding, saving and reopening acquisitions belong here.
A fast decoder is not a complete load measurement: retain storage/cache state,
original counts, transfer, first usable products and peak memory separately.

## Start here

- [HDF5 and QEM loading campaign](reports/2026-10-23-io-loading-limits.md): full-data CUDA, Python MPS and native Swift/Metal evidence, with unresolved cases retained.
- [CUDA count codecs](codecs.md): compression speed and memory accounting.
- [IO reports](reports/index.md): dated measurements and rejected approaches.
- [Reproduce loading checks](../../developer/testing.md): maintained runners and required private inputs.

## Contracts and related guides

Read [load/decode/bin](../../kernels/load-decode-bin.md) for the operation,
[IO API](../../api/io.md) for public calls, and [methodology](../methodology.md)
for cold, cached and prepared timing definitions. Conversion verification
belongs to IO; shared cross-backend rules live in [parity](../parity/index.md).
