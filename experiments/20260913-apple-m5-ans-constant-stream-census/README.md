# Seven-source polar constant-stream census

Status: **refuted**. This CPU/source census asked whether exact zero-width
streams let the current ADF center-8→center-20 polar plan skip scan-axis field
work or hoist packet-constant values.

The exact radial1/leaf16 plan had 371 selected fields and 1,067 residual
detector pixels per source, across 512 packets and seven distinct full
`(512, 512, 192, 192)` uint16 acquisitions. The census inspected
**1,329,664 field/packet descriptors**. All had nonzero width; none had a
zero-width stream, whether the base was zero or nonzero. Therefore this
hypothesis can remove **0%** of the selected scan-axis field terms on this
measured transition. It is not a GPU timing result and makes no speed claim.

This rules out one proposed shortcut for the current transition. It does not
show that the polar index or ANS decoder cannot be optimized by a different,
exact algorithm. See `results/census.json` for the machine-readable result.
