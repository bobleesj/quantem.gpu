# IO Metal kernels for Python MPS

These resources implement the Python MPS count-IO paths. The owning Python
module loads and dispatches each file; applications use `quantem.gpu.io.load`
instead of compiling these resources independently.

| Resource | Owner | Purpose |
|---|---|---|
| [bslz4.msl](bslz4.msl) | [dense.py](../dense.py) | Bitshuffle/LZ4 decode and explicit detector binning |
| [compact_v3.msl](compact_v3.msl) | [packed.py](../packed.py) | Prepared direct-bitpacked loading and resident count operations |
| [ans_counts.msl](ans_counts.msl) | [_ans.py](../_ans.py) | Exact ANS validation, requested DP, mask sums, and ANS-to-packed conversion |

Source counts retain their declared integer dtype, independently of storage
words. ANS and packed profiles are not interchangeable decoder layouts. The
new ANS path has small physical integer parity; full-volume timing, peak
memory, reverse conversion, and additional products remain qualification gates.

Keep shader code in `.msl` resources for compiler diagnostics and source review.
Native Swift owns separate packaged resources under `swift/Sources/`; it does
not import this Python dispatcher. See the
[representation contract](../../../../../../../docs/api/representations.md)
and [MPS implementation guide](../../../../../../../docs/platforms/mps.md).
