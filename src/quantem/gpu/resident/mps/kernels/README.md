# Metal kernels for the MPS residents

The Python module in each row reads and compiles its file, except
`count_tables.msl`, which only native Swift compiles; applications use
`quantem.gpu.io.load` and `quantem.gpu.detector` instead of compiling these
sources themselves.

| Resource | Owner | Purpose |
|---|---|---|
| [streamed_counts.msl](streamed_counts.msl) | [counts.py](../counts.py), [precision.py](../precision.py), [float_ans.py](../float_ans.py) | Count-ANS encode and decode |
| [count_tables.msl](count_tables.msl) | Swift `MetalEncodedSource` | Count-ANS frequency tables built on the GPU; Python CUDA and MPS upload the same tables from `formats/qem/reference.py` |
| [runtime_spatial.msl](runtime_spatial.msl) | [counts.py](../counts.py) | Camera spatial kernels appended to the count-ANS runtime |
| [float_ans.msl](float_ans.msl) | [float_ans.py](../float_ans.py) | Float bit-lane decode |
| [hot_pixels.msl](hot_pixels.msl) | [hot_pixels.py](../hot_pixels.py) | Hot-pixel correction of bounded count batches |
| [precision.msl](precision.msl) | [precision.py](../precision.py) | Scaled-uint16 conversion and direct queries |
| [reductions.msl](reductions.msl) | [virtual_image.py](../virtual_image.py) | Dense masked and detector sums, binning, mean diffraction and CoM over chunked frames |

Source counts retain their declared integer dtype, independently of storage
words. Exact integer and floating-output entry points have separate contracts;
they are not interchangeable precision modes.

Keep shader code in `.msl` resources for compiler diagnostics and source review.
Native Swift compiles some of these sources too: `Package.swift` copies
`runtime_spatial.msl` into the `Metal4DSTEMKernels` bundle and
`streamed_counts.msl`, `count_tables.msl`, `hot_pixels.msl` and `precision.msl`
into `MetalCountResources`, so each source exists once. Swift-only kernels live
under `native/swift/Sources/`. See the
[representation contract](../../../../../../docs/api/representations.md)
and [MPS implementation guide](../../../../../../docs/platforms/mps.md).
