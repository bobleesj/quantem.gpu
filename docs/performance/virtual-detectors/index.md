# Virtual detectors

BF, DF, ADF, arbitrary detector masks and diffraction reductions operate on the
same acquisition but can touch very different numbers of detector pixels.
Record the mask, original scan/detector shape, representation and reduction
dtype when comparing implementations.

## Evidence and implementation

- [Virtual-image validation record](validation.md): retained CUDA, Metal and WebGPU gates, timings and unresolved memory cases. Its implementation-status statements belong to the recorded work, not an automatic current-release guarantee.
- [Scientific operation and source map](../../kernels/virtual-detectors.md): detector coordinates, masks and optimized reduction ownership.
- [Detector API](../../api/images_dpc.md): public inputs and outputs.
- [Cross-backend parity](../parity/index.md): exact integer-product versus floating-point requirements.

[CoM and DPC](../dpc/index.md) have their own entry point because moment fields,
rotation and phase integration need additional numerical checks. Full source
loading is tracked under [IO](../io/index.md), not included implicitly in a
resident detector-update time.
