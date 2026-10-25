# SSB reports

Dated records retain their original workload, backend, source revision and
acceptance limits. They are evidence, not a statement of current release speed.

- [Redraw at slider speed: q, -q symmetry for SSB and a drag window for regularized methods](2026-10-24-redraw-symmetry-and-drag-window.md)
- [Lower CUDA SSB staging memory without changing numerical results](2026-10-22-ssb-staging-memory.md)
- [Initial fitting before fifth-order refinement](2026-10-22-ssb-initialization-and-refinement.md)
- [Reuse Fourier preparation in four-case SSB review](2026-10-22-rotation-comparison-reuse.md)
- [Faster broad SSB search with identical results](2026-10-22-independent-tpe-reuse.md)
- [CUDA broad-fifth research workflow](2026-10-22-broad-fifth-cuda-study.md)
- [CUDA SSB region latency — 2026-10-21](2026-10-21-ssb-region-latency.md)
- [Sign of the odd aberrations in SSB reconstruction (2026-10-21)](2026-10-21-ssb-odd-aberration-sign.md)
- [SSB joint-fit benchmark, 2026-10-21](2026-10-21-ssb-joint-benchmark.md)
- [Coma, trefoil and spherical aberration in the model-based SSB fit (2026-10-21)](2026-10-21-ssb-higher-order-fit.md)
- [The MPS standard SSB fit uses the agreement objective (2026-10-20)](2026-10-20-ssb-standard-fit-objective.md)
- [Tilt-corrected SSB on CUDA: fit and preview speed (2026-10-20)](2026-10-20-ssb-cuda-tilt-speed.md)
- [CUDA standard fit on the agreement objective, and one phase-variance search (2026-10-20)](2026-10-20-ssb-cuda-standard-fit.md)
- [Tilt-corrected SSB on Metal: preview and fit speed (2026-10-20)](2026-10-20-mps-tilt-speed.md)
- [Separable C10/C12 coherent SSB on Metal, 2026-10-20](2026-10-20-mps-ssb-separable-coherent.md)
- [MPS performance: what got faster, why the rest is slow, how to profile (2026-10-20)](2026-10-20-mps-performance.md)
- [MPS memory: where SSB's gigabytes go and what fits which Mac (2026-10-20)](2026-10-20-mps-memory.md)
- [Reuse source addresses in MPS SSB reconstruction](2026-10-18-mps-ssb-source-addressing.md)
- [Exact group support in CUDA SSB](2026-10-18-cuda-ssb-group-support.md)
- [MPS SSB preview reuse](mps-ssb-preview-reuse-2026-10-17.md)
- [MPS fine-grid optical calculations, 2026-10-17](mps-ssb-optical-shortcuts-2026-10-17.md)
- [Full BF MPS redraw comparison, 2026-10-17](mps-ssb-full-bf-redraw-2026-10-17.md)
- [MPS coherent SSB redraw — 2026-10-17](mps-coherent-redraw-2026-10-17.md)
- [Exact aperture bounds for interactive MPS SSB](2026-10-17-ssb-aperture-bounds.md)
- [Reuse MPS loading buffers and decode detector regions directly](2026-10-17-mps-ssb-loading.md)
- [MPS SSB correction and reduction fusion — 17 October 2026](2026-10-17-mps-ssb-group-fusion.md)
- [MPS SSB execution and coherent reconstruction, 2026-10-17](2026-10-17-mps-ssb-dispatch.md)
- [MPS SSB coherent Fourier support — 17 October 2026](2026-10-17-mps-ssb-coherent-support.md)
- [MPS slider correction and interruptible diagnostics](2026-10-17-mps-interruptible-preview.md)
- [Coherent SSB reduction on Metal](2026-10-17-mps-coherent-reduction.md)
- [Coherent SSB reconstruction and scan-mean aliases](2026-10-16-ssb-coherent-reconstruction.md)
- [MPS SSB sampling and search, 2026-10-15](2026-10-15-mps-ssb-upsampling.md)
- [CUDA SSB refinement batching](2026-10-15-cuda-batched-refinement.md)
- [Later SSB performance checkpoints](2026-09-11-ssb-performance-checkpoints.md)

## CUDA fine-grid kernel experiments, 23-24 October 2026

These records measured in-kernel q, -q pairing, compact support and
closed-sideband skips on a separate branch. Main took a different design
(half-plane spectrum with the -conj mirror, bounded BF groups and drag
windows), so the kernels they describe are not in the current source. The
numbers and rejected ideas remain as evidence.

- [CUDA SSB: shared BF coordinates and inactive blocks](2026-10-24-ssb-shared-coordinates.md)
- [CUDA SSB compact support and boundary dispatch](2026-10-24-ssb-compact-support.md)
- [CUDA SSB closed-sideband correction and host transfer](2026-10-24-ssb-closed-sideband.md)
- [Skip closed-aperture loads in fine-grid SSB](2026-10-23-ssb-zero-contributions.md)
- [Full-field SSB redraws, 2026-10-23](2026-10-23-ssb-redraw.md)
- [Native 512 SSB instruction-reduction experiments](2026-10-23-ssb-instruction-reduction.md)
- [Native 512 SSB hardware-counter diagnosis](2026-10-23-ssb-hardware-counters.md)
- [Native 512 SSB frequency pairing](2026-10-23-ssb-frequency-pairing.md)
- [CUDA SSB fine-grid correction](2026-10-23-ssb-fine-grid-pairing.md)
- [Classify fine-grid SSB mirror reuse once per frequency](2026-10-23-ssb-fine-grid-classification.md)
- [Native 512 SSB: fixed-optics caching and launch layout](2026-10-23-ssb-c10-cache.md)
- [Higher-order SSB: group alias-edge work on CUDA](2026-10-23-ssb-alias-layout.md)
