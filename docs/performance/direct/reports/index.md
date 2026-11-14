# Direct ptychography reports

- [Exact broad-search density batching](2026-11-01-broad-density-batching.md)

Dated records retain their original workload, backend, source revision and
acceptance limits. They are evidence, not a statement of current release speed.
A record that describes since-retired behaviour (the phase-variance search,
Nelder–Mead or finite-difference refinement, the `mean_phase` and
`complex_wave` estimators) says so in a **Historical** line at its top.

- [The whole-scan fifth-order fit on a 4 GB CUDA device](2026-11-13-whole-scan-fit-small-device.md)
- [Direct ptychography through fifth order on a 4 GB CUDA device: 512 x 512 x 192 x 192](2026-11-12-small-device-direct-ptycho.md)
- [Native Direct Ptycho preparation and exact QEM column decoding](2026-11-05-direct-ptycho-preparation.md)
- [Full-scan native Metal fitting on Apple M5](2026-11-05-mps-full-scan.md)
- [Exact-derivative direct ptychography fit in Python on CUDA and MPS](2026-10-25-direct-exact-derivative-fit.md)
- [Native Metal direct ptychography redraw precision at the lowest scan frequencies](2026-10-25-direct-redraw-precision.md)
- [Where the 9 s of regularized preparation goes](2026-10-24-regularized-preparation.md)
- [Direct phase reconstruction and viewer handoff](2026-10-24-direct-live-handoff.md)
- [Which reconstruction to measure drift on](2026-10-24-drift-method-comparison.md)
- [Live registration notch beyond frame 0](2026-10-24-drift-live-notch.md)
- [Redraw at slider speed: q, -q symmetry for direct ptychography and a drag window for regularized methods](2026-10-24-redraw-symmetry-and-drag-window.md)
- [Native Swift/Metal direct ptychography redraw while dragging aberration controls](2026-10-24-native-metal-direct-redraw.md)
- [Lower CUDA direct ptychography staging memory without changing numerical results](2026-10-22-direct-staging-memory.md)
- [Initial fitting before fifth-order refinement](2026-10-22-direct-initialization-and-refinement.md)
- [Reuse Fourier preparation in four-case direct ptychography review](2026-10-22-rotation-comparison-reuse.md)
- [Faster broad direct ptychography search with identical results](2026-10-22-independent-tpe-reuse.md)
- [CUDA broad-fifth research workflow](2026-10-22-broad-fifth-cuda-study.md)
- [CUDA direct ptychography region latency — 2026-10-21](2026-10-21-direct-region-latency.md)
- [Sign of the odd aberrations in direct ptychography reconstruction (2026-10-21)](2026-10-21-direct-odd-aberration-sign.md)
- [direct ptychography joint-fit benchmark, 2026-10-21](2026-10-21-direct-joint-benchmark.md)
- [Coma, trefoil and spherical aberration in the model-based direct ptychography fit (2026-10-21)](2026-10-21-direct-higher-order-fit.md)
- [The MPS standard direct ptychography fit uses the agreement objective (2026-10-20)](2026-10-20-direct-standard-fit-objective.md)
- [Tilt-corrected direct ptychography on CUDA: fit and preview speed (2026-10-20)](2026-10-20-direct-cuda-tilt-speed.md)
- [CUDA standard fit on the agreement objective, and one phase-variance search (2026-10-20)](2026-10-20-direct-cuda-standard-fit.md)
- [Tilt-corrected direct ptychography on Metal: preview and fit speed (2026-10-20)](2026-10-20-mps-tilt-speed.md)
- [Separable C10/C12 coherent direct ptychography on Metal, 2026-10-20](2026-10-20-mps-direct-separable-coherent.md)
- [MPS performance: what got faster, why the rest is slow, how to profile (2026-10-20)](2026-10-20-mps-performance.md)
- [MPS memory: where direct ptychography's gigabytes go and what fits which Mac (2026-10-20)](2026-10-20-mps-memory.md)
- [Reuse source addresses in MPS direct ptychography reconstruction](2026-10-18-mps-direct-source-addressing.md)
- [Exact group support in CUDA direct ptychography](2026-10-18-cuda-direct-group-support.md)
- [MPS coherent direct ptychography dispatch](2026-10-18-mps-direct-coherent-dispatch.md)
- [MPS direct ptychography preview reuse](mps-direct-preview-reuse-2026-10-17.md)
- [MPS fine-grid optical calculations, 2026-10-17](mps-direct-optical-shortcuts-2026-10-17.md)
- [Full BF MPS redraw comparison, 2026-10-17](mps-direct-full-bf-redraw-2026-10-17.md)
- [MPS coherent direct ptychography redraw — 2026-10-17](mps-coherent-redraw-2026-10-17.md)
- [Exact aperture bounds for interactive MPS direct ptychography](2026-10-17-direct-aperture-bounds.md)
- [Reuse MPS loading buffers and decode detector regions directly](2026-10-17-mps-direct-loading.md)
- [MPS direct ptychography correction and reduction fusion — 17 October 2026](2026-10-17-mps-direct-group-fusion.md)
- [MPS direct ptychography execution and coherent reconstruction, 2026-10-17](2026-10-17-mps-direct-dispatch.md)
- [MPS direct ptychography coherent Fourier support — 17 October 2026](2026-10-17-mps-direct-coherent-support.md)
- [MPS slider correction and interruptible diagnostics](2026-10-17-mps-interruptible-preview.md)
- [Coherent direct ptychography reduction on Metal](2026-10-17-mps-coherent-reduction.md)
- [Coherent direct ptychography reconstruction and scan-mean aliases](2026-10-16-direct-coherent-reconstruction.md)
- [MPS direct ptychography sampling and search, 2026-10-15](2026-10-15-mps-direct-upsampling.md)
- [CUDA direct ptychography refinement batching](2026-10-15-cuda-batched-refinement.md)
- [Later direct ptychography performance checkpoints](2026-09-11-direct-performance-checkpoints.md)

## CUDA fine-grid kernel experiments, 23-24 October 2026

These records measured in-kernel q, -q pairing, compact support and
closed-sideband skips on a separate branch. Main took a different design
(half-plane spectrum with the -conj mirror, bounded BF groups and drag
windows), so the kernels they describe are not in the current source. The
numbers and rejected ideas remain as evidence.

- [CUDA direct ptychography: shared BF coordinates and inactive blocks](2026-10-24-direct-shared-coordinates.md)
- [CUDA direct ptychography compact support and boundary dispatch](2026-10-24-direct-compact-support.md)
- [CUDA direct ptychography closed-sideband correction and host transfer](2026-10-24-direct-closed-sideband.md)
- [Skip closed-aperture loads in fine-grid direct ptychography](2026-10-23-direct-zero-contributions.md)
- [Full-field direct ptychography redraws, 2026-10-23](2026-10-23-direct-redraw.md)
- [Native 512 direct ptychography instruction-reduction experiments](2026-10-23-direct-instruction-reduction.md)
- [Native 512 direct ptychography hardware-counter diagnosis](2026-10-23-direct-hardware-counters.md)
- [Native 512 direct ptychography frequency pairing](2026-10-23-direct-frequency-pairing.md)
- [CUDA direct ptychography fine-grid correction](2026-10-23-direct-fine-grid-pairing.md)
- [Classify fine-grid direct ptychography mirror reuse once per frequency](2026-10-23-direct-fine-grid-classification.md)
- [Native 512 direct ptychography: fixed-optics caching and launch layout](2026-10-23-direct-c10-cache.md)
- [Higher-order direct ptychography: group alias-edge work on CUDA](2026-10-23-direct-alias-layout.md)
