# Choose a scientific operation

Each page starts from the science, then maps it to optimized kernels. The same
page is the shared specification for CUDA, Python MPS, native Swift/Metal,
WebGPU, and the CPU reference.

| Operation | Read this page when you need to | Main outputs |
|---|---|---|
| [Data model and coordinates](data-model.md) | interpret axes, shapes, units, regions, binning, and provenance | one cross-language contract |
| [Load, decode, and bin](load-decode-bin.md) | move compressed detector data into an accelerator-resident representation | resident 4D-STEM data and load provenance |
| [BF, DF, and ADF](virtual-detectors.md) | reduce detector counts with circular, annular, or arbitrary masks | scan-shaped scalar images |
| [CoM, DPC, and iDPC](com-dpc-idpc.md) | compute detector first moments, align the vector field, and integrate phase | two-component vector fields and phase |
| [Single-sideband ptychography](ssb.md) | reconstruct complex object information from overlapping diffraction information | object amplitude, phase, loss, aberrations |
| [Explicit scan regions](scan-regions.md) | intentionally analyze a half-open real-space subset | labeled scan subset |
| [Display and export kernels](display-export.md) | compute ranges, histograms, colors, FFT views, or encoded frames | presentation-ready derived buffers |

## Coordinate rule used everywhere

Array order is always

$$
(\text{row},\text{column}) \equiv (r,c).
$$

For 4D-STEM data,

$$
I[s_r,s_c,q_r,q_c],
$$

where $\mathbf s=(s_r,s_c)$ is the scan coordinate and
$\mathbf q=(q_r,q_c)$ is the detector coordinate. A backend-specific launch
order may differ internally, but its public inputs, outputs, masks, metadata,
and parity artifacts must preserve this order.

## Optimization rule used everywhere

Optimize transfers and traversals, not the scientific definition. Keep large
arrays accelerator-resident; fuse compatible operations; reuse buffers; avoid
per-batch synchronization; and return only the small result that the caller
requested. Any crop, bin, mask, precision change, or cached result remains an
explicit, provenance-recorded choice.
