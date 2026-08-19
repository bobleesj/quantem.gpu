# Display and export kernels

Display kernels transform already-computed scientific arrays into ranges,
histograms, colors, FFT views, or encoded frames. They do not redefine the
underlying detector or reconstruction result.

For a scalar image $A[r,c]$, display statistics operate in the same public
order

$$
(\text{row},\text{column})\equiv(r,c).
$$

Colormap and movie output preserve that orientation unless an explicit
presentation transform is requested and recorded.

## Shared operations

- finite min/max and percentile range;
- histogram accumulation and contrast intervals;
- linear, logarithmic, and other declared transfer functions;
- scalar-to-RGBA colormap conversion;
- two-dimensional FFT magnitude/phase views; and
- GIF/MP4 frame composition and encoding.

```python
from quantem.gpu import movie

movie.save_gif(stack, "preview.gif", fps=8)
movie.save_mp4(stack, "preview.mp4", fps=24, backend="auto")
```

Stacks use `(frame, row, column) ≡ (frame, r, c)`. Multi-panel stacks add a
leading movie/panel axis.

## Optimization model

Keep scientific results accelerator-resident while computing statistics,
histograms, transfer functions, FFT views, and RGBA output. Reuse histogram and
surface buffers, avoid host round trips between display stages, and encode from
the resident representation when the platform supports it. Presentation
latency must be measured separately from the scientific load/reduction time.

## Source map and gates

| Layer | Source |
|---|---|
| Python display math | `src/quantem/gpu/display` |
| WebGPU display kernels | `src/quantem/gpu/display/webgpu` |
| Native Metal display | `MetalDisplayKernels` and `MetalImageRuntime` |
| Native FFT views | `MetalImageFFT` |
| Movie encoding | `src/quantem/gpu/movie` |

Parity covers nonfinite values, constant images, row/column orientation,
histogram totals, colormap bytes, FFT normalization, frame order, and encoded
metadata. Integer histograms and RGBA buffers are byte-exact where formats
match; encoded video bitstreams may require decoded-frame comparison.
