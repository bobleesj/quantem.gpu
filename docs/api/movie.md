# Movie API

The movie namespace converts scientific image stacks into GIF or MP4 artifacts
without owning the upstream scientific computation. Install the `movie` extra
using the current command on the [installation page](../install.md).

## Inputs and outputs

```python
from quantem.gpu import movie

movie.save_gif(data, "movie.gif")
movie.save_mp4(data, "movie.mp4", backend="auto")
movie.save_movie(data, "movie.mp4")
```

`save_movie()` dispatches from the path suffix: `.gif` uses `save_gif()` and
`.mp4` uses `save_mp4()`. Each function returns the written path according to
its public signature.

## Shapes, coordinates, dtypes, and units

One stack has shape `(frame, row, column)`. Several aligned stacks use
`(movie, frame, row, column)` or a sequence of 3D arrays. All stacks must have
the same frame count and spatial shape. Input scientific dtype is preserved up
to the declared display normalization; encoded frames are presentation data,
not a replacement for the source array. Frame rate is in frames per second.

## Errors and unsupported requests

- Empty data, unsupported rank, mismatched stack shapes, or mismatched frame
  counts raise with the conflicting shape.
- An unsupported suffix passed to `save_movie()` raises instead of guessing a
  container.
- An explicitly requested accelerator backend must be available. Automatic
  dispatch may choose another documented encoder, but the reported backend
  must match the path actually used.

## Provenance

For a scientific export, retain the source/result identity, source shape and
dtype, selected frames, spatial orientation, normalization/percentiles,
colormap, labels, frame rate, encoder backend, codec/container, output checksum,
and package revision. Encoded bitstreams can differ across valid encoders, so
parity compares decoded frames and metadata rather than requiring identical
container bytes.

## Backend dispatch

`save_mp4(..., backend="auto")` tries CUDA/NVENC when available, then Apple
Metal/MPS on macOS, then CPU frame rendering plus ffmpeg. Use
`backend="cuda"` or `backend="mps"` only when that exact path is part of the
required contract or benchmark.

## Integration boundary

QuantEM.GPU owns deterministic frame preparation, backend selection, encoding,
and export metadata. A consuming application owns which scientific result and
frames to export, user-visible labels, destination choice, and presentation.

See [Display and export kernels](../kernels/display-export.md) for the shared
orientation, color, histogram, FFT, and parity rules.
