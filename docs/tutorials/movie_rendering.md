# Render GIF and MP4 movies

Use `quantem.gpu.movie` when movie creation needs the shared QuantEM rendering
path or CUDA/NVENC MP4 acceleration.

```python
from quantem.gpu import movie

movie.save_gif(stack, "preview.gif", fps=8)
movie.save_mp4(stack, "preview.mp4", fps=24, backend="auto")
```

Movie data can be:

- one stack with shape `(frame, row, col)`
- several stacks with shape `(movie, frame, row, col)`
- a list of stacks with matching frame count and spatial shape

For side-by-side comparisons:

```python
movie.save_mp4(
    [raw_stack, denoised_stack, residual_stack],
    "comparison.mp4",
    labels=["raw", "denoised", "residual"],
    cols=3,
    fps=12,
    backend="auto",
)
```

Backend choices:

| Backend | Meaning |
|---|---|
| `auto` | use CUDA/NVENC when available, otherwise use CPU rendering and ffmpeg |
| `cuda` | require CUDA/NVENC path and fail honestly if unavailable |
| `cpu` | render frames on CPU and write MP4 with ffmpeg |

Widget export buttons can call these helpers, but the user-facing export UI
belongs in `quantem.widget`.
