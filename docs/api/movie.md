# Movie API

Primary imports:

```python
from quantem.gpu import movie

movie.save_gif(data, "movie.gif")
movie.save_mp4(data, "movie.mp4", backend="auto")
movie.save_movie(data, "movie.mp4")
```

`save_movie()` dispatches from the path suffix:

- `.gif` -> `save_gif`
- `.mp4` -> `save_mp4`

`save_mp4(..., backend="auto")` tries the CUDA/NVENC writer when it is
available, then falls back to CPU frame rendering plus ffmpeg. Use
`backend="cuda"` only when the CUDA path is required for a benchmark or release
claim.
