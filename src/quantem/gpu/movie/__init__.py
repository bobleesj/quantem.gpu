"""Simple GIF and MP4 export helpers for STEM result movies.

The public API is intentionally small:

``save_gif(data, path, ...)`` and ``save_mp4(data, path, ...)``.

``data`` may be one stack with shape ``(frame, row, col)``, several stacks as
``(movie, frame, row, col)``, a list of stacks, or a list of pre-rendered PIL
frames from a widget method. ``backend="auto"`` uses CUDA MP4 when available,
then Apple Metal/MPS when available, and otherwise falls back to the portable
CPU writer.
"""

from quantem.gpu.movie.export import MovieData, save_gif, save_movie, save_mp4

__all__ = ["MovieData", "save_gif", "save_movie", "save_mp4"]
