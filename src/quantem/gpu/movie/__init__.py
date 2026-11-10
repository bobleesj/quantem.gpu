"""Simple GIF and MP4 export helpers for STEM result movies.

The public API is intentionally small:

``save_gif(data, path, ...)`` and ``save_mp4(data, path, ...)``.

``data`` may be one stack with shape ``(frame, row, col)``, several stacks as
``(movie, frame, row, col)``, or a list of stacks. Panels keep their native
pixels unless ``max_width`` is given, which prints one line. ``device="auto"``
uses CUDA MP4 when available, then Apple Metal/MPS when available; otherwise it
prints one line saying why and encodes with ffmpeg, whose frames differ from the
GPU writers'.
"""

from quantem.gpu.movie.export import MovieData, save_gif, save_movie, save_mp4

__all__ = ["MovieData", "save_gif", "save_movie", "save_mp4"]
