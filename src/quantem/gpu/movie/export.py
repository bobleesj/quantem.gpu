"""GIF and MP4 export of STEM result movies: frame layout, labels, contrast, and backend choice."""

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from quantem.gpu.movie import cuda, mps
from quantem.gpu.movie.layout import grid_layout, label_font, label_font_size

MovieData = np.ndarray | Sequence[np.ndarray]

_CUDA_NVENC_MIN_EDGE_PX = 256


def _as_stack_list(data: MovieData) -> list[np.ndarray]:
    """Normalize public movie data to a list of 3D stacks."""
    if isinstance(data, np.ndarray):
        array = np.asarray(data)
        if array.ndim == 3:
            return [array]
        if array.ndim == 4:
            return [array[index] for index in range(array.shape[0])]
        raise ValueError(
            "movie data must have shape (frame, row, col) or "
            f"(movie, frame, row, col), got {array.shape}"
        )

    stacks = [np.asarray(item) for item in data]
    if not stacks:
        raise ValueError("movie data must contain at least one stack")
    for index, stack in enumerate(stacks):
        if stack.ndim != 3:
            raise ValueError(
                "each movie stack must have shape (frame, row, col), "
                f"item {index} has shape {stack.shape}"
            )
    return stacks


def _validate_stacks(stacks: list[np.ndarray]) -> tuple[int, int, int]:
    """Return the common (frames, height, width) of the stacks, or raise naming the first mismatch.

    Every writer tiles one frame of each movie into the same grid frame, so a
    stack with another frame count or image size cannot be laid out.
    """
    frames, height, width = stacks[0].shape
    for index, stack in enumerate(stacks[1:], start=1):
        if stack.shape[0] != frames:
            raise ValueError(
                "all movie stacks must have the same number of frames; "
                f"stack 0 has {frames}, stack {index} has {stack.shape[0]}"
            )
        if stack.shape[1:] != (height, width):
            raise ValueError(
                "all movie stacks must have the same spatial dimensions; "
                f"stack 0 has {(height, width)}, stack {index} has {stack.shape[1:]}"
            )
    return frames, height, width


def _contrast_limits(
    stacks: list[np.ndarray],
    *,
    percentile: tuple[float, float],
    shared: bool,
    ref_stacks: Sequence[np.ndarray] | None,
) -> list[tuple[float, float]]:
    """Return each movie's display range ``(low, high)`` from the given percentiles.

    Percentiles rather than min and max keep a few hot or dead pixels from
    flattening the contrast. ``shared`` uses one range for every movie, taken
    from ``ref_stacks`` when given, so panels stay comparable; a flat image
    gets a range of one so the scaling never divides by zero.
    """
    low_percentile, high_percentile = percentile
    if shared:
        reference_stacks = (
            [np.asarray(item) for item in ref_stacks]
            if ref_stacks is not None
            else stacks
        )
        values = np.concatenate([np.asarray(stack).ravel() for stack in reference_stacks])
        low, high = np.percentile(values, [low_percentile, high_percentile])
        if high <= low:
            high = low + 1.0
        return [(float(low), float(high))] * len(stacks)

    limits = []
    for stack in stacks:
        low, high = np.percentile(np.asarray(stack).ravel(), [low_percentile, high_percentile])
        if high <= low:
            high = low + 1.0
        limits.append((float(low), float(high)))
    return limits


def _stacks_with_limits(
    data: MovieData,
    *,
    percentile: tuple[float, float],
    shared_contrast: bool,
    ref_stacks: Sequence[np.ndarray] | None,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Validate array movie data and set its display ranges once for a GPU writer."""
    stacks = _as_stack_list(data)
    _validate_stacks(stacks)
    return stacks, _contrast_limits(
        stacks, percentile=percentile, shared=shared_contrast, ref_stacks=ref_stacks
    )


def _to_uint8(stack: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map ``[low, high]`` linearly onto 0..255 and clip, for 8-bit GIF and MP4 frames.

    The scaling runs in float64 so integer counts above 2**24 are represented
    exactly before the final truncation to uint8.
    """
    scaled = np.clip(
        (stack.astype(np.float64) - low) / (high - low) * 255.0,
        0.0,
        255.0,
    )
    return scaled.astype(np.uint8)


def _movie_frames(
    data: MovieData,
    *,
    labels: Sequence[str] | None,
    gap: int,
    label_height: int,
    max_width: int | None,
    columns: int | None,
    shared_contrast: bool,
    ref_stacks: Sequence[np.ndarray] | None,
    percentile: tuple[float, float],
) -> list[Image.Image]:
    """Render the portable writer's RGB frames: the movies tiled in a labeled grid.

    Stacks are scaled to 8 bits with their display ranges, resized with Lanczos
    when the caller set a ``max_width`` the grid exceeds, and labeled
    ``"<name> [frame/frames]"`` above each panel.
    """
    stacks = _as_stack_list(data)
    n_frames, height, width = _validate_stacks(stacks)
    names = (
        [str(item) for item in labels]
        if labels is not None
        else [f"Movie {index + 1}" for index in range(len(stacks))]
    )
    limits = _contrast_limits(
        stacks,
        percentile=percentile,
        shared=shared_contrast,
        ref_stacks=ref_stacks,
    )
    uint8_stacks = [
        _to_uint8(stack, low, high)
        for stack, (low, high) in zip(stacks, limits)
    ]

    # The GPU writers' layout, so every backend places panels on the same grid: panel sizes and
    # gaps are rounded first and the canvas is their sum, padded to even sides for H.264.
    layout = grid_layout(
        len(stacks),
        height,
        width,
        columns=columns,
        gap=gap,
        label_height=label_height,
        max_width=max_width,
    )
    resized = (layout.frame_width, layout.frame_height) != (width, height)
    fonts = [
        label_font(label_font_size(layout.label_height, name.count("\n") + 1))
        for name in names
    ]

    frames: list[Image.Image] = []
    for frame_index in range(n_frames):
        canvas = Image.new("RGB", (layout.width, layout.height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for movie_index, stack_uint8 in enumerate(uint8_stacks):
            row, col = divmod(movie_index, layout.columns)
            tile = Image.fromarray(stack_uint8[frame_index], mode="L").convert("RGB")
            if resized:
                tile = tile.resize((layout.frame_width, layout.frame_height), Image.LANCZOS)
            left = col * (layout.frame_width + layout.gap)
            top = row * (layout.frame_height + layout.label_height + layout.gap)
            canvas.paste(tile, (left, top + layout.label_height))
            if movie_index < len(names) and layout.label_height > 0:
                draw.multiline_text(
                    (left + 4, top + 2),
                    f"{names[movie_index]} [{frame_index + 1}/{n_frames}]",
                    fill=(255, 255, 255),
                    font=fonts[movie_index],
                    spacing=2,
                )
        frames.append(canvas)
    return frames


def _announce_max_width(
    data: MovieData,
    name: str,
    *,
    gap: int,
    label_height: int,
    max_width: int | None,
    columns: int | None,
) -> None:
    """Print one line when ``max_width`` shrinks the panels, since the movie then no longer shows native pixels.

    Without it a wide grid would be resampled with no sign that each frame lost resolution.
    """
    if max_width is None:
        return
    stacks = _as_stack_list(data)
    _, height, width = _validate_stacks(stacks)
    layout = grid_layout(
        len(stacks), height, width, columns=columns, gap=gap, label_height=label_height, max_width=max_width
    )
    if (layout.frame_height, layout.frame_width) != (height, width):
        print(f"{name}: max_width={max_width} resizes each {height} x {width} panel to "
              f"{layout.frame_height} x {layout.frame_width} pixels; pass max_width=None for native resolution.")


def save_gif(
    data: MovieData,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    fps: float = 10,
    gap: int = 12,
    label_height: int = 28,
    max_width: int | None = None,
    columns: int | None = None,
    shared_contrast: bool = True,
    ref_stacks: Sequence[np.ndarray] | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> Path:
    """Save one or more grayscale movie stacks as an animated GIF.

    Panels keep their native pixels; ``max_width`` shrinks the grid to that many pixels and prints one line saying so.
    """
    _announce_max_width(data, "save_gif", gap=gap, label_height=label_height, max_width=max_width, columns=columns)
    frames = _movie_frames(
        data, labels=labels, gap=gap, label_height=label_height, max_width=max_width, columns=columns,
        shared_contrast=shared_contrast, ref_stacks=ref_stacks, percentile=percentile,
    )
    return _write_gif(frames, path, float(fps))


def save_mp4(
    data: MovieData,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    fps: float = 10,
    gap: int = 12,
    label_height: int = 28,
    max_width: int | None = None,
    columns: int | None = None,
    shared_contrast: bool = True,
    ref_stacks: Sequence[np.ndarray] | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
    crf: int = 18,
    device: int | str = "auto",
) -> Path:
    """Save one or more grayscale movie stacks as an H.264 MP4.

    Parameters
    ----------
    max_width : int or None
        Panels keep their native pixels by default. A width shrinks the grid to that many pixels and prints one line
        saying so.
    device : {"auto", "cuda", "cuda:N", "mps"} or int
        Where the movie is written. ``"auto"`` uses the NVIDIA CUDA/NVENC writer
        on CUDA device 0 when it is available and the frame is at least 256
        pixels on each side, then the Apple Metal writer. When neither writes
        the movie it prints one line saying why and encodes the frames with
        ffmpeg, whose frames differ from the GPU writers' (Lanczos instead of
        nearest-neighbour resizing, unoutlined labels). A GPU number ``N`` or
        ``"cuda:N"`` selects the NVENC writer on CUDA device N; ``"cuda"`` is
        device 0. A named GPU writer raises instead of using another.
    """
    backend, index = _movie_device(device)
    _announce_max_width(data, "save_mp4", gap=gap, label_height=label_height, max_width=max_width, columns=columns)
    # the grid and contrast options every writer takes
    grid = dict(labels=labels, gap=gap, label_height=label_height, max_width=max_width, columns=columns)
    contrast = dict(percentile=percentile, shared_contrast=shared_contrast, ref_stacks=ref_stacks)
    # why "auto" did not reach a GPU writer, announced before the ffmpeg writer runs
    fallback = "no CUDA/NVENC or Apple Metal movie writer is available"
    if backend in {"auto", "cuda"}:
        try_cuda = backend == "cuda"
        if backend == "auto":
            stacks = _as_stack_list(data)
            _, height, width = _validate_stacks(stacks)
            layout = grid_layout(
                len(stacks), height, width, columns=columns, gap=gap, label_height=label_height, max_width=max_width
            )
            # PyNvVideoCodec may import successfully while NVENC rejects
            # small H.264 surfaces at encoder initialization time.
            large_enough = min(layout.width, layout.height) >= _CUDA_NVENC_MIN_EDGE_PX
            try_cuda = cuda.is_available()
            if try_cuda and not large_enough:
                try_cuda = False
                fallback = f"NVENC needs frames of at least {_CUDA_NVENC_MIN_EDGE_PX} pixels per side"
        if try_cuda:
            stacks, limits = _stacks_with_limits(data, **contrast)
            try:
                return cuda.save_mp4(
                    stacks, path, fps=float(fps), limits=limits, qp=int(crf), device=index, **grid
                )
            except (RuntimeError, MemoryError) as error:
                # the writer reports NVENC, CUDA and ffmpeg failures as RuntimeError
                if backend == "cuda":
                    raise
                fallback = f"the CUDA/NVENC writer failed ({error})"
    if backend in {"auto", "mps"}:
        try_mps = backend == "mps" or (backend == "auto" and mps.is_available())
        if try_mps:
            stacks, limits = _stacks_with_limits(data, **contrast)
            try:
                return mps.save_mp4(stacks, path, fps=float(fps), limits=limits, crf=int(crf), **grid)
            except (RuntimeError, MemoryError) as error:
                if backend == "mps":
                    raise
                fallback = f"the Apple Metal writer failed ({error})"
    print(f"save_mp4: {fallback}; encoding with ffmpeg, which resizes with Lanczos and draws "
          "unoutlined labels. Pass device='cuda' or 'mps' to get an error instead.")
    return _write_mp4(_movie_frames(data, **grid, **contrast), path, float(fps), crf=int(crf))


def _movie_device(device: int | str) -> tuple[str, int]:
    """``save_mp4``'s ``device`` as the writer it names, ``"auto"``, ``"cuda"`` or ``"mps"``, and the CUDA device index
    of the NVENC writer."""
    if isinstance(device, int) and not isinstance(device, bool):
        # a GPU number is CUDA device N; 0 on a Mac is its one Apple GPU, as quantem.gpu.device reads it
        if device == 0 and not cuda.is_available() and mps.is_available():
            return "mps", 0
        return "cuda", device
    name = str(device).strip().lower()
    if name.startswith("cuda:") and name[5:].isdigit():
        return "cuda", int(name[5:])
    if name not in {"auto", "cuda", "mps"}:
        raise ValueError(f"unknown movie device {device!r}; use 'auto', a GPU number, 'cuda:N' or 'mps'")
    return name, 0


def save_movie(
    data: MovieData,
    path: str | Path,
    *,
    format: str | None = None,
    **kwargs,
) -> Path:
    """Save a GIF or MP4, using ``format`` or the output suffix."""
    suffix = (format or Path(path).suffix.lstrip(".")).lower()
    if suffix == "gif":
        return save_gif(data, path, **kwargs)
    if suffix == "mp4":
        return save_mp4(data, path, **kwargs)
    raise ValueError(
        "movie format must be 'gif' or 'mp4', or path must end with .gif or .mp4"
    )


def _even_rgb_array(frame) -> np.ndarray:
    """Return an RGB uint8 array padded to even dimensions for H.264."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    pad_rows = height % 2
    pad_columns = width % 2
    if pad_rows or pad_columns:
        padded = np.zeros((height + pad_rows, width + pad_columns, 3), dtype=np.uint8)
        padded[:height, :width] = rgb
        if pad_rows:
            padded[height:, :width] = rgb[height - 1:height]
        if pad_columns:
            padded[:height, width:] = rgb[:, width - 1:width]
        if pad_rows and pad_columns:
            padded[height:, width:] = rgb[height - 1, width - 1]
        rgb = padded
    return rgb


def _write_gif(frames: list[Image.Image], path: str | Path, fps: float) -> Path:
    """Assemble RGB PIL frames into a looping GIF at the given fps."""
    if not frames:
        raise ValueError("write_gif requires at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, round(1000.0 / max(0.1, fps)))
    frames[0].save(
        str(path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


def _write_mp4(
    frames: list[Image.Image],
    path: str | Path,
    fps: float,
    *,
    crf: int = 18,
) -> Path:
    """Assemble RGB PIL frames into an H.264 MP4 using ffmpeg."""
    if not frames:
        raise ValueError("write_mp4 requires at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
    except ImportError:
        ffmpeg = "ffmpeg"
    else:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    arrays = [_even_rgb_array(frame) for frame in frames]
    height, width = arrays[0].shape[:2]
    for index, rgb in enumerate(arrays[1:], start=1):
        if rgb.shape[:2] != (height, width):
            raise ValueError(
                "all MP4 frames must have the same size; "
                f"frame 0 is {(height, width)}, frame {index} is {rgb.shape[:2]}"
            )
    with tempfile.TemporaryDirectory(prefix="quantem-gpu-mp4-") as frame_folder_name:
        frame_folder = Path(frame_folder_name)
        for index, rgb in enumerate(arrays):
            Image.fromarray(rgb, mode="RGB").save(frame_folder / f"frame_{index:06d}.png")
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{max(0.1, float(fps))}",
            "-i",
            str(frame_folder / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(crf)),
            str(path),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "save_mp4 requires ffmpeg on PATH. Install ffmpeg or use save_gif instead."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed while writing MP4: {exc}") from exc
    return path

