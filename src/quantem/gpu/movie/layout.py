"""Grid layout and panel labels shared by the movie writers.

The CUDA and Metal writers render every movie of a grid into one NV12 frame
and stamp each panel's label from masks drawn once on the CPU, so the grid
geometry, the label placement and the label masks are computed here for both.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Searched in order: the macOS system fonts, then DejaVu as installed on Linux.
FONT_FILES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
)


@dataclass(frozen=True)
class GridLayout:
    """Pixel geometry of one grid frame of movies.

    ``width`` and ``height`` are the frame size, rounded up to even numbers
    because H.264 encodes 2 x 2 chroma blocks; the other sizes are per panel,
    after scaling the grid down to ``max_width``.
    """

    columns: int
    width: int
    height: int
    frame_width: int
    frame_height: int
    gap: int
    label_height: int


@dataclass(frozen=True)
class PanelLabel:
    """One panel's label text, the (x, y) pixel of its top-left corner, and its point size."""

    text: str
    x: int
    y: int
    font_size: int


def grid_layout(
    panels: int,
    height: int,
    width: int,
    *,
    cols: int | None,
    gap: int,
    label_height: int,
    max_width: int | None,
) -> GridLayout:
    """Lay ``panels`` movies of ``height`` x ``width`` pixels in rows of ``cols`` (default up to 3).

    Each panel has a label row of ``label_height`` above it and ``gap`` pixels
    between panels; the whole grid scales down to ``max_width`` when wider.
    """
    columns = min(panels, 3) if cols is None else max(1, int(cols))
    rows = math.ceil(panels / columns)
    gap = max(0, int(gap))
    label_height = max(0, int(label_height))
    total_width = columns * width + (columns - 1) * gap
    scale = 1.0
    if max_width is not None and total_width > int(max_width):
        scale = int(max_width) / total_width
    frame_width = max(1, round(width * scale))
    frame_height = max(1, round(height * scale))
    gap_scaled = max(0, round(gap * scale))
    label_height_scaled = max(0, round(label_height * scale))
    out_width = columns * frame_width + (columns - 1) * gap_scaled
    out_height = rows * (frame_height + label_height_scaled) + (rows - 1) * gap_scaled
    return GridLayout(
        columns,
        out_width + out_width % 2,
        out_height + out_height % 2,
        frame_width,
        frame_height,
        gap_scaled,
        label_height_scaled,
    )


def panel_labels(
    layout: GridLayout,
    labels: Sequence[str] | None,
    panels: int,
    frames: int,
) -> list[list[PanelLabel]]:
    """Return every frame's panel labels, ``"<name> [frame/frames]"``, none without a label row.

    Names default to ``"Movie 1"``, ``"Movie 2"``, ... Each label sits 4 pixels
    right of and 2 pixels below its panel's top-left corner, at a point size
    that fits its lines into the label row.
    """
    names = (
        [str(item) for item in labels]
        if labels is not None
        else [f"Movie {index + 1}" for index in range(panels)]
    )
    if layout.label_height <= 0:
        return [[] for _ in range(frames)]
    result = []
    for frame_index in range(frames):
        frame_labels = []
        for panel_index in range(panels):
            row, col = divmod(panel_index, layout.columns)
            text = f"{names[panel_index]} [{frame_index + 1}/{frames}]"
            frame_labels.append(
                PanelLabel(
                    text,
                    col * (layout.frame_width + layout.gap) + 4,
                    row * (layout.frame_height + layout.label_height + layout.gap) + 2,
                    label_font_size(layout.label_height, text.count("\n") + 1),
                )
            )
        result.append(frame_labels)
    return result


def label_font_size(label_height: int, lines: int) -> int:
    """Point size, 8 to 24, that fits ``lines`` lines of a label into a row of ``label_height`` pixels.

    Every movie writer (CPU, CUDA, Metal) draws its labels at this size.
    """
    return min(24, max(8, label_height // lines - 4))


def label_font(size: int) -> ImageFont.ImageFont:
    """Return the first available bold sans-serif font at ``size`` points, or PIL's default."""
    for name in FONT_FILES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_mask(text: str, font: ImageFont.ImageFont) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one label as uint8 ``(white, black)`` masks.

    The white mask is the text; the black mask is the text shifted one pixel
    diagonally each way, an outline that keeps the label readable on any
    image. Both have 4 pixels of padding around the text.
    """
    probe = Image.new("L", (1, 1), 0)
    left, top, right, bottom = ImageDraw.Draw(probe).multiline_textbbox(
        (0, 0), text, font=font, spacing=2
    )
    pad = 4
    mask_width = max(1, right - left + 2 * pad)
    mask_height = max(1, bottom - top + 2 * pad)
    white = Image.new("L", (mask_width, mask_height), 0)
    black = Image.new("L", (mask_width, mask_height), 0)
    white_draw = ImageDraw.Draw(white)
    black_draw = ImageDraw.Draw(black)
    text_x = pad - left
    text_y = pad - top
    for dx in (-1, 1):
        for dy in (-1, 1):
            black_draw.multiline_text(
                (text_x + dx, text_y + dy), text, font=font, fill=255, spacing=2
            )
    white_draw.multiline_text((text_x, text_y), text, font=font, fill=255, spacing=2)
    return np.asarray(white, dtype=np.uint8), np.asarray(black, dtype=np.uint8)
