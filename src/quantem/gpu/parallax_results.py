"""Result containers for parallax reconstruction."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BFImage:
    """Bright-field image extracted from one diffraction-plane position.

    Parameters
    ----------
    data
        Image data with shape ``(scan_row, scan_col)``.
    k_col
        Column coordinate relative to the bright-field disk center.
    k_row
        Row coordinate relative to the bright-field disk center.
    """

    data: object
    k_col: int
    k_row: int


@dataclass
class ParallaxResult:
    """Result from parallax reconstruction."""

    image: object
    density: object
    shifts: list[tuple[float, float]]
    bf_images: list[BFImage] | None = None
    aberrations: dict | None = None
    elapsed: float | None = None

    def __repr__(self) -> str:
        lines = ["ParallaxResult:"]
        lines.append(f"  Image shape:  {tuple(self.image.shape)}")
        if self.shifts:
            lines.append(f"  BF positions: {len(self.shifts)}")
        if self.elapsed is not None:
            lines.append(f"  Elapsed:      {self.elapsed:.2f}s")
        if self.aberrations:
            lines.append("  Aberrations:")
            for key, value in self.aberrations.items():
                lines.append(f"    {key:<8} {float(value):>12.6g}")
        return "\n".join(lines)
