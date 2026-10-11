"""Result container for parallax reconstruction."""
from dataclasses import dataclass


@dataclass
class ParallaxResult:
    """Result from parallax reconstruction.

    ``image`` and ``density`` are CuPy arrays on the acquisition's GPU:
    ``image`` sums the bright-field images, each shifted once by its measured
    shift, and ``density`` counts the samples reaching each output pixel
    (the number of bright-field pixels when not upsampled), so
    ``image / density`` is their mean. ``shifts`` holds one ``(row, col)``
    shift in scan pixels per bright-field pixel, in row-major detector order;
    ``elapsed`` is in seconds.

    ``aberrations`` (``C10`` and ``C12`` in nm, ``phi12`` in radians, scan
    frame) and ``rotation_angle_deg`` hold the fit of
    ``parallax.run(fit_aberrations=True)``, in the units and convention
    ``SSB(aberrations=..., scan_detector_rotation_deg=...)`` takes; both are None
    without a fit.
    """

    image: object
    density: object
    shifts: list[tuple[float, float]]
    aberrations: dict[str, float] | None = None
    rotation_angle_deg: float | None = None
    elapsed: float | None = None

    def __repr__(self) -> str:
        lines = ["ParallaxResult:"]
        lines.append(f"  Image shape:  {tuple(self.image.shape)}")
        if self.shifts:
            lines.append(f"  BF positions: {len(self.shifts)}")
        if self.elapsed is not None:
            lines.append(f"  Elapsed:      {self.elapsed:.2f}s")
        if self.aberrations:
            lines.append(f"  C10:          {self.aberrations['C10']:.3f} nm")
            lines.append(f"  C12:          {self.aberrations['C12']:.3f} nm")
            lines.append(f"  phi12:        {self.aberrations['phi12']:.4f} rad")
            lines.append(f"  Rotation:     {self.rotation_angle_deg:.2f} deg")
        return "\n".join(lines)
