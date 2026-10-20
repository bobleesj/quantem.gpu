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
    ``elapsed_s`` is in seconds.

    ``aberrations`` (``C10_nm`` and ``C12_nm`` in nm, ``phi12_deg`` in degrees,
    scan frame) and ``scan_detector_rotation_deg`` (degrees, in [0, 360)) hold
    the fit of ``parallax.run(fit_aberrations=True)``, in the units and convention
    ``SSB(aberrations=..., scan_detector_rotation_deg=...)`` takes; both are None
    without a fit.
    """

    image: object
    density: object
    shifts: list[tuple[float, float]]
    aberrations: dict[str, float] | None = None
    scan_detector_rotation_deg: float | None = None
    elapsed_s: float | None = None

    def __repr__(self) -> str:
        lines = ["ParallaxResult:"]
        lines.append(f"  Image shape:  {tuple(self.image.shape)}")
        if self.shifts:
            lines.append(f"  BF positions: {len(self.shifts)}")
        if self.elapsed_s is not None:
            lines.append(f"  Elapsed:      {self.elapsed_s:.2f}s")
        if self.aberrations:
            lines.append(f"  C10:          {self.aberrations['C10_nm']:.3f} nm")
            lines.append(f"  C12:          {self.aberrations['C12_nm']:.3f} nm")
            lines.append(f"  phi12:        {self.aberrations['phi12_deg']:.2f}°")
            lines.append(f"  Rotation:     {self.scan_detector_rotation_deg:.2f}°")
        return "\n".join(lines)
