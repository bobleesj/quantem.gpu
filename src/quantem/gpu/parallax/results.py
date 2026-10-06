"""Result containers for parallax reconstruction."""
import math
from dataclasses import dataclass

_PARALLAX_ABERRATION_KEYS = {"C10", "C12", "phi12"}


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

    Notes
    -----
    ``aberrations`` follows the original QuantEM parallax convention:
    ``C10`` and ``C12`` are in Angstroms, while ``phi12`` and
    ``rotation_angle`` are in radians. Use :meth:`to_ssb_aberrations` and
    :meth:`rotation_angle_deg` before passing fitted parallax aberrations to SSB,
    which expects nanometers and degrees.
    """

    image: object
    density: object
    shifts: list[tuple[float, float]]
    aberrations: dict | None = None
    elapsed: float | None = None

    def to_ssb_aberrations(self) -> dict[str, float]:
        """Return fitted aberrations in the format expected by SSB.

        Returns
        -------
        dict[str, float]
            ``{"C10", "C12", "phi12"}`` with ``C10``/``C12`` in nm and
            ``phi12`` in radians.

        Raises
        ------
        ValueError
            If this result does not contain a complete parallax aberration fit.
        """
        if self.aberrations is None:
            raise ValueError(
                "ParallaxResult has no fitted aberrations. Run parallax.run(..., "
                "fit_aberrations=True, scan_sampling=...) first."
            )
        missing = _PARALLAX_ABERRATION_KEYS - self.aberrations.keys()
        if missing:
            raise ValueError(
                "Parallax aberrations must contain C10, C12, and phi12 before "
                f"conversion to SSB format. Missing: {sorted(missing)}."
            )
        return {
            "C10": float(self.aberrations["C10"]) / 10.0,
            "C12": float(self.aberrations["C12"]) / 10.0,
            "phi12": float(self.aberrations["phi12"]),
        }

    def rotation_angle_deg(self) -> float:
        """Return fitted scan-detector rotation angle in degrees.

        Missing ``rotation_angle`` is treated as zero because some parallax
        workflows fit only aberration coefficients.
        """
        if self.aberrations is None:
            return 0.0
        return math.degrees(float(self.aberrations.get("rotation_angle", 0.0)))

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
