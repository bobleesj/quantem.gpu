"""Result containers for parallax reconstruction."""
from __future__ import annotations

import math
from dataclasses import dataclass


_PARALLAX_ABERRATION_KEYS = {"C10", "C12", "phi12"}


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
    """Result from parallax reconstruction.

    Notes
    -----
    ``aberrations`` follows the original QuantEM parallax convention:
    ``C10`` and ``C12`` are in Angstroms, while ``phi12`` and
    ``rotation_angle`` are in radians. Use :meth:`to_ssb_aberrations` or
    :meth:`to_ssb_kwargs` before passing fitted parallax aberrations to SSB,
    which expects nanometers and degrees.
    """

    image: object
    density: object
    shifts: list[tuple[float, float]]
    bf_images: list[BFImage] | None = None
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
                "ParallaxResult has no fitted aberrations. Run parallax(..., "
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

    def to_ssb_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for an SSB reconstruction seeded by parallax.

        Examples
        --------
        >>> prlx = parallax(data, scan_shape=(256, 256), fit_aberrations=True,
        ...                 scan_sampling=0.5)
        >>> ssb_result = ssb(data, voltage_kV=300, semiangle_mrad=30,
        ...                  scan_sampling_A=0.5, **prlx.to_ssb_kwargs())
        """
        return {
            "aberrations": self.to_ssb_aberrations(),
            "rotation_angle_deg": self.rotation_angle_deg(),
        }

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
