"""Independent rotation/focus comparisons and their matched phase display."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from quantem.gpu.ssb.contract import RefineMethod
from quantem.gpu.ssb.results import SSBResult, host_array

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from quantem.gpu.ssb.workflow import SSB


@dataclass
class RotationDefocusComparison:
    """Four constrained fits without an automatically selected rotation.

    Attributes
    ----------
    fits : tuple of SSBResult
        Native-grid fits ordered by rotation, then negative/positive C10 range.
    reconstructions : tuple of SSBResult
        Each fit reconstructed with Auto sampling.
    focus_ranges : tuple of tuple of float
        Negative and positive C10 intervals, in nm, both including zero.

    Examples
    --------
    >>> comparison = ssb.compare_rotation_defocus()  # doctest: +SKIP
    >>> comparison.show()  # doctest: +SKIP
    """

    fits: tuple[SSBResult, ...]
    reconstructions: tuple[SSBResult, ...]
    focus_ranges: tuple[tuple[float, float], tuple[float, float]]

    @property
    def status(self) -> str:
        """No branch is certified by phase appearance or fit loss alone."""
        return "Unresolved without an independent focus or rotation constraint."

    def report(self) -> "pd.DataFrame":
        """Return fitted parameters and search bounds as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per constrained fit, including the unresolved status.

        Examples
        --------
        >>> comparison.report()  # doctest: +SKIP
        """
        import pandas as pd

        records = []
        for index, fit in enumerate(self.fits):
            lower, upper = self.focus_ranges[index % 2]
            c10 = fit.aberrations["C10_nm"]
            tilt_row, tilt_col = fit.sample_tilt_mrad or (0.0, 0.0)
            records.append({
                "Rotation (deg)": fit.scan_detector_rotation_deg,
                "C10 range (nm)": (lower, upper),
                "C10 (nm)": c10, "f = -C10 (nm)": -c10,
                "Tilt row (mrad)": tilt_row, "Tilt col (mrad)": tilt_col,
                "Depth spread (nm)": fit.depth_spread_nm,
                "Loss": fit.loss,
                "At zero boundary": abs(c10) < 1e-6,
            })
        table = pd.DataFrame(records)
        table.index.name = self.status
        return table

    def __repr__(self) -> str:
        return self.report().to_string()

    def _repr_html_(self) -> str:
        return self.report()._repr_html_()

    def show(self, *, view_fraction: float = 0.25) -> "Figure":
        """Show matched phase images above native-pixel histograms.

        Parameters
        ----------
        view_fraction : float, optional
            Central fraction of each scan axis to display. Use 1 for the full
            field. Fits and stored arrays are unchanged. One symmetric contrast
            scale is shared; only each region's median phase is subtracted.

        Returns
        -------
        matplotlib.figure.Figure
            Four phase panels, paired histograms, and an unresolved status.

        Examples
        --------
        >>> comparison.show(view_fraction=1)  # doctest: +SKIP
        """
        import matplotlib.pyplot as plt

        cases = [
            {"angle": fit.scan_detector_rotation_deg,
             "C10_nm": fit.aberrations["C10_nm"],
             "tilt_mrad": fit.sample_tilt_mrad or (0.0, 0.0),
             "depth_nm": fit.depth_spread_nm or 0.0,
             "phase": result.phase, "native_phase": fit.phase,
             "sampling_A": result.object_sampling_A,
             "upsampling": result.upsampling_factor}
            for fit, result in zip(self.fits, self.reconstructions, strict=True)
        ]
        focus_ranges = self.focus_ranges

        if not 0 < view_fraction <= 1:
            raise ValueError("Set view_fraction between 0 and 1.")
        images, histograms = [], []
        for case in cases:
            native_height, native_width = case["native_phase"].shape
            for key, collection in (("phase", images), ("native_phase", histograms)):
                phase = case[key]
                height, width = phase.shape
                scale = case["upsampling"] if key == "phase" else 1
                top = int(native_height * (1 - view_fraction) / 2) * scale
                left = int(native_width * (1 - view_fraction) / 2) * scale
                region = host_array(phase[top:height - top, left:width - left])
                collection.append(region - np.median(region))
        limit = np.percentile(np.abs(np.concatenate([p.ravel() for p in images])), 99)
        bins = np.linspace(min(p.min() for p in histograms),
                           max(p.max() for p in histograms), 81)
        colors = ["#9b51b6", "#1487a7"]
        fig, axes = plt.subplots(3, 2, figsize=(10, 11), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1, 1, 0.55]})
        for index, case in enumerate(cases):
            row, col = divmod(index, 2)
            ax = axes[row, col]
            image = ax.imshow(images[index], cmap="inferno", vmin=-limit, vmax=limit,
                              interpolation="nearest",
                              aspect=case["sampling_A"][0] / case["sampling_A"][1])
            c10 = case["C10_nm"]
            tilt = np.linalg.norm(case["tilt_mrad"])
            focus_label = ("C₁₀ = f = 0 nm (boundary)" if abs(c10) < 1e-6 else
                           f"C₁₀ {c10:+.2f} nm · f {(-c10):+.2f} nm")
            ax.set_title(f"{case['angle']:.1f}° · {focus_label}\n"
                         f"Tilt {tilt:.2f} mrad · depth spread {case['depth_nm']:.1f} nm")
            ax.set_xticks([])
            ax.set_yticks([])
            # Integer-unit scale bar, based on the reconstructed object sampling.
            bar_pixels = 10 / case["sampling_A"][1]  # 1 nm = 10 Å
            width, height = images[index].shape[1], images[index].shape[0]
            if bar_pixels < width * 0.6:
                ax.plot([0.06 * width, 0.06 * width + bar_pixels], [0.91 * height] * 2,
                        color="white", linewidth=3)
                ax.text(0.06 * width, 0.86 * height, "1 nm", color="white", weight="bold")
            counts, _ = np.histogram(histograms[index], bins=bins)
            fraction = counts / histograms[index].size
            axes[2, col].plot((bins[:-1] + bins[1:]) / 2,
                             np.where(fraction > 0, fraction, np.nan),
                             color=colors[row], label=f"{case['angle']:.1f}°")
        for col, bounds in enumerate(focus_ranges):
            axes[2, col].set(title=f"C₁₀ search: [{bounds[0]:g}, {bounds[1]:g}] nm",
                             xlabel="Phase − median (rad)", ylabel="Native pixel fraction / bin",
                             yscale="log")
            axes[2, col].set_xlim(bins[0], bins[-1])
            axes[2, col].axvline(0, color="0.6", linewidth=0.8, linestyle=":")
            axes[2, col].legend(frameon=False)
        fig.colorbar(image, ax=axes[:2, :], shrink=0.65, label="Phase − median (rad)")
        fig.suptitle("Four-case review · unresolved without an independent constraint\n"
                     "Two rotations × two C₁₀ signs · SSB", fontsize=14)
        plt.close(fig)
        return fig


def compare_rotation_defocus(
    session: "SSB", *, num_trials: int = 200, refinement: RefineMethod | Literal["auto"] = "auto",
    search_ranges: dict[str, tuple[float, float] | float] | None = None,
    seed: int = 42, tilt: bool = True, verbose: bool = True,
) -> RotationDefocusComparison:
    """Run four independent searches, preserving the session's scientific state."""
    ranges = {"C10_nm": (-30.0, 30.0), "C12_nm": (0.0, 20.0)}
    ranges.update(search_ranges or {})
    bounds = ranges["C10_nm"]
    if np.ndim(bounds) != 1 or len(bounds) != 2 or not bounds[0] < 0 < bounds[1]:
        raise ValueError("Use a C10_nm search range spanning zero, for example (-30, 30).")
    focus_ranges = ((float(bounds[0]), 0.0), (0.0, float(bounds[1])))
    angles = sorted((session.scan_detector_rotation_deg % 360,
                     (session.scan_detector_rotation_deg + 180) % 360))
    rows, cols = session.scan_shape
    full_field = (0, int(rows), 0, int(cols))
    fits, reconstructions = [], []
    from tqdm.auto import tqdm

    with tqdm(total=4, desc="SSB comparison", unit="case", disable=not verbose) as progress:
        for angle in angles:
            for interval in focus_ranges:
                case = f"{angle:.1f}°, C10 [{interval[0]:g}, {interval[1]:g}] nm"
                progress.set_postfix_str(f"{case} | Preparing full field")
                with session.select_region(
                    full_field, scan_detector_rotation_deg=angle,
                    aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
                ) as candidate:
                    refinement_label = "" if refinement is None else f" + {refinement} refinement"
                    progress.set_postfix_str(f"{case} | Fitting: {num_trials} trials{refinement_label}")
                    fitted = candidate.find_aberrations(
                        tilt=tilt, num_trials=num_trials, refinement=refinement,
                        search_ranges={**ranges, "C10_nm": interval}, seed=seed,
                        check_rotation=False, compare_standard=False, verbose=False,
                    )
                    progress.set_postfix_str(f"{case} | Reconstructing full field")
                    reconstructed = candidate.reconstruct(upsampling_factor="auto")
                    fits.append(fitted)
                    reconstructions.append(reconstructed)
                progress.update(1)
        progress.set_postfix_str("Complete")
    return RotationDefocusComparison(tuple(fits), tuple(reconstructions), focus_ranges)
