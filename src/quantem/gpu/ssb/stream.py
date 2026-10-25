"""Stream an acquisition series: defocus and astigmatism refit per frame, every other optic held fixed.

A full fifth-order fit takes seconds to minutes; between frames of a time series the specimen height and
the objective lens drift, but coma, trefoil, spherical aberration and the sample tilt change far less.
``stream`` therefore keeps the reference fit's coordinates and refines only C10 and the two C12 components
on each new frame, starting from the previous frame, with the same model-agreement objective and bounded
Newton step as the full fit. On the 151-191 time series this is about 0.05 s of fitting per frame, so the
per-frame time is dominated by reading the compressed acquisition, far inside the 13 s scan.

Each new phase is also placed live: registered against the frame placed before it by the lattice-safe band
registration of quantem.gpu.drift, with Bragg-spot bands from the first frame, and added in register to a
combined image that grows with the drift. On the time series this adds 0.08 s a frame and agrees with the full pairwise
drift to 0.09 Å over 107 Å of drift; ``SSBStream.align`` then re-places every frame by that pairwise drift.
"""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantem.gpu import io
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.drift import bands, peaks, prepared, register_spectra, shifted, spots
from quantem.gpu.ssb._broad_newton import BudgetReached, Tracker, newton
from quantem.gpu.ssb.fifth_order_fit import _objective
from quantem.gpu.ssb.fifth_order_parameters import public_params
from quantem.gpu.ssb.results import SSBResult, host_array

# Coordinates 0..2 of the fifth-order vector: C10 and the two C12 components.
VARIED = slice(0, 3)
STEP = np.full(3, 0.02)
EVALUATION_CAP = 300
# Pair confidence below which a frame is not added to the combined image. On frames 151-191 every pair
# drift rejected as a wrong region or lattice image scored at most 0.034 and every neighbour pair at
# least 0.050; the live check is a neighbour pair (each frame against the one placed before it), whose
# lowest live score there is 0.043.
CONFIDENCE = 0.04
# The live search reaches as far as drift's pairs: the residual past the extrapolated guess is a few Å.
REACH_A = 30.0


def stream(ptycho, sources, fit: SSBResult, *, upsampling_factor="auto", show: bool = True,
           verbose: bool = True) -> "SSBStream":
    """Refit C10, C12 and phi12 on each source with ``fit``'s higher orders, tilt and depth held.

    Each source is read compressed onto the reference session's GPU and opened with the reference
    calibration (voltage, semiangle, scan step, fitted detector sampling and bright-field disk, rotation),
    so frames differ only in their counts. Frame ``k`` starts from frame ``k - 1``'s low-order values and
    maximizes the normalized model agreement ``A(z) / A(z_start)`` over the three coordinates; its
    conventional SSB phase is reconstructed with the result. ``fit`` must be a fifth-order fit
    (``search="standard"`` or ``"broad"``) of the reference session.

    Each phase is then registered against the last confidently placed frame (``_place``: lattice-safe
    band registration with frame 0's Bragg spots) and added in register to ``combined``. With ``show``
    in a notebook, one figure (latest frame, combined image, drift) refreshes at most twice a second.
    """
    reference = np.array(fit.optimizer["coordinates"])
    bounds = np.array(fit.optimizer["bounds"])[VARIED]
    device = f"cuda:{cp.cuda.runtime.getDevice()}"
    settings = {
        "backend": ptycho.backend, "voltage_kV": ptycho.voltage_kV, "semiangle_mrad": ptycho.semiangle_mrad,
        "scan_sampling_A": ptycho.scan_sampling_A, "detector_sampling_mrad": ptycho.detector_sampling_mrad,
        "bright_field_center_px": ptycho.bright_field_center_px, "bright_field_radius_px": ptycho.bright_field_radius_px,
        "scan_detector_rotation_deg": fit.scan_detector_rotation_deg,
        "scan_detector_rotation_flipped": fit.scan_detector_rotation_flipped,
    }
    sources = [Path(source) for source in sources]
    iterator = sources
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(sources, desc="Streaming defocus and astigmatism", unit="frame")
    start = reference[VARIED].copy()
    phases, aberrations, timings, shifts, confidence = [], [], [], [], []
    panel = _LivePanel(fit.phase_limits, enabled=show)
    for index, source in enumerate(iterator):
        clock = {}
        began = time.perf_counter()
        data = io.load(source, device=device)
        clock["load_s"] = time.perf_counter() - began
        began = time.perf_counter()
        session = type(ptycho)(data, **settings)
        objective = _objective(session)
        clock["setup_s"] = time.perf_counter() - began
        began = time.perf_counter()
        coordinates, evaluations, stop = _refit(objective, reference, start, bounds)
        clock["refit_s"] = time.perf_counter() - began
        began = time.perf_counter()
        optics, tilt, depth = public_params(objective, coordinates, 5)
        chosen = copy.copy(fit)
        chosen.aberrations, chosen.sample_tilt_mrad, chosen.depth_spread_nm = optics, tilt, depth
        result = session.run(chosen, upsampling_factor=upsampling_factor, compute_loss=False,
                             force=True, verbose=False)
        phases.append(cp.asarray(result.phase, dtype=cp.float32))
        sampling = result.object_sampling_A
        clock["reconstruction_s"] = time.perf_counter() - began
        began = time.perf_counter()
        if index == 0:
            lattice, canvas, placed = _lattice(phases[0], sampling), _start(phases[0]), [(0, (0.0, 0.0))]
            shift, score = (0.0, 0.0), float("inf")
        else:
            shift, score = _place(canvas, phases[-1], lattice, sampling, REACH_A, _guess(placed, index))
            if score >= CONFIDENCE:
                placed.append((index, shift))
        shifts.append(shift)
        confidence.append(score)
        cp.cuda.Device().synchronize()
        clock["align_s"] = time.perf_counter() - began
        began = time.perf_counter()
        panel.update(source.name.removesuffix("_master.h5"), phases[-1], canvas, np.array(shifts) * np.array(sampling),
                     sampling, final=index == len(sources) - 1)
        clock["display_s"] = time.perf_counter() - began
        clock["total_s"] = sum(clock.values())
        aberrations.append({**optics, "evaluations": evaluations, "stop": stop})
        timings.append(clock)
        start = coordinates[VARIED].copy()
        del data, session, objective, result
        # Release this frame's resident counts before reading the next one.
        cp.get_default_memory_pool().free_all_blocks()
    return SSBStream(
        frames=tuple(source.name.removesuffix("_master.h5") for source in sources),
        phase=cp.stack(phases), aberrations=tuple(aberrations), timings=tuple(timings),
        reference=fit, object_sampling_A=tuple(float(value) for value in sampling),
        shifts_A=np.array(shifts) * np.array(sampling), confidence=tuple(float(value) for value in confidence),
        combined=_combined(canvas),
    )


@dataclass
class SSBStream:
    """Phases and low-order optics of a streamed series; higher orders, tilt and depth are ``reference``'s."""

    frames: tuple[str, ...]
    phase: object = field(repr=False)
    aberrations: tuple[dict, ...]
    timings: tuple[dict, ...]
    reference: SSBResult = field(repr=False)
    object_sampling_A: tuple[float, float]
    shifts_A: np.ndarray = field(repr=False)
    confidence: tuple[float, ...] = field(repr=False)
    combined: object = field(repr=False)

    def align(self):
        """The final pass: every pair of frames up to three apart by its own lattice-safe search,
        solved by least squares (quantem.gpu.drift.drift), which outvotes a pair that took a wrong
        lattice image where the live pass, one frame against the last, cannot. Replaces shifts_A
        (specimen displacement against frame 0, Å) and rebuilds combined from every frame in
        register; returns the Drift with its pairs, closure and σ."""
        from quantem.gpu.drift import drift

        found = drift(torch.from_dlpack(self.phase), list(self.frames), sampling_A=self.object_sampling_A)
        canvas = _start(self.phase[0])
        for phase, shift in zip(self.phase[1:], found.displacement_A[1:]):
            _add(canvas, phase, tuple(shift / np.array(self.object_sampling_A)))
        self.shifts_A, self.combined = found.displacement_A, _combined(canvas)
        return found

    def report(self, *, to_dict: bool = False):
        """One row per frame: fitted C10, C12, phi12, refit evaluations, the shift against frame 0
        and its confidence (below CONFIDENCE: measured, not added), and seconds per stage."""
        import pandas as pd

        rows = {}
        for frame, optics, clock, shift, score in zip(self.frames, self.aberrations, self.timings, self.shifts_A,
                                                      self.confidence, strict=True):
            rows[frame] = {
                "C10 (nm)": optics["C10_nm"], "C12 (nm)": optics["C12_nm"], "phi12 (deg)": optics["phi12_deg"],
                "Refit evaluations": optics["evaluations"], "Shift row (Å)": shift[0], "Shift col (Å)": shift[1],
                "Confidence": score, "Load (s)": clock["load_s"], "Setup (s)": clock["setup_s"],
                "Refit (s)": clock["refit_s"], "Reconstruction (s)": clock["reconstruction_s"],
                "Align (s)": clock["align_s"], "Display (s)": clock["display_s"], "Total (s)": clock["total_s"],
            }
        return rows if to_dict else pd.DataFrame.from_dict(rows, orient="index").rename_axis("Frame")

    def show(self, *, interactive: bool = False, size="medium", cmap: str = "gray"):
        """Phase of every frame under the reference contrast, with the low-order drift.

        Interactive returns a Show3D stack to scrub through the frames; static shows the first and last
        frame beside C10, C12 and phi12 against frame.
        """
        from quantem.gpu.display._layout import panel_figure, panel_inches, scalebar

        vmin, vmax = self.reference.phase_limits
        if interactive:
            from quantem.widget import Show3D

            return Show3D(host_array(self.phase), labels=list(self.frames), title="Streamed phase (rad)",
                          sampling=self.object_sampling_A[1], units="Å", cmap=cmap, vmin=vmin, vmax=vmax,
                          size=size if isinstance(size, str) else "medium", display_bin=1, verbose=False)
        import matplotlib.pyplot as plt
        from quantem.core.visualization import show_2d

        width = panel_inches(size)
        # The drift strip sits under the two phase panels, with room for both y axes and the x label.
        strip = .6 * width
        figure, axes = panel_figure(1, 2, width, width, title_lines=2, bottom_in=strip + 1.4)
        figure_width, figure_height = figure.get_size_inches()
        ends = [host_array(self.phase[0]), host_array(self.phase[-1])]
        show_2d(ends, title=[f"{self.frames[0]}\nphase (rad)", f"{self.frames[-1]}\nphase (rad)"],
                figax=(figure, axes[0]), cmap=cmap, cbar=True, tight_layout=False,
                norm={"interval_type": "manual", "vmin": vmin, "vmax": vmax},
                scalebar=[scalebar(self.object_sampling_A[1], "Å", end.shape[0], width) for end in ends],
                title_fontsize=min(14, max(8, width * 2.2)))
        drift = figure.add_axes((.9 / figure_width, .6 / figure_height,
                                 1 - 1.9 / figure_width, strip / figure_height))
        index = np.arange(len(self.frames))
        drift.plot(index, [optics["C10_nm"] for optics in self.aberrations], "o-", label="C10 (nm)")
        drift.plot(index, [optics["C12_nm"] for optics in self.aberrations], "s-", label="C12 (nm)")
        drift.set(xlabel="Frame index", ylabel="nm", title="Defocus and astigmatism\nheld: orders 2-5, tilt, depth")
        angle = drift.twinx()
        angle.plot(index, [optics["phi12_deg"] for optics in self.aberrations], "^-", color="gray", label="phi12 (deg)")
        angle.set_ylabel("phi12 (deg)")
        drift.legend(handles=drift.get_lines() + angle.get_lines(), fontsize=8)
        plt.close(figure)
        return figure


class _LivePanel:
    """The stream's live view, only in a notebook kernel: the latest frame, the combined image and
    the drift so far in one figure, redrawn at most twice a second (as FitDisplay.progress) so
    drawing stays a small part of each frame's time."""

    def __init__(self, phase_limits, *, enabled):
        self.phase_limits, self.display, self.handle, self.last_update = phase_limits, None, None, 0.0
        if not enabled:
            return
        from IPython import get_ipython
        from IPython.display import display

        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            self.display = display

    def update(self, frame, phase, canvas, shifts_A, sampling_A, *, final):
        if self.display is None or (not final and time.perf_counter() - self.last_update < 0.5):
            return
        import io

        import matplotlib.pyplot as plt
        from IPython.display import Image

        figure = _live_figure(frame, phase, _combined(canvas), shifts_A, sampling_A, self.phase_limits)
        # One PNG render: the inline backend's tight bounding box draws the figure twice.
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        plt.close(figure)
        if self.handle is None:
            self.handle = self.display(Image(data=buffer.getvalue()), display_id=True)
        else:
            self.handle.update(Image(data=buffer.getvalue()))
        self.last_update = time.perf_counter()


def _live_figure(frame, phase, combined, shifts_A, sampling_A, phase_limits, size="medium"):
    """The latest frame and the combined image under the reference contrast, with scale bars, beside
    each frame's displacement against frame 0 (row and column, Å). Plain imshow at full resolution:
    show_2d builds a float64 RGBA copy of each image first, 3.5 s for these two, longer than a frame."""
    from quantem.core.visualization.visualization_utils import add_scalebar_to_ax

    from quantem.gpu.display._layout import panel_figure, panel_inches, scalebar

    width = panel_inches(size)
    figure, axes = panel_figure(1, 3, width, width, title_lines=2, bottom_in=.6)
    titles = [f"{frame}\nphase (rad)", f"Combined, {len(shifts_A)} frames\nphase (rad)"]
    for axis, image, title in zip(axes[0, :2], (host_array(phase), host_array(combined)), titles):
        # Antialias on the values, not on colours: Matplotlib's automatic choice smooths the RGBA image
        # when it shrinks this much, 3 to 4 s a figure in a notebook against 0.2 s.
        shown = axis.imshow(image, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1], interpolation_stage="data")
        figure.colorbar(shown, ax=axis, fraction=.046, pad=.02)
        bar = scalebar(sampling_A[1], "Å", image.shape[0], width)
        add_scalebar_to_ax(axis, image.shape[1], bar["sampling"], None, "Å", bar["width_px"], 0.5, "white",
                           "lower right", bar["fontsize"])
        axis.set(xticks=[], yticks=[])
        axis.set_title(title, fontsize=min(14, max(8, width * 2.2)))
    trace = axes[0, 2]
    trace.plot(shifts_A[:, 0], "o-", ms=3, label="row")
    trace.plot(shifts_A[:, 1], "s-", ms=3, label="column")
    trace.set(xlabel="Frame index", title="Displacement against\nframe 0 (Å)")
    # Ticks on the right, clear of the combined image's colorbar in the gap before it.
    trace.yaxis.tick_right()
    trace.legend(fontsize=8, frameon=False)
    return figure


def _combined(canvas):
    """The frames in register averaged where they reach: total over cover, zero where none does."""
    return (canvas.total / cp.maximum(canvas.cover, 1)).astype(cp.float32)


def _refit(objective, reference, start, bounds):
    """Bounded Newton over (C10, C12 components) with the other 25 coordinates at ``reference``."""
    base = float(objective.evaluate(np.concatenate([start, reference[3:]])[None], 5)[0])

    def evaluate(points):
        full = np.tile(reference, (len(points), 1))
        full[:, VARIED] = points
        return objective.evaluate(full, 5) / base

    tracker = Tracker(evaluate, start, bounds, EVALUATION_CAP)
    try:
        tracker(start)
        _, stop = newton(tracker, start, bounds, STEP)
    except BudgetReached:
        stop = "evaluation cap"
    coordinates = reference.copy()
    coordinates[VARIED] = tracker.best_z
    return coordinates, tracker.nfev, stop




def _lattice(phase, sampling_A, band_A=(50.0, 0.67)):
    """What the live registration balances, from the first frame alone: each lattice's Bragg spots
    (spots) and every other spot to cut from the aperiodic part (peaks: moiré and its harmonics), in
    Å⁻¹, so the bands can be drawn on any crop (bands); drift finds them on the series' mean power
    spectrum, which a stream does not have yet."""
    power = torch.fft.fft2(prepared(torch.from_dlpack(phase).float())).abs().square()
    return SimpleNamespace(layers=spots(power, sampling_A), notch=peaks(power, sampling_A, band_A), band_A=band_A)



def _start(phase):
    """The sum of the frames in register, begun with frame 0: total and cover (how many frames reach
    each pixel) on a canvas whose pixel origin is frame 0's top-left corner, grown as the drift
    carries later frames past it; last is the frame added last, moved into register (its whole-pixel
    shift whole, its sub-pixel-shifted copy piece), the reference the next frame is registered to."""
    return SimpleNamespace(total=cp.asarray(phase, dtype=cp.float64), cover=cp.ones(phase.shape, dtype=cp.float32),
                           origin=(0, 0), last=SimpleNamespace(whole=(0, 0), piece=cp.asarray(phase, dtype=cp.float64)))


def _guess(placed, index):
    """Where frame index should sit, px: the last two placed frames' (index, shift_px) extrapolated
    at their velocity, so the residual the registration finds is the drift's change of speed (about
    1 Å a frame on the time series), not its speed (up to 5 Å a frame)."""
    if len(placed) < 2:
        return placed[-1][1]
    (before, earlier), (after, later) = placed[-2], placed[-1]
    return tuple(late + (late - early) * (index - after) / (after - before) for early, late in zip(earlier, later))


def _place(canvas, phase, lattice, sampling_A, reach_A, guess_px):
    """Register one frame against the frame added last and, when confident, add it in register.

    The frame is expected at n = round(guess_px) px from frame 0 and the last one sits at m. Both are
    cut to the rectangle they share there, so each crop holds content everywhere as the Hann-tapered
    correlation assumes, and the residual d - n between the crops is small, well inside reach_A where
    the whole drift (107 Å over the 40 frames) would not be. register_spectra with the first
    frame's bands gives it with moving(r) = reference(r - residual), so the frame's shift against
    frame 0 is d = n + residual, px. The reference is the last frame, not the sum of every earlier
    one: on the time series the specimen changes along the series, and even with every earlier frame placed at
    the pairwise drift the full sum took a wrong lattice image for frames 28 to 34, where the last
    frame alone agrees with the pairwise drift to 0.02 Å. The confidence is drift's pair confidence:
    the lesser of the fine parts' lead over the next lattice image and the coarse part's lead over its
    next region. A frame below CONFIDENCE is measured but not added: a wrong lattice image would smear
    the sum by 3.2 Å, and the next frame is registered to the last confident one."""
    rows, cols = phase.shape
    whole = tuple(round(value) for value in guess_px)
    last = canvas.last.whole
    # Frame 0's pixel r holds what this frame shows at r + n and the last frame's piece at r + m.
    first_row, first_col = max(-whole[0], -last[0]), max(-whole[1], -last[1])
    end_row, end_col = min(-whole[0], -last[0]) + rows, min(-whole[1], -last[1]) + cols
    reference = canvas.last.piece[first_row + last[0]:end_row + last[0], first_col + last[1]:end_col + last[1]]
    moving = phase[first_row + whole[0]:end_row + whole[0], first_col + whole[1]:end_col + whole[1]]
    spectra = [torch.fft.fft2(prepared(torch.from_dlpack(cp.ascontiguousarray(image)).float())) for image in (reference, moving)]
    masks = bands(spectra[0].real, sampling_A, lattice.layers, lattice.band_A, notch=lattice.notch)
    found = register_spectra(spectra[0], spectra[1], masks, sampling_A, reach_A)
    shift = tuple(step + value for step, value in zip(whole, found.shift_px))
    confidence = min(found.margin, found.parts["coarse"] - found.best["coarse_second"])
    if confidence >= CONFIDENCE:
        _add(canvas, phase, shift)
    return shift, confidence


def _add(canvas, phase, shift_px):
    """Add one frame to the sum in register: with d = n + f (n whole pixels, |f| <= 1/2), the frame
    seen at r shows frame 0's specimen at r - d, so its Fourier-shifted copy phase(r + f) (shifted,
    exact for a derived phase image) joins the canvas with its corner at origin - n, whole pixels."""
    whole = tuple(round(value) for value in shift_px)
    top, left = _grow(canvas, whole, phase.shape)
    rows, cols = phase.shape
    piece = shifted(phase, [whole[0] - shift_px[0], whole[1] - shift_px[1]])
    canvas.total[top:top + rows, left:left + cols] += piece
    canvas.cover[top:top + rows, left:left + cols] += 1
    canvas.last = SimpleNamespace(whole=whole, piece=piece)

def _grow(canvas, whole, shape):
    """Pad the canvas with empty pixels until it holds a frame moved by whole pixels; return that
    frame's top-left corner on it."""
    top, left = canvas.origin[0] - whole[0], canvas.origin[1] - whole[1]
    before = (max(0, -top), max(0, -left))
    after = (max(0, top + shape[0] - canvas.total.shape[0]), max(0, left + shape[1] - canvas.total.shape[1]))
    if any(before) or any(after):
        width = ((before[0], after[0]), (before[1], after[1]))
        canvas.total, canvas.cover = cp.pad(canvas.total, width), cp.pad(canvas.cover, width)
        canvas.origin = (canvas.origin[0] + before[0], canvas.origin[1] + before[1])
    return top + before[0], left + before[1]
