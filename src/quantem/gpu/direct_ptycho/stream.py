"""Reconstruct a recorded acquisition series: defocus and astigmatism refit per frame, every other optic held fixed.

A full fifth-order fit takes seconds to minutes; between frames of a time series the specimen height and
the objective lens drift, but coma, trefoil, spherical aberration and the sample tilt change far less.
``series`` therefore keeps the reference fit's coordinates and refines only C10 and the two C12 components
on each new frame, starting from the previous frame, with the same model-agreement objective and exact-gradient
trust-region refinement as the full fit. On the measured series (frames 151-191) this is about 0.05 s of fitting per frame, so the
per-frame time is dominated by reading the compressed acquisition, far inside the 13 s scan.

Each new phase is also placed live: registered against the frame placed before it by the lattice-safe band
registration of quantem.gpu.drift, with Bragg-spot bands from the first frame, and added in register to a
combined image that grows with the drift. On the measured series this adds 0.013 s a frame (median) and
agrees with the full pairwise drift to 0.09 Å over 107 Å of drift; ``SSBSeries.align`` then re-places every frame by that pairwise drift.

With ``save_dir`` each frame's optics and shift are written as it finishes, so a restarted kernel
reconstructs the finished frames from them, with no refit and no registration, and continues.

``method`` adds the regularized phase-only and amplitude + phase reconstructions at each frame's optics,
moved by the shifts measured on the SSB phase: on a measured 40-frame series at 1536 x 1536 (2026-10-24,
one GPU) they cost a median 0.039 s and 0.050 s a frame beside SSB's 0.034 s, 0.84 s a frame in all,
with shifts identical to SSB alone and a 26 GB peak above the reference session.
"""

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantem.gpu import io
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.drift import bands, peaks, prepared, register_spectra, shifted, spots
from quantem.gpu.resident.hot_pixels import hot_pixel_outcome
from quantem.gpu.direct_ptycho.fifth_order_fit import _objective, _refine_with_gradients
from quantem.gpu.direct_ptycho.fifth_order_parameters import public_params
from quantem.gpu.direct_ptycho.persistence import _write_atomically, acquisition_identity, json_value, software_signature
from quantem.gpu.direct_ptycho.results import SSBResult, host_array

# Coordinates 0..2 of the fifth-order vector: C10 and the two C12 components.
VARIED = slice(0, 3)
EVALUATION_CAP = 300
# Pair confidence below which a frame is not added to the combined image. On the measured series (frames 151-191) every pair
# drift rejected as a wrong region or lattice image scored at most 0.034 and every neighbour pair at
# least 0.050; the live check is a neighbour pair (each frame against the one placed before it), whose
# lowest live score there is 0.043.
CONFIDENCE = 0.04
# The live search reaches as far as drift's pairs: the residual past the extrapolated guess is a few Å.
REACH_A = 30.0
# The layout of a saved frame record; a record of another layout is computed again.
# 3: the shared settings name the output factor upsampling (upsampling_factor in 2).
# 4: each frame's C10/C12 refit is the exact-gradient trust region, held coordinates fixed (main's /3).
RECORD_SCHEMA = "quantem.ssb-stream-frame/4"


class _FrameStep:
    """The per-frame work of ``series``: one recorded or resident frame at a time.

    For each frame: open a session on its counts with the reference calibration, refit the
    three low-order coordinates (C10 and the two C12 components) from the previous frame's
    values with every other optic held at ``fit``'s, reconstruct each requested method at
    those optics, and register the SSB phase against the last confidently placed frame onto
    one combined canvas per method. Both entry points call these methods, so a live frame
    and a recorded frame are processed by the same code.
    """

    def __init__(self, ptycho, fit: SSBResult, methods):
        self.fit = fit
        self.methods = methods
        # SSB first: its phase is what each frame is registered on.
        self.runs = ("ssb", *(name for name in methods if name != "ssb"))
        self.reference = np.array(fit.optimizer["coordinates"])
        self.bounds = np.array(fit.optimizer["bounds"])[VARIED]
        self.start = self.reference[VARIED].copy()
        self.kind = type(ptycho)
        self.settings = {
            "backend": ptycho.backend, "voltage_kV": ptycho.voltage_kV, "semiangle_mrad": ptycho.semiangle_mrad,
            "scan_sampling_A": ptycho.scan_sampling_A, "detector_sampling_mrad": ptycho.detector_sampling_mrad,
            "bright_field_center_px": ptycho.bright_field_center_px,
            "bright_field_radius_px": ptycho.bright_field_radius_px,
            "scan_detector_rotation_deg": fit.scan_detector_rotation_deg,
            "scan_detector_rotation_flipped": fit.scan_detector_rotation_flipped,
        }
        self.lattice = self.canvases = None
        self.placed = []

    def open(self, data):
        """A session on one frame's counts with the reference calibration, so frames differ only in counts."""
        return self.kind(data, **self.settings)

    def refit(self, session, clock, began):
        """Maximize the normalized model agreement over C10, C12 and phi12 from the previous frame's values.

        ``began`` is when the frame's setup started (before its session opened), so ``setup_s`` covers both.
        """
        objective = _objective(session, gradients=True)
        clock["setup_s"] = time.perf_counter() - began
        began = time.perf_counter()
        coordinates, evaluations, stop = _refit(objective, self.reference, self.start, self.bounds)
        clock["refit_s"] = time.perf_counter() - began
        optics, tilt, depth = public_params(objective, coordinates, 5)
        return objective, coordinates, evaluations, stop, optics, tilt, depth

    def reconstruct(self, session, optics, tilt, depth, upsampling, strengths):
        """Every requested method at the frame's optics; returns the fit copy that carries them."""
        chosen = copy.copy(self.fit)
        chosen.aberrations, chosen.sample_tilt_mrad, chosen.depth_spread_nm = optics, tilt, depth
        reconstructed, sampling, seconds = _reconstruct(session, chosen, self.runs, upsampling, strengths)
        return chosen, reconstructed, sampling, seconds

    def place(self, index, reconstructed, sampling, saved=None):
        """Register the frame's SSB phase and add every method's image to its canvas in register.

        Frame 0 starts the canvases and has no pair to score (infinite confidence). Later frames are
        registered against the last confidently placed frame unless ``saved`` gives the shift and
        confidence already measured; a frame below ``CONFIDENCE`` stays out of every combined image.
        """
        if index == 0:
            self.lattice = _lattice(reconstructed["ssb"], sampling)
            # The SSB canvas registers the frames; every other image only follows its shifts.
            self.canvases = {key: _start(reconstructed[key]) for key in ("ssb", *_keys(self.methods)) if key in reconstructed}
            self.placed = [(0, (0.0, 0.0))]
            return (0.0, 0.0), float("inf")
        if saved is None:
            shift, score = _place(self.canvases["ssb"], reconstructed["ssb"], self.lattice, sampling, REACH_A,
                                  _guess(self.placed, index))
        else:
            shift, score = saved
            if score >= CONFIDENCE:
                _add(self.canvases["ssb"], reconstructed["ssb"], shift)
        if score >= CONFIDENCE:
            self.placed.append((index, shift))
            for key, canvas in self.canvases.items():
                if key != "ssb":
                    _add(canvas, reconstructed[key], shift)
        return shift, score

    def advance(self, coordinates):
        """The next frame's refit starts from this frame's low-order values."""
        self.start = coordinates[VARIED].copy()


def series(ptycho, sources, fit: SSBResult, *, refit="defocus", method="ssb", upsampling="auto", save_dir=None,
           show: bool = True, verbose: bool = True, **strengths) -> "SSBSeries":
    """Refit C10, C12 and phi12 on each source with ``fit``'s higher orders, tilt and depth held.

    Each source, a recorded file or a scan already resident as an ANS dataset (``ArinaListener.datasets``), is
    read compressed onto the reference session's GPU and opened with the reference
    calibration (voltage, semiangle, scan step, fitted detector sampling and bright-field disk, rotation),
    so frames differ only in their counts. Frame ``k`` starts from frame ``k - 1``'s low-order values and
    maximizes the normalized model agreement ``A(z) / A(z_start)`` over the three coordinates; its
    conventional SSB phase is reconstructed with the result, and so is each ``method`` (``"ssb"``,
    ``"phase"``, ``"complex"`` or a tuple of them) at the same optics, the regularized ones with
    ``strengths`` (``run``'s ``phase_damping``, ``amplitude_damping``, ``smoothness_inv_A``,
    ``detector_weighting``). ``fit`` must be a fifth-order fit (``search="standard"`` or ``"broad"``) of
    the reference session. ``refit="full"`` instead fits every frame independently (``fit_aberrations``,
    standard search, the reference rotation kept): the baseline the defocus refit is compared with.

    Each SSB phase is then registered against the last confidently placed frame (``_place``: lattice-safe
    band registration with frame 0's Bragg spots), whatever the methods: the regularized images move by
    the shift measured on it, so every combined image shares one registration. With ``show`` in a
    notebook, one figure (latest frame and combined image of the first method, drift) refreshes at most
    twice a second.

    With ``save_dir``, ``<save_dir>/stream/<frame>.json`` records each frame's optics, shift, confidence
    and timings. A later call with the same ``fit`` and sources reconstructs a recorded frame from its
    saved optics and places it at its saved shift, then refits the first unrecorded frame from the last
    recorded one, so the series equals one uninterrupted run (``_recorded`` says what must match). The
    records hold no images, so a resumed call may ask for other methods or strengths.

    The frames are added one at a time (``SSBSeries.add``), so a series given no sources is the same
    series built while scans arrive: each ``add`` is one more frame of this loop.
    """
    # Files are read one at a time and freed after their frame; scans already resident (a live stream's
    # encoded datasets) are used in place and stay the caller's.
    sources = [Path(source) if isinstance(source, (str, Path)) else source for source in sources]
    result = SSBSeries._begin(ptycho, fit, refit=refit, method=method, upsampling=upsampling, save_dir=save_dir,
                              show=show, strengths=strengths, frames=len(sources))
    iterator = sources
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(sources, desc="Fitting every frame" if refit == "full" else "Refitting defocus and astigmatism",
                        unit="frame")
    for index, source in enumerate(iterator):
        result._add(source, final=index == len(sources) - 1)
    # The registration's torch blocks are reused frame to frame; return them once the series is done.
    torch.cuda.empty_cache()
    if verbose:
        result._announce_flagged()
    return result


@dataclass
class SSBSeries:
    """A recorded series: per frame the low-order optics (``aberrations``, higher orders, tilt and depth
    are ``reference``'s), the exact parameters each reconstruction received (``fits``), the refit's
    bookkeeping (``refits``: evaluations, stop, resumed from a record), the SSB phase every frame is
    registered on (``phase``), one stack per requested image (``images``: each method's phase in rad,
    and ``"amplitude"`` with ``"complex"``), each image combined in register (``combined``) and how
    many frames reach each of its pixels (``cover``); after ``align``, ``sigma_A`` holds the pairwise
    solve's σ per frame (row, col, Å)."""

    frames: tuple[str, ...]
    phase: object = field(repr=False)
    images: dict = field(repr=False)
    methods: tuple[str, ...]
    aberrations: tuple[dict, ...]
    refits: tuple[dict, ...] = field(repr=False)
    fits: tuple[SSBResult, ...] = field(repr=False)
    timings: tuple[dict, ...] = field(repr=False)
    reference: SSBResult = field(repr=False)
    object_sampling_A: tuple[float, float]
    shifts_A: np.ndarray = field(repr=False)
    confidence: tuple[float, ...] = field(repr=False)
    placed: tuple[bool, ...] = field(repr=False)
    combined: dict = field(repr=False)
    cover: object = field(repr=False)
    sources: tuple = field(repr=False)  # recorded files (Path) or resident ANS datasets
    session: dict = field(repr=False)
    sigma_A: np.ndarray | None = field(default=None, repr=False)
    # What adding the next frame needs: the frame step, the record chain and the growing image stacks.
    _state: SimpleNamespace | None = field(default=None, repr=False)

    @classmethod
    def _begin(cls, ptycho, fit, *, refit, method, upsampling, save_dir, show, strengths, frames=0):
        """An empty series on ``ptycho``'s calibration; ``frames`` sizes the image stacks when known."""
        if refit not in ("defocus", "full"):
            raise ValueError(f'refit={refit!r}: use "defocus" (C10, C12, phi12 from the frame before) or "full" '
                             "(fit_aberrations on every frame).")
        records = None if save_dir is None else Path(save_dir).expanduser() / "stream"
        if records is not None and refit == "full":
            raise ValueError('save_dir resumes defocus refits; refit="full" keeps no frame records.')
        methods = (method,) if isinstance(method, str) else tuple(dict.fromkeys(method))
        step = _FrameStep(ptycho, fit, methods)
        device = f"cuda:{cp.cuda.runtime.getDevice()}"
        # Every frame has the reference calibration, so "auto" is resolved (and a capped factor announced) once.
        factor = ptycho._resolve_upsampling(upsampling)
        # What every frame's record shares: the reference fit it refits from and the code that ran it.
        shared = None if records is None else {
            "reference": {"coordinates": fit.optimizer["coordinates"], "bounds": fit.optimizer["bounds"],
                          "signature": fit.metadata.get("signature")},
            "settings": {**step.settings, "upsampling": upsampling}, "software": software_signature(),
        }
        state = SimpleNamespace(
            step=step, refit=refit, strengths=strengths, factor=factor, records=records, shared=shared,
            resuming=records is not None, previous=None, panel=_LivePanel(fit.phase_limits, methods[0], enabled=show),
            # What each load did with the detector's flagged pixels, announced once for the series.
            flagged={}, capacity=frames, stacks=None, scan=None, views=[], frozen=False,
        )
        return cls(frames=(), phase=None, images={}, methods=methods, aberrations=(), refits=(), fits=(), timings=(),
                   reference=fit, object_sampling_A=None, shifts_A=np.zeros((0, 2)), confidence=(), placed=(),
                   combined={}, cover=None, sources=(),
                   session={"kind": type(ptycho), "settings": step.settings, "upsampling": factor, "device": device},
                   _state=state)

    def add(self, scan) -> dict:
        """Add one more completed scan: refit (or fit), reconstruct, place it, and return its report row.

        ``scan`` is a recorded file or a resident ANS dataset (one ``ArinaListener`` scan); it is processed
        exactly as ``series`` processes its next frame, so a series built scan by scan as scans arrive
        equals ``series`` of the same scans. A resident scan is named by its acquisition (``series``
        metadata). Open interactive views (``show(interactive=True)``) are redrawn with it. Pass
        ``series.add`` as an ``ArinaListener`` ``process`` to reconstruct every scan while reception continues.
        """
        with cp.cuda.Device(int(self.session["device"].split(":")[1])):
            self._add(Path(scan) if isinstance(scan, (str, Path)) else scan, final=True)
        return {"Frame": self.frames[-1], **self.report(to_dict=True)[self.frames[-1]]}

    def _add(self, source, *, final):
        """One frame of the series: load (a file), open the session with the reference calibration, refit,
        reconstruct each method, store the images, place the SSB phase, refresh the views, write the record."""
        state, step, methods = self._state, self._state.step, self.methods
        if state.frozen:
            raise RuntimeError("This series was aligned or reconstructed again; start a new series to add scans.")
        index = len(self.frames)
        resident = not isinstance(source, Path)
        if resident and state.records is not None:
            raise ValueError("save_dir resumes a series of recorded files; resident scans have no file to identify "
                             "them by. Leave save_dir unset, or save the scans first and pass their paths.")
        if resident:
            name = str(source.metadata.get("series") or f"scan {index + 1}")
        else:
            name = source.name.removesuffix("_master.h5")
        clock = {}
        began = time.perf_counter()
        data = source if resident else io.load(source, device=self.session["device"], verbose=False)
        clock["load_s"] = time.perf_counter() - began
        if "hot_pixel_correction" in data.metadata:
            state.flagged.setdefault(hot_pixel_outcome(data.metadata["hot_pixel_correction"]), []).append(name)
        scan = tuple(data.shape[:2])
        if state.scan is None:
            state.scan = scan
        elif scan != state.scan:
            # Frames are placed by shifting whole images onto one canvas; another scan size is another field.
            raise ValueError(f"{name} has scan shape {scan}, the series began with {state.scan}: "
                             "pass frames of one scan shape")
        began = time.perf_counter()
        session = step.open(data)
        saved = objective = None
        if state.records is not None:
            # A frame's refit starts from the frame before and its placement from the frames placed
            # before it, so its identity holds the SHA-256 of the previous frame's record, which holds
            # the one before it: a recomputed frame invalidates every later record, also when the run
            # stopped before rewriting them, and only an unbroken chain from the first frame is reused.
            identity = {**state.shared, "index": index, "previous": state.previous, "source": acquisition_identity(
                source, hot_pixel_correction=data.metadata.get("hot_pixel_correction"))}
            saved = _recorded(state.records / f"{name}.json", identity) if state.resuming else None
            state.resuming = saved is not None
        if saved is not None:
            clock["setup_s"], clock["refit_s"] = time.perf_counter() - began, 0.0
            coordinates, evaluations, stop = np.array(saved["coordinates"]), saved["evaluations"], saved["stop"]
            optics, tilt, depth = saved["aberrations"], tuple(saved["sample_tilt_mrad"]), saved["depth_spread_nm"]
        elif state.refit == "full":
            clock["setup_s"] = time.perf_counter() - began
            began = time.perf_counter()
            # The reference rotation is kept: the session is opened with it and a flipped frame is another calibration.
            fitted = session.fit_aberrations(check_rotation=False, compute_loss=False, show=False, verbose=False)
            clock["refit_s"] = time.perf_counter() - began
            coordinates, evaluations = np.array(fitted.optimizer["coordinates"]), fitted.optimizer.get("evaluations")
            stop = fitted.optimizer.get("stop", "full fit")
            optics, tilt, depth = fitted.aberrations, fitted.sample_tilt_mrad, fitted.depth_spread_nm
        else:
            objective, coordinates, evaluations, stop, optics, tilt, depth = step.refit(session, clock, began)
        chosen, reconstructed, sampling, seconds = step.reconstruct(session, optics, tilt, depth, state.factor,
                                                                    state.strengths)
        clock.update(seconds)
        began = time.perf_counter()
        phase, images = self._store(index, reconstructed)
        shift, score = step.place(index, reconstructed, sampling,
                                  None if saved is None or index == 0 else (tuple(saved["shift_px"]), saved["confidence"]))
        cp.cuda.Device().synchronize()
        clock["align_s"] = time.perf_counter() - began
        clock["display_s"] = clock["total_s"] = 0.0
        # Published in one statement, after the frame is complete: a view or report read from another thread
        # (a live stream's worker adds scans while the notebook reads) sees whole frames only.
        (self.phase, self.images, self.object_sampling_A, self.frames, self.sources, self.shifts_A, self.confidence,
         self.placed, self.combined, self.cover, self.aberrations, self.refits, self.fits, self.timings) = (
            phase, images, tuple(float(value) for value in sampling), (*self.frames, name), (*self.sources, source),
            np.vstack([self.shifts_A, np.array(shift) * np.array(sampling)]), (*self.confidence, float(score)),
            (*self.placed, score >= CONFIDENCE), {key: _combined(step.canvases[key]) for key in images},
            step.canvases["ssb"].cover, (*self.aberrations, dict(optics)),
            (*self.refits, {"evaluations": evaluations, "stop": stop, "resumed": saved is not None}),
            (*self.fits, chosen), (*self.timings, clock))
        began = time.perf_counter()
        state.panel.update(name, images[methods[0]][index], step.canvases[methods[0]], self.shifts_A, sampling,
                           len(step.placed), final=final)
        self._redraw()
        clock["display_s"] = time.perf_counter() - began
        clock["total_s"] = sum(clock.values())
        if state.records is not None and saved is None:
            # Frame 0 has no pair to score (its confidence is infinite); strict JSON has no Infinity.
            record = json_value({
                "schema": RECORD_SCHEMA, "identity": identity, "frame": name, "aberrations": optics,
                "sample_tilt_mrad": tilt, "depth_spread_nm": depth, "coordinates": coordinates,
                "evaluations": evaluations, "stop": stop, "shift_px": shift,
                "shift_A": np.array(shift) * np.array(sampling), "confidence": None if index == 0 else score,
                "object_sampling_A": sampling, "timings": clock,
            })
            _write_atomically(state.records / f"{name}.json", "w",
                              lambda handle: json.dump(record, handle, indent=1, allow_nan=False), encoding="utf-8")
        if state.records is not None:
            state.previous = hashlib.sha256(json.dumps(saved or record, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if state.refit == "defocus":
            step.advance(coordinates)
        # close breaks the engine's cycle with its thick-sample model, so the frame's 10 GB of resident
        # counts and spectra go now; after del alone they waited for the cycle collector (measured
        # series: 66 GB process peak over 40 frames, 40.5 GB with close).
        session.close()
        # The session borrows the dataset, so a loaded file's encoded resident is released here.
        if not resident:
            data.close()
        del data, session, objective, reconstructed
        cp.get_default_memory_pool().free_all_blocks()

    def _store(self, index, reconstructed):
        """Write frame ``index``'s images into one stack per image, filled in place (stacking a list at the end
        held every frame twice), and return the SSB stack and the images through this frame; a series of
        unknown length doubles its stacks when full."""
        state = self._state
        if state.stacks is None:
            phase = cp.empty((max(state.capacity, 1), *reconstructed["ssb"].shape), dtype=cp.float32)
            state.stacks = {"ssb": phase, **{key: cp.empty_like(phase) for key in _keys(self.methods) if key != "ssb"}}
        elif index == len(state.stacks["ssb"]):
            grown = {}
            for key, stack in state.stacks.items():
                grown[key] = cp.empty((2 * len(stack), *stack.shape[1:]), dtype=stack.dtype)
                grown[key][:index] = stack
            state.stacks = grown
        for key, stack in state.stacks.items():
            stack[index] = reconstructed[key]
        phase = state.stacks["ssb"][:index + 1]
        return phase, {key: phase if key == "ssb" else state.stacks[key][:index + 1] for key in _keys(self.methods)}

    def _announce_flagged(self):
        for outcome, flagged_names in self._state.flagged.items():
            print(f"Series: {len(flagged_names)} of {len(self.frames)} frames, {outcome}.")


    def align(self):
        """The final pass: every pair of frames up to three apart by its own lattice-safe search,
        solved by least squares (quantem.gpu.drift.drift), which outvotes a pair that took a wrong
        lattice image where the live pass, one frame against the last, cannot. Measured on the SSB
        phase; replaces shifts_A (specimen displacement against frame 0, Å), places every frame and
        rebuilds each combined image with it; returns the Drift with its pairs, closure and σ. Frames
        go to drift by index, so Drift.plot draws a numeric frame axis."""
        from quantem.gpu.drift import drift

        found = drift(torch.from_dlpack(self.phase), list(range(len(self.frames))), sampling_A=self.object_sampling_A)
        self._freeze()
        self.shifts_A, self.sigma_A, self.placed = found.displacement_A, found.sigma_A, (True,) * len(self.frames)
        for key in self.images:
            self.combined[key], self.cover = self._combine(key)
        return found

    def reconstruct(self, method, **strengths):
        """Reconstruct ``method`` again on every frame after the series, at its refitted optics.

        For trying other strengths without refitting or registering: each source is read again (a resident
        scan is used in place, so it must still be held, not released) with
        the series' calibration and upsampling, ``method`` runs at ``fits[k]`` with ``strengths`` (they
        apply to the regularized methods only), and the result is placed at ``shifts_A`` (the live
        shifts, or align()'s after align). Replaces
        ``images`` and ``combined`` for the method (and ``"amplitude"`` for ``"complex"``) and returns
        those combined images by name."""
        session = self.session
        self._freeze()
        stacks = {}
        with cp.cuda.Device(int(session["device"].split(":")[1])):
            for index, (source, chosen) in enumerate(zip(self.sources, self.fits, strict=True)):
                resident = not isinstance(source, Path)
                data = source if resident else io.load(source, device=session["device"], verbose=False)
                frame = session["kind"](data, **session["settings"])
                reconstructed, _, _ = _reconstruct(frame, chosen, (method,), session["upsampling"], strengths)
                for key, image in reconstructed.items():
                    if index == 0:
                        stacks[key] = cp.empty((len(self.frames), *image.shape), dtype=cp.float32)
                    stacks[key][index] = image
                frame.close()
                if not resident:
                    data.close()
                del data, frame, reconstructed
                cp.get_default_memory_pool().free_all_blocks()
            self.images.update(stacks)
            # phase is the SSB stack registration and align read; it stays the array images["ssb"] is
            self.phase = stacks.get("ssb", self.phase)
            if method not in self.methods:
                self.methods = (*self.methods, method)
            for key in stacks:
                self.combined[key], self.cover = self._combine(key)
        return {key: self.combined[key] for key in stacks}

    def _view(self, keys):
        """The interactive view's panels: the latest scan's first image, labelled with its acquisition and fitted
        defocus and astigmatism, then each combined image; Show2D takes finite pixels only, so pixels no frame
        reaches hold the median of the covered pixels, a flat field."""
        covered = self.cover > 0
        latest, optics = self.frames[-1], self.aberrations[-1]
        limits = [self.reference.phase_limits, *(_limits(self.combined[key], covered) if key == "amplitude"
                                                 else self.reference.phase_limits for key in keys)]
        images = [host_array(self.images[keys[0]][-1]), *(host_array(cp.where(covered, self.combined[key],
                  cp.median(self.combined[key][covered]))) for key in keys)]
        labels = [f"{latest}: C10 {optics['C10_nm']:.2f} nm, C12 {optics['C12_nm']:.2f} nm, "
                  f"phi12 {optics['phi12_deg']:.1f} deg", *(f"{_NAMES[key]}, combined" for key in keys)]
        return images, labels, f"{len(self.frames)} scans, {sum(self.placed)} combined", limits

    def _redraw(self):
        """Refresh every open interactive view with the scan just added."""
        for view, keys in self._state.views:
            images, labels, title, _ = self._view(keys)
            view.set_image(images, labels=labels)
            view.title = title

    def _freeze(self):
        """align and reconstruct replace stacks and shifts that adding a frame extends; no frame is added after."""
        if self._state is not None:
            self._state.frozen = True

    def _combine(self, key):
        """images[key] in register and its cover: frame 0 at the origin, every placed frame at its
        shift (shifts_A over the sampling, px), the frames reaching each pixel averaged."""
        stack = self.images[key]
        canvas = _start(stack[0])
        for image, shift, placed in zip(stack[1:], self.shifts_A[1:], self.placed[1:]):
            if placed:
                _add(canvas, image, tuple(shift / np.array(self.object_sampling_A)))
        return _combined(canvas), canvas.cover

    def report(self, *, to_dict: bool = False):
        """One row per frame: fitted C10, C12, phi12, refit evaluations, whether the frame came from
        its saved record (its evaluations are then the original refit's), the shift against frame 0,
        the live pass's pair confidence (below CONFIDENCE: measured, not added), and seconds per stage
        and per method. Before align, Near threshold marks a confidence within 10% above CONFIDENCE
        (placed, worth a look); after align the shifts are the pairwise solve's, so its σ per frame
        replaces that flag and the live confidence stays only as a record of the live pass."""
        import pandas as pd

        rows = {}
        sigma = [None] * len(self.frames) if self.sigma_A is None else self.sigma_A
        for frame, optics, refit, clock, shift, score, spread in zip(self.frames, self.aberrations, self.refits,
                                                                     self.timings, self.shifts_A, self.confidence,
                                                                     sigma, strict=True):
            methods = {f"{_METHODS[name]} (s)": clock[f"{name}_s"] for name in _METHODS if f"{name}_s" in clock}
            placement = ({"Near threshold": bool(CONFIDENCE <= score < 1.1 * CONFIDENCE)} if spread is None
                         else {"σ row (Å)": spread[0], "σ col (Å)": spread[1]})
            rows[frame] = {
                "C10 (nm)": optics["C10_nm"], "C12 (nm)": optics["C12_nm"], "phi12 (deg)": optics["phi12_deg"],
                "Refit evaluations": refit["evaluations"], "Resumed": refit["resumed"],
                "Shift row (Å)": shift[0], "Shift col (Å)": shift[1], "Live confidence": score, **placement,
                "Load (s)": clock["load_s"], "Setup (s)": clock["setup_s"], "Refit (s)": clock["refit_s"],
                **methods, "Align (s)": clock["align_s"], "Display (s)": clock["display_s"], "Total (s)": clock["total_s"],
            }
        return rows if to_dict else pd.DataFrame.from_dict(rows, orient="index").rename_axis("Frame")

    def show(self, method=None, *, interactive: bool = False, size="medium", cmap: str = "gray"):
        """The combined image of each requested method (or of ``method``), with the drift.

        Phase images share the reference phase contrast; the amplitude gets its own, the 1st and 99th
        percentiles of its covered pixels. Pixels no frame reaches are no data: masked in a neutral
        blue-gray in the static figure; Show2D takes finite pixels only, so there they hold the
        median of the covered pixels, a flat field. Static: the images side by side with scale bars and
        colour bars over a strip of the displacement against frame 0 (row, column, Å) with C10 and
        C12 (nm) on a second axis. Interactive returns a Show2D of the latest scan (labelled with its
        acquisition and fitted C10, C12, phi12) and the combined images at full resolution, redrawn with
        every scan ``add`` brings; ``size`` is a preset name or a panel width in pixels for either.
        """
        keys = list(self.combined) if method is None else [method, *(["amplitude"] if method == "complex" else [])]
        if interactive:
            from quantem.widget import Show2D

            images, labels, title, limits = self._view(keys)
            width = {"size": size} if isinstance(size, str) else {"panel_width_px": size}
            view = Show2D(images, labels=labels, title=title, cmap=cmap, sampling=self.object_sampling_A[1], units="Å",
                          vmin=[low for low, _ in limits], vmax=[high for _, high in limits], link_contrast=False,
                          display_bin=1, verbose=False, **width)
            if self._state is not None:
                # A series still growing redraws this view with every added scan.
                self._state.views.append((view, keys))
            return view
        covered = self.cover > 0
        limits = [_limits(self.combined[key], covered) if key == "amplitude" else self.reference.phase_limits
                  for key in keys]
        import matplotlib.pyplot as plt

        from quantem.gpu.display._layout import panel_figure, panel_inches

        width = panel_inches(size)
        # The drift strip sits under the image panels, with room for both y axes and the x label.
        strip = .6 * width
        figure, axes = panel_figure(1, len(keys), width, width, title_lines=2, bottom_in=strip + 1.4)
        figure_width, figure_height = figure.get_size_inches()
        uncovered = host_array(~covered)
        for axis, key, (low, high) in zip(axes[0], keys, limits):
            _panel(figure, axis, host_array(self.combined[key]), f"{_NAMES[key]}\ncombined, {sum(self.placed)} frames"
                   f"{'' if key == 'amplitude' else ', rad'}", (low, high), self.object_sampling_A, width, cmap,
                   uncovered=uncovered)
        drift = figure.add_axes((.9 / figure_width, .6 / figure_height,
                                 1 - 1.9 / figure_width, strip / figure_height))
        index = np.arange(len(self.frames))
        drift.plot(index, self.shifts_A[:, 0], "o-", ms=3, color="C0", label="row (Å)")
        drift.plot(index, self.shifts_A[:, 1], "s-", ms=3, color="C1", label="column (Å)")
        drift.set(xlabel="Frame index", ylabel="Displacement against frame 0 (Å)",
                  title="Drift, defocus and astigmatism\nheld: orders 2-5, tilt, depth")
        optics = drift.twinx()
        optics.plot(index, [values["C10_nm"] for values in self.aberrations], "^--", ms=3, color="C2", label="C10 (nm)")
        optics.plot(index, [values["C12_nm"] for values in self.aberrations], "v--", ms=3, color="C3", label="C12 (nm)")
        optics.set_ylabel("nm")
        drift.legend(handles=drift.get_lines() + optics.get_lines(), fontsize=8)
        plt.close(figure)
        return figure


# Each method and image as a microscopist names it: report columns, figure titles, viewer labels.
_METHODS = {"ssb": "SSB", "phase": "Phase-only", "complex": "Amplitude + phase"}
_NAMES = {"ssb": "SSB", "phase": "Phase-only (phase)", "complex": "Amplitude + phase (phase)",
          "amplitude": "Amplitude + phase (amplitude)"}
# Pixels no frame reaches: a colour no gray value takes.
_UNCOVERED = "#b9c7d6"


def _keys(methods):
    """The images a series keeps: each method's phase, then the complex method's amplitude."""
    return [key for name in methods for key in ((name, "amplitude") if name == "complex" else (name,))]


def _reconstruct(session, chosen, methods, upsampling, strengths):
    """Run each method once on one frame at ``chosen``: the images by name (phase in rad, and the
    complex method's amplitude), the object sampling (Å per pixel) and seconds per method. SSB takes
    no strengths; force rebuilds a regularized method's prepared equations for this frame's counts."""
    images, seconds, sampling = {}, {}, None
    for name in methods:
        began = time.perf_counter()
        result = session.reconstruct(chosen, method=name, upsampling=upsampling, compute_loss=False,
                             force=True, verbose=False, **({} if name == "ssb" else strengths))
        images[name] = cp.asarray(result.phase, dtype=cp.float32)
        if name == "complex":
            images["amplitude"] = cp.asarray(result.amplitude, dtype=cp.float32)
        sampling = sampling or result.object_sampling_A
        cp.cuda.Device().synchronize()
        seconds[f"{name}_s"] = time.perf_counter() - began
    return images, sampling, seconds


def _limits(image, covered):
    """Display limits of an amplitude: its 1st and 99th percentiles over the pixels a frame reaches
    (the empty canvas margin is zero and would set the lower limit)."""
    low, high = cp.percentile(image[covered], (1, 99)).get()
    return float(low), float(high)


class _LivePanel:
    """The series' live view, only in a notebook kernel: the latest frame and the combined image of
    the first requested method and the drift so far in one figure, redrawn at most twice a second (as
    FitDisplay.progress) so drawing stays a small part of each frame's time."""

    def __init__(self, phase_limits, method, *, enabled):
        self.phase_limits, self.method, self.display, self.handle, self.last_update = phase_limits, method, None, None, 0.0
        if not enabled:
            return
        from IPython import get_ipython
        from IPython.display import display

        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            self.display = display

    def update(self, frame, phase, canvas, shifts_A, sampling_A, placed, *, final):
        if self.display is None or (not final and time.perf_counter() - self.last_update < 0.5):
            return
        import io

        import matplotlib.pyplot as plt
        from IPython.display import Image

        figure = _live_figure(frame, self.method, phase, canvas, shifts_A, sampling_A, placed, self.phase_limits)
        # One PNG render: the inline backend's tight bounding box draws the figure twice.
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        plt.close(figure)
        if self.handle is None:
            self.handle = self.display(Image(data=buffer.getvalue()), display_id=True)
        else:
            self.handle.update(Image(data=buffer.getvalue()))
        self.last_update = time.perf_counter()


def _live_figure(frame, method, phase, canvas, shifts_A, sampling_A, placed, phase_limits, size="medium"):
    """The latest frame and the combined image of the placed frames under the reference contrast, with
    scale bars, beside each frame's displacement against frame 0 (row and column, Å)."""
    from quantem.gpu.display._layout import panel_figure, panel_inches

    width = panel_inches(size)
    figure, axes = panel_figure(1, 3, width, width, title_lines=2, bottom_in=.6)
    titles = [f"{_NAMES[method]}\n{frame}, rad", f"{_NAMES[method]}\ncombined, {placed} of {len(shifts_A)} frames, rad"]
    if canvas is None:
        # A live run's first frame, shown while its rows arrive: nothing is combined yet.
        _panel(figure, axes[0, 0], host_array(phase), titles[0], phase_limits, sampling_A, width, "gray")
        axes[0, 1].set_axis_off()
        axes[0, 1].set_title("combined image\nafter the first frame", fontsize=min(14, max(8, width * 2.2)))
    else:
        images, uncovered = (host_array(phase), host_array(_combined(canvas))), (None, host_array(canvas.cover == 0))
        for axis, image, title, empty in zip(axes[0, :2], images, titles, uncovered):
            _panel(figure, axis, image, title, phase_limits, sampling_A, width, "gray", uncovered=empty)
    trace = axes[0, 2]
    trace.plot(shifts_A[:, 0], "o-", ms=3, label="row")
    trace.plot(shifts_A[:, 1], "s-", ms=3, label="column")
    trace.set(xlabel="Frame index", title="Displacement against\nframe 0 (Å)")
    # Ticks on the right, clear of the combined image's colorbar in the gap before it.
    trace.yaxis.tick_right()
    trace.legend(fontsize=8, frameon=False)
    return figure


def _panel(figure, axis, image, title, limits, sampling_A, width, cmap, *, uncovered=None):
    """One image with its colour bar and scale bar, pixels in uncovered (no frame reaches them) masked
    in _UNCOVERED. Plain imshow at full resolution: show_2d builds a float64 RGBA copy of each image
    first, 3.5 s for two of these, longer than a frame."""
    import matplotlib.pyplot as plt
    from quantem.core.visualization.visualization_utils import add_scalebar_to_ax

    from quantem.gpu.display._layout import scalebar

    # Antialias on the values, not on colours: Matplotlib's automatic choice smooths the RGBA image
    # when it shrinks this much, 3 to 4 s a figure in a notebook against 0.2 s.
    image = image if uncovered is None else np.ma.masked_array(image, uncovered)
    shown = axis.imshow(image, cmap=plt.get_cmap(cmap).with_extremes(bad=_UNCOVERED), vmin=limits[0], vmax=limits[1],
                        interpolation_stage="data")
    figure.colorbar(shown, ax=axis, fraction=.046, pad=.02)
    bar = scalebar(sampling_A[1], "Å", image.shape[0], width)
    add_scalebar_to_ax(axis, image.shape[1], bar["sampling"], None, "Å", bar["width_px"], 0.5, "white",
                       "lower right", bar["fontsize"])
    axis.set(xticks=[], yticks=[])
    axis.set_title(title, fontsize=min(14, max(8, width * 2.2)))


def _combined(canvas):
    """The frames in register averaged where they reach: total over cover, zero where none does."""
    return (canvas.total / cp.maximum(canvas.cover, 1)).astype(cp.float32)


def _recorded(path, identity):
    """A frame's saved record when it was made from this very frame and fit, else None.

    It must carry RECORD_SCHEMA and an identity equal to this call's: the frame's position in the
    series, the SHA-256 of the previous frame's record, the frame's acquisition_identity (its file
    name, size and the SHA-256 of its first MiB and of each linked data file, and how flagged pixels
    were held), the reference fit's coordinates,
    bounds and saved signature, the calibration and upsampling, and the SSB source hash. The record
    then holds the exact float64 optics and shift the uninterrupted run would find again. A file cut
    short (a crash while it was copied, a full disk) is no record: the frame is computed again."""
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if saved.get("schema") != RECORD_SCHEMA or saved.get("identity") != json.loads(json.dumps(json_value(identity))):
        return None
    return saved


def _refit(objective, reference, start, bounds):
    """The exact-gradient refinement over (C10, C12 components) with the other 25 coordinates held at ``reference``."""
    z = np.array(reference, dtype=float)
    z[VARIED] = np.clip(start, bounds[:, 0], bounds[:, 1])
    box = np.column_stack([z, z])
    box[VARIED] = bounds
    record = _refine_with_gradients(objective, z, box, float(objective.values(z, 1)[0]), EVALUATION_CAP)
    return np.asarray(record["coordinates"]), record["objectives"], record["stop"]


def _lattice(phase, sampling_A, band_A=(50.0, 0.67)):
    """What the live registration balances, from the first frame alone: each lattice's Bragg spots
    (spots) and every other spot to cut from the aperiodic part (peaks: moiré and its harmonics), in
    Å⁻¹, so the bands can be drawn on any crop (bands); drift finds them on the series' mean power
    spectrum, which a series does not have yet."""
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
    1 Å a frame on the measured series), not its speed (up to 5 Å a frame)."""
    if len(placed) < 2:
        return placed[-1][1]
    (before, earlier), (after, later) = placed[-2], placed[-1]
    return tuple(late + (late - early) * (index - after) / (after - before) for early, late in zip(earlier, later))


def _place(canvas, phase, lattice, sampling_A, reach_A, guess_px):
    """Register one frame against the frame added last and, when confident, add it in register.

    The frame is expected at n = round(guess_px) px from frame 0 and the last one sits at m. Both are
    cut to the rectangle they share there, so each crop holds content everywhere as the Hann-tapered
    correlation assumes, and the residual d - n between the crops is small, well inside reach_A where
    the whole drift (107 Å over the measured series' 40 frames) would not be. register_spectra with the first
    frame's bands gives it with moving(r) = reference(r - residual), so the frame's shift against
    frame 0 is d = n + residual, px. The reference is the last frame, not the sum of every earlier
    one: on the measured series the specimen changes along the series, and even with every earlier frame placed at
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
    masks = bands(spectra[0].abs(), sampling_A, lattice.layers, lattice.band_A, notch=lattice.notch)
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
