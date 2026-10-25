"""How far the specimen moved between the frames of a time series, from each frame's phase image, and the
series moved back into register.

    found = drift(phases, sampling_A=0.124)    # or drift(stream), drift([result, ...]): the SSB stream's own phases
    found.show(phases)                         # the path, and the mean before and after correction
    aligned, mean = found.corrected(phases)    # every frame moved onto frame 0, over the region all frames share

Convention: displacement_A[f] = (row, col) Å is where frame f sees the specimen against frame 0, so frame f's image
is frame 0's moved by it, I_f(r) = I_0(r - d_f), and corrected reads I_f(r + d_f). The drift measured on one
reconstruction applies to any other of the same frames (SSB, phase-only and amplitude + phase agree on the time series to
0.05 Å, docs/performance/ssb/reports/2026-10-24-drift-method-comparison.md).

Both registrations return one Drift. Which one:

- drift: every pair of frames up to neighbours apart, each by its own search (register_spectra), solved by least
  squares; for frames that each image the specimen well (the bilayer at full dose). The redundancy gives each pair's
  residual, the closure of every triple and each frame's σ.
- running: each frame against the sum of the frames before it in register, then against all the others; for frames
  too sparse to image alone (Ducky at 1 % of the dose a frame), where one frame against another is noise-limited
  (~0.6 px a pair, ~1 px triple closure) and a reference holding every earlier frame's dose reaches 0.22 px (×8;
  0.79 px at ×1). It has no pairs, so no closure: check it by blocks of frames summed in register.

drift keeps the pairs that agree with the most confident ones (consistent) and solves d_f by least squares with
d_0 = 0: the redundancy gives each pair's residual, the closure of every triple (d_ij + d_jk - d_ik, three independent
searches) and each frame's uncertainty. Closure cannot catch a frame whose two pairs both took the same lattice image
of one layer (the errors cancel over its triple); the confidence of those pairs, and an independent image, can."""

from dataclasses import dataclass, field

import numpy as np
import torch

from quantem.gpu.drift.registration import bands, peaks, prepared, register_spectra, spots


@dataclass
class Drift:
    """A series' drift: displacement_A[f] = (row, col) Å where frame f sees the specimen against
    frame 0 (the images' pixels times sampling_A); sigma_A its one-sigma uncertainty from the
    least-squares residuals; pairs every measured pair (frames, shift_A, residual_A, parts, margin,
    confidence, kept); closure_A each consecutive triple's d_ij + d_jk - d_ik from three independent
    searches, all three kept; layers the Bragg spots (Å⁻¹) of each lattice drift found.
    A running registration (running) has no pairs: sigma_A is nan and closure_A empty."""

    frames: list
    displacement_A: np.ndarray
    sigma_A: np.ndarray
    pairs: list = field(repr=False)
    closure_A: np.ndarray = field(repr=False)
    pair_rms_A: float = 0.0
    sampling_A: tuple = (1.0, 1.0)
    layers: list = field(default_factory=list, repr=False)

    def plot(self, others=None):
        """The drift path, row and column against frame with each σ, beside others (name: (frames,
        2) Å, e.g. the ADF drift or an earlier one) dashed, and each frame's difference from them;
        without others, the path alone."""
        import matplotlib.pyplot as plt

        frames = self.frames
        figure, axes = plt.subplots(1, 2 if others else 1, figsize=(10 if others else 6, 3.6), layout="constrained", squeeze=False)
        axes = axes[0]
        for axis, label, colour in ((0, "row", "#1f6fb4"), (1, "column", "#d9661f")):
            axes[0].errorbar(frames, self.displacement_A[:, axis], yerr=self.sigma_A[:, axis], fmt="o-", ms=3, color=colour, label=f"{label}, SSB")
        styles = ["--", ":", "-."]
        for (name, values), style in zip((others or {}).items(), styles):
            values = np.asarray(values)
            for axis, colour in ((0, "#1f6fb4"), (1, "#d9661f")):
                axes[0].plot(frames, values[:, axis], style, color=colour, alpha=0.7, label=f"{'row' if axis == 0 else 'column'}, {name}")
            axes[1].plot(frames, np.linalg.norm(values - self.displacement_A, axis=1), "o" + style, ms=3, label=name)
        axes[0].set(xlabel="frame", ylabel="displacement (Å)", title="Where each frame saw the specimen")
        if others:
            axes[1].set(xlabel="frame", ylabel="|difference| from SSB (Å)", title="Other measures against it", yscale="symlog", ylim=(0, None))
            axes[1].axhline(3.16, color="#999999", lw=0.8)
            axes[1].text(frames[0], 3.4, "one lattice vector", fontsize=8, color="#666666")
        for axis in axes:
            axis.legend(frameon=False, fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)
        plt.close(figure)
        return figure

    def corrected(self, images, sampling_A=None):
        """Every frame of images moved back onto frame 0's specimen, cropped to the region every frame
        sees: aligned (frames, rows, cols) and their mean (rows, cols), float32 beside the data. images
        is what drift took (a stack, a stream, a list of SSB results), the same frames in the same
        order and with the same frame-0 origin, from any reconstruction: the drift measured on SSB
        places phase-only or amplitude + phase frames as well. sampling_A is the images' own Å per
        pixel, read from a stream or result, else the drift's: displacement_A is in Å, so frames at
        another pixel size move by as many of their own pixels. Frame f moves by -displacement_A[f]
        by the Fourier shift theorem, I_f(r + d_f), exact for a band-limited phase image and never
        resampled; what a frame moved in from beyond its edge wraps round, so the result keeps only
        the rows and columns inside every frame."""
        images, _, sampling_A = phases(images, sampling_A=sampling_A, default=self.sampling_A)
        shifts = self.displacement_A / np.asarray(sampling_A)
        rows, cols = common_region(shifts, images.shape[-2:])
        aligned = torch.stack([moved(image, -shift)[rows, cols] for image, shift in zip(images, shifts)])
        return aligned, aligned.mean(0)

    def show(self, images=None, sampling_A=None):
        """The drift at a glance: without images, its path (plot); with them (as corrected takes them),
        frame 0, the mean of every frame as recorded and as corrected, each over the region all frames
        share, gray with a scale bar, the two means' power spectra (a drift blurs the mean and its
        spots; corrected, the spots sharpen), and the path."""
        if images is None:
            return self.plot()
        import matplotlib.pyplot as plt

        images, _, sampling_A = phases(images, sampling_A=sampling_A, default=self.sampling_A)
        images = images.float()
        rows, cols = common_region(self.displacement_A / np.asarray(sampling_A), images.shape[-2:])
        aligned, mean = self.corrected(images, sampling_A)
        panels = [(images[0, rows, cols], f"frame {self.frames[0]}"), (images[:, rows, cols].mean(0), f"mean of {len(images)}, as recorded"),
                  (mean, f"mean of {len(images)}, drift corrected")]
        figure, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
        for axis, (image, title) in zip(axes[0], panels):
            image = image.cpu().numpy()
            low, high = np.percentile(image, [0.5, 99.5])
            axis.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="lanczos")
            scale_bar(axis, image.shape, sampling_A[1])
            axis.set_title(title)
            axis.set_axis_off()
        for axis, (image, title) in zip(axes[1], panels[1:]):
            power = log_power(image, sampling_A).cpu().numpy()
            axis.imshow(power, cmap="gray", vmin=np.percentile(power, 50), vmax=np.percentile(power, 99.95), interpolation="lanczos")
            axis.set_title(f"power spectrum to 1.5 Å⁻¹, {title.split(', ')[1]}")
            axis.set_axis_off()
        path = axes[1, 2]
        index = np.arange(len(self.frames))
        for axis, label, colour in ((0, "row", "#1f6fb4"), (1, "column", "#d9661f")):
            path.errorbar(index, self.displacement_A[:, axis], yerr=self.sigma_A[:, axis], fmt="o-", ms=3, color=colour, label=label)
        path.set(xlabel="frame index", ylabel="displacement against frame 0 (Å)", title="where each frame saw the specimen")
        path.legend(frameon=False, fontsize=8)
        path.spines[["top", "right"]].set_visible(False)
        plt.close(figure)
        return figure

    def _repr_markdown_(self):
        if self.pairs:
            header = (f"**Drift of {len(self.frames)} frames**, every pair up to {max(pair['j'] - pair['i'] for pair in self.pairs)} apart: pairs rms {self.pair_rms_A:.3f} Å, "
                      f"triple closure up to {np.abs(self.closure_A).max() if len(self.closure_A) else 0:.3f} Å, {sum(not pair['kept'] for pair in self.pairs)} of {len(self.pairs)} pairs rejected")
        elif not np.isnan(self.sigma_A).all():  # a subset of a pairwise drift: its pairs stayed with the whole series
            header = f"**Drift of {len(self.frames)} frames** of a pairwise drift: σ from every pair of the whole series, pairs rms {self.pair_rms_A:.3f} Å"
        else:
            header = f"**Drift of {len(self.frames)} frames**, each against the running sum of the others ({self.sampling_A[0]:.3f} Å a pixel): no pairs, so no σ or closure"
        lines = [header, "", "| frame | row (Å) | col (Å) | σ row | σ col |", "|---|---|---|---|---|"]
        lines += [f"| {frame} | {float(row):+.3f} | {float(col):+.3f} | {float(sigma_row):.3f} | {float(sigma_col):.3f} |".replace("nan", "-")
                  for frame, (row, col), (sigma_row, sigma_col) in zip(self.frames, self.displacement_A, self.sigma_A)]
        return "\n".join(lines)


def drift(images, frames=None, *, sampling_A=None, neighbours=3, band_A=(50.0, 0.67), reach_A=30.0, outlier_A=0.5):
    """Every frame's displacement against the first from images, one phase image per frame: a stack
    (frames, rows, cols) or list of them (torch, CuPy or NumPy, NumPy moved to the GPU), an SSB stream
    (its phase, frame names and sampling), or a list of SSB results. sampling_A (row, col) or one
    number, Å per pixel, defaults to the stream's or results' own, else 1 (displacements in pixels). Each pair up to neighbours apart is
    registered by its own search within reach_A (register_spectra); the pairs that agree within
    outlier_A with the more confident ones are kept (consistent) and d solved from them by least
    squares with d_0 = 0. band_A keeps periods between 50 Å (slower is the image's background) and
    0.67 Å (1.5 Å⁻¹; past it SSB's transfer is noise). The lattices' Bragg spots (spots) are found
    on the series' mean power spectrum."""
    images, frames, sampling_A = phases(images, frames, sampling_A)
    images = images.float()
    count = len(images)
    power = sum(torch.fft.fft2(prepared(images[index])).abs().square() for index in range(count)) / count
    layers = spots(power, sampling_A)
    masks = bands(power, sampling_A, layers, band_A, notch=peaks(power, sampling_A, band_A))
    spectra = {}  # each frame's spectrum while a pair still needs it: a long series at 3x does not fit at once
    measured = []
    for i in range(count):
        for index in [index for index in spectra if index < i]:
            del spectra[index]
        for index in range(i, min(count, i + neighbours + 1)):
            if index not in spectra:
                spectra[index] = torch.fft.fft2(prepared(images[index]))
        for j in range(i + 1, min(count, i + neighbours + 1)):
            found = register_spectra(spectra[i], spectra[j], masks, sampling_A, reach_A)
            measured.append({"i": i, "j": j, "frames": (frames[i], frames[j]), "shift_A": tuple(value * step for value, step in zip(found.shift_px, sampling_A)),
                          "parts": found.parts, "best": found.best, "margin": found.margin, "kept": True,
                          "confidence": min(found.margin, found.parts["coarse"] - found.best["coarse_second"]),
                          "runner_up_A": tuple(value * step for value, step in zip(found.runner_up_px, sampling_A))})
    consistent(measured, count, outlier_A)
    displacement, sigma, residual = solve(measured, count)
    kept = {(pair["i"], pair["j"]): np.array(pair["shift_A"]) for pair in measured if pair["kept"]}
    closure = []  # each consecutive triple whose three pairs were kept: d_ij + d_jk - d_ik
    for i in range(count - 2):
        if {(i, i + 1), (i + 1, i + 2), (i, i + 2)} <= kept.keys():
            closure.append(kept[i, i + 1] + kept[i + 1, i + 2] - kept[i, i + 2])
    closure = np.array(closure).reshape(-1, 2)
    return Drift(frames, displacement, sigma, measured, closure, residual, sampling_A, layers)


def running(images, frames=None, *, sampling_A=None, passes=2):
    """Every frame's displacement against the first from images (as drift takes them), for frames
    too sparse to register one against another: each frame against the sum of the frames before it
    moved into register, then passes rounds of each against all the others in register (their sum
    over their tapers' sum, under its own taper, so both carry one window and only the residual
    shift moves the peak), frame 0's shift taken out after each; every correlation to 1/100 px
    (cross_correlation_shift_batch_cp). The reference gathers the dose of all earlier frames, so a
    frame of 500 electrons a pattern is placed to a fifth of a pixel where a pair of them is placed
    to half of one."""
    import cupy

    images, frames, sampling_A = phases(images, frames, sampling_A)
    with cupy.cuda.Device(images.device.index or 0):
        shifts = running_shifts([cupy.from_dlpack(image.double().contiguous()) for image in images], passes)
        cupy.get_default_memory_pool().free_all_blocks()
    # running_shifts gives s with I_f(r) = I_0(r + s): the specimen against frame 0 is -s
    displacement = -np.asarray(shifts) * np.asarray(sampling_A)
    return Drift(frames, displacement, np.full_like(displacement, np.nan), [], np.zeros((0, 2)), float("nan"), tuple(sampling_A))


# ---


def phases(images, frames=None, sampling_A=None, default=(1.0, 1.0)):
    """The phase stack (frames, rows, cols) on the GPU, the frame labels and sampling_A (row, col) Å
    from any form drift takes: a stream or result carries its phase and object_sampling_A, a stream
    also its frame names, and default (1, 1: pixels) holds where neither it nor sampling_A says; a
    NumPy array moves to the GPU, the data never computed on the CPU."""
    if isinstance(images, (list, tuple)) and len(images) and hasattr(images[0], "phase"):
        sampling_A = images[0].object_sampling_A if sampling_A is None else sampling_A
        images = [result.phase for result in images]
    elif hasattr(images, "phase"):
        frames = list(images.frames) if frames is None else frames
        sampling_A = images.object_sampling_A if sampling_A is None else sampling_A
        images = images.phase
    images = torch.stack([gpu(image) for image in images]) if isinstance(images, (list, tuple)) else gpu(images)
    frames = list(range(len(images))) if frames is None else list(frames)
    sampling_A = tuple(float(value) for value in np.broadcast_to(default if sampling_A is None else sampling_A, 2))
    return images, frames, sampling_A


def gpu(array):
    """array as a torch tensor where it lives (torch or CuPy), a NumPy array moved to the GPU."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to("cuda" if torch.cuda.is_available() else "mps")
    return torch.as_tensor(array)


def common_region(shifts_px, shape):
    """The rows and columns of frame 0 every frame sees once moved back by shifts_px (frames, 2):
    frame f's pixel r + d_f exists for -d_f <= r < size - d_f, so the region runs from the largest
    -d_f to the smallest size - d_f, rounded inward to whole pixels."""
    first = np.ceil(np.maximum(0.0, -shifts_px.min(0))).astype(int)
    last = np.floor(np.asarray(shape) - np.maximum(0.0, shifts_px.max(0))).astype(int)
    return slice(first[0], last[0]), slice(first[1], last[1])


def moved(image, shift_px):
    """image (rows, cols) moved by shift_px (row, col) by the Fourier shift theorem, image(r - shift),
    in float64 so a 40-frame correction adds no rounding of its own, returned float32."""
    rows, cols = image.shape
    q_row = torch.fft.fftfreq(rows, device=image.device, dtype=torch.float64)[:, None]
    q_col = torch.fft.fftfreq(cols, device=image.device, dtype=torch.float64)[None, :]
    ramp = torch.exp(-2j * torch.pi * (q_row * float(shift_px[0]) + q_col * float(shift_px[1])))
    return torch.fft.ifft2(torch.fft.fft2(image.double()) * ramp).real.float()


def log_power(image, sampling_A, q_max_A=1.5):
    """log10 of image's power spectrum to q_max_A Å⁻¹ (the registration's band edge), zero frequency
    centred: Hann-tapered and mean removed, then zero-padded to the same extent in Å along rows and
    columns, so both axes share one Å⁻¹ step and a rectangular region, or unequal sampling, still
    gives round spots."""
    extent_A = max(size * step for size, step in zip(image.shape, sampling_A))
    padded_shape = [int(np.ceil(extent_A / step)) for step in sampling_A]
    padded = torch.nn.functional.pad(prepared(image.float()), (0, padded_shape[1] - image.shape[1], 0, padded_shape[0] - image.shape[0]))
    power = torch.fft.fftshift(torch.fft.fft2(padded)).abs().square()
    window = []
    for size, step in zip(padded_shape, sampling_A):
        half = min(size // 2, round(q_max_A * size * step))
        window.append(slice(size // 2 - half, size // 2 + half + 1))
    return torch.log10(power[window[0], window[1]] + power.max() * 1e-9)


def scale_bar(axis, shape, sampling_A):
    """A white bar a fifth of the image wide, rounded to 1, 2 or 5 nm (or Å below 1 nm), at the lower left."""
    length = shape[1] * sampling_A / 5
    nice = [step * 10.0**power for power in range(-1, 4) for step in (1, 2, 5)]
    length = max(value for value in nice if value <= length)
    label = f"{length / 10:g} nm" if length >= 10 else f"{length:g} Å"
    row, col = shape[0] * 0.95, shape[1] * 0.04
    axis.plot([col, col + length / sampling_A], [row, row], color="white", lw=4, solid_capstyle="butt")
    axis.text(col, row - shape[0] * 0.025, label, color="white", fontsize=10)


def solve(pairs, count):
    """Least squares d_j - d_i = shift_ij over the kept pairs, d_0 = 0: displacement (count, 2),
    sigma from the residuals' variance, and each pair's residual (kept or not) written into it."""
    kept = [pair for pair in pairs if pair["kept"]]
    design = np.zeros((len(kept), count))
    for row, pair in enumerate(kept):
        design[row, pair["j"]], design[row, pair["i"]] = 1.0, -1.0
    design = design[:, 1:]
    measured = np.array([pair["shift_A"] for pair in kept])
    solution, *_ = np.linalg.lstsq(design, measured, rcond=None)
    displacement = np.vstack([np.zeros(2), solution])
    errors = measured - design @ solution
    dof = max(1, len(kept) - (count - 1))
    variance = (errors**2).sum(0) / dof
    sigma = np.vstack([np.zeros(2), np.sqrt(np.outer(np.diag(np.linalg.pinv(design.T @ design)), variance))])
    for pair in pairs:
        pair["residual_A"] = tuple(float(value) for value in np.array(pair["shift_A"]) - (displacement[pair["j"]] - displacement[pair["i"]]))
    return displacement, sigma, float(np.sqrt((errors**2).mean()))


def consistent(pairs, count, outlier_A):
    """Keep the pairs that agree, most confident first: pairs sorted by confidence (the lesser of how
    clearly the coarse part chose its region and the fine parts their lattice image) join frames
    into groups with their shifts; a pair inside one group is kept
    only if it agrees with the shift its group already implies to within outlier_A. A wrong lattice
    image is a whole lattice vector (3.2 Å) or a moiré vector (11 Å) off, so a confident chain
    outvotes a weak pair that took one, however many weak pairs agree with each other."""
    root, offset = list(range(count)), [np.zeros(2) for _ in range(count)]

    def find(index):
        path = np.zeros(2)
        while root[index] != index:
            path = path + offset[index]
            index = root[index]
        return index, path

    for pair in sorted(pairs, key=lambda pair: -pair["confidence"]):
        (first, to_first), (second, to_second) = find(pair["i"]), find(pair["j"])
        shift = np.array(pair["shift_A"])
        if first == second:
            pair["kept"] = bool(np.hypot(*(to_second - to_first - shift)) <= outlier_A)
        else:
            root[second], offset[second] = first, to_first + shift - to_second
            pair["kept"] = True


def running_shifts(images, passes=2):
    """running on CuPy images: each image's shift s against the first, pixels, I_f(r) = I_0(r + s). CuPy, as
    cross_correlation_shift_batch_cp takes it."""
    import cupy

    from quantem.gpu.parallax.cuda.correlation import cross_correlation_shift_batch_cp

    padded = [footprint(image) for image in images]
    reference, shifts = padded[0], []
    for index, image in enumerate(padded):
        shift = cross_correlation_shift_batch_cp(reference, image[None], upsample_factor=100)[0].get()
        if index:
            reference = reference + shifted(image, shift)
        shifts.append(shift)
    window = footprint(cupy.ones(images[0].shape), centred=False)
    estimate = cupy.asnumpy(cupy.asarray(shifts))
    for _ in range(passes):
        placed = [shifted(image, shift) for image, shift in zip(padded, estimate)]
        tapers = [shifted(window, shift) for shift in estimate]
        total, cover = sum(placed), sum(tapers)
        for index in range(len(padded)):
            others = (total - placed[index]) / cupy.maximum(cover - tapers[index], 0.5)
            estimate[index] += cross_correlation_shift_batch_cp(others * tapers[index], placed[index][None], upsample_factor=100)[0].get()
        estimate -= estimate[0]
    return estimate.tolist()


def footprint(image, centred=True):
    """The image less its mean under a Tukey window over half its width, zero-padded to twice its
    size, so a correlation of two is the linear one for any drift under half the field."""
    import cupy
    from cupyx.scipy.signal.windows import tukey

    rows, cols = image.shape
    padded = cupy.zeros((2 * rows, 2 * cols))
    padded[:rows, :cols] = (image - image.mean() * centred) * (tukey(rows, 0.5)[:, None] * tukey(cols, 0.5)[None, :])
    return padded


def shifted(image, shift):
    """An image moved by shift (rows, cols) pixels by the Fourier shift theorem, image(r - shift): running moves
    each frame into register before it joins the sum, exactly for a derived phase image, never resampled."""
    import cupy

    rows, cols = image.shape
    ramp = cupy.fft.fftfreq(rows)[:, None] * shift[0] + cupy.fft.fftfreq(cols)[None, :] * shift[1]
    return cupy.fft.ifft2(cupy.fft.fft2(image) * cupy.exp(-2j * cupy.pi * ramp)).real
