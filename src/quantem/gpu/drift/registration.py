"""Where one image of a specimen sees another's: the shift between two phase images of a crystal, to 1/100 px.

Frame f sees the specimen moved by d_f: its image is frame 0's, I_f(r) = I_0(r - d_f), so the correlation of the two
peaks at d_f, and a matrix DFT around the peak finds it to 1/100 px (Guizar-Sicairos, Thurman and Fienup, Opt. Lett.
33, 156, 2008).

A crystal repeats: the correlation peaks at d_f plus every lattice vector, and noise or a change of the specimen can
raise a wrong lattice image above the true one (classic phase correlation, every frequency whitened, takes a wrong one
on a bilayer with one strong layer). A twisted bilayer has two lattices, and the true shift is where both layers and
the aperiodic rest (defects, edges, contamination) agree. So the cross-power spectrum is split by spatial frequency,
coarse to fine. The aperiodic part, every Bragg and moiré spot cut out, at periods longer than 3 Å, chooses the region:
it cannot take a lattice image, and the moiré must go too, since it repeats every ~11 Å where both layers nearly
repeat. Within 5 Å of it each layer's Bragg spots (first and second order) and the aperiodic part each give a
correlation divided by its own Σ|cross| (a perfect match scores 1), summed with equal weight: for two layers 15.6°
apart the layers alone score 0.7 again only 2.9 Å from the truth (the layers' mean cos 2π g·t, first reaching 0.7
again at 2.885 Å). The plain correlation over the whole band (the least-squares estimate for white noise) then gives
the position to 1/100 px. Each pair's confidence is the lesser of the coarse part's lead over its next region and the
fine parts' lead over the next lattice image."""

from types import SimpleNamespace

import numpy as np
import torch


def register(reference, moving, *, sampling_A=(1.0, 1.0), band_A=(50.0, 0.67), reach_A=30.0, layers=None):
    """Where moving sees reference's specimen: the shift (row, col) px with moving(r) = reference(r - shift),
    by the part-balanced phase correlation of the module docstring."""
    images = torch.stack([torch.as_tensor(reference, dtype=torch.float32), torch.as_tensor(moving, dtype=torch.float32)])
    spectra = torch.fft.fft2(prepared(images))
    power = spectra.abs().square().mean(0)
    if layers is None:
        layers = spots(power, sampling_A)
    masks = bands(power, sampling_A, layers, band_A, notch=peaks(power, sampling_A, band_A))
    return register_spectra(spectra[0], spectra[1], masks, sampling_A, reach_A)


# ---


def prepared(images):
    """Each image less its mean, tapered to zero at its edges by a Hann window, so the transform's
    wrap-around adds no edge of its own to the correlation."""
    return (images - images.mean((-2, -1), keepdim=True)) * hann(images)


def spots(power, sampling_A, *, q_range=(0.25, 0.45), contrast=8.0, count=12):
    """The first-order Bragg spots of each lattice in power (rows, cols), the mean power spectrum
    of the frames, grouped into lattices: [(spots, 2) Å⁻¹ (row, col), ...]. A spot is a local
    maximum standing contrast times above its neighbourhood's mean within q_range Å⁻¹ (MoS2's
    (100) at 0.365 Å⁻¹); the count strongest on the strongest one's |g| shell are kept and split
    into hexagonal lattices by their angle modulo 60°: one cluster is one lattice, two lattices 15°
    apart form two. A lattice shows two of its three Friedel pairs at least, so fewer than four
    spots are none. One shell and one angle split: a heterobilayer whose lattices differ by 10 % or
    more loses its second, and two lattices under 10° apart read as one."""
    q_row, q_col = frequencies(power, sampling_A)
    q = torch.sqrt(q_row**2 + q_col**2)
    shifted = torch.fft.fftshift(torch.log(power + power.mean() * 1e-6))
    q = torch.fft.fftshift(q)
    found = standing(shifted, contrast) & (q > q_range[0]) & (q < q_range[1])
    rows, cols = torch.nonzero(found, as_tuple=True)
    order = torch.argsort(shifted[rows, cols], descending=True)
    rows, cols = rows[order], cols[order]
    centre = torch.tensor(power.shape, device=power.device) // 2
    g = torch.stack([(rows - centre[0]) / (power.shape[0] * sampling_A[0]), (cols - centre[1]) / (power.shape[1] * sampling_A[1])], 1).double().cpu().numpy()
    if len(g) == 0:
        return []
    size = np.linalg.norm(g, axis=1)
    g = g[np.abs(size - size[0]) < 0.1 * size[0]][:count]  # the strongest spot's |g| shell, as wide as the scan's distortion stretches it (time series: 0.339 to 0.363 Å⁻¹): noise maxima elsewhere are no lattice's, nor take a weak layer's places
    if len(g) < 4:
        return []
    angle = np.degrees(np.arctan2(g[:, 0], g[:, 1])) % 60.0
    # two clusters on the 60° circle: split at the two largest gaps between sorted angles; one lattice's largest gap is the wrap-around between its ends
    ordered = np.sort(angle)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 60.0]))
    if np.sort(gaps)[-2] < 10.0:
        return [g]
    cuts = np.argsort(gaps)[-2:]
    bounds = sorted(ordered[(cuts + 1) % len(ordered)])
    first = (angle >= bounds[0]) & (angle < bounds[1])
    return [lattice for lattice in (g[first], g[~first]) if len(lattice) >= 4]


def peaks(power, sampling_A, band_A, *, contrast=6.0, q_min=0.05):
    """Every spot of power (rows, cols), unshifted, within the band and past q_min Å⁻¹: local
    maxima standing contrast times above their neighbourhood's mean. Beside the layers' Bragg
    spots these are the moiré (the layers' difference frequencies, 0.1 Å⁻¹ for 15.6°) and its
    harmonics: periodic content whose correlation repeats every moiré vector, where both layers
    nearly repeat too, so it must not stand in for the aperiodic part."""
    q_row, q_col = frequencies(power, sampling_A)
    q = torch.sqrt(q_row**2 + q_col**2)
    found = standing(torch.log(power + power.mean() * 1e-6), contrast) & (q > max(q_min, 1 / band_A[0])) & (q < 1 / band_A[1])
    rows, cols = torch.nonzero(found, as_tuple=True)
    return torch.stack([q_row[rows, 0], q_col[0, cols]], 1).double().cpu().numpy()


def standing(log_power, contrast):
    """Where log_power (rows, cols) is the maximum of its 7 x 7 neighbourhood and stands more than contrast times above
    the mean of its 31 x 31 one (in log, log contrast above): a spot of a spectrum, not its background."""
    peak = torch.nn.functional.max_pool2d(log_power[None, None], 7, 1, 3)[0, 0]
    local = torch.nn.functional.avg_pool2d(log_power[None, None], 31, 1, 15, count_include_pad=False)[0, 0]
    return (log_power == peak) & (log_power - local > np.log(contrast))


def bands(power, sampling_A, layers, band_A, *, notch=(), radius=4.0):
    """One mask per band of the spectrum of power (rows, cols) the registration balances: each lattice's Bragg spots (disks
    of radius pixels around each spot and its second order), then the aperiodic rest of the band
    with every other spot (notch: moiré and its harmonics) cut out."""
    shape = power.shape
    q_row, q_col = frequencies(power, sampling_A)
    q = torch.sqrt(q_row**2 + q_col**2)
    in_band = (q > 1 / band_A[0]) & (q < 1 / band_A[1])
    reach = radius * max(1 / (shape[0] * sampling_A[0]), 1 / (shape[1] * sampling_A[1]))
    masks, taken = [], torch.zeros(shape, dtype=torch.bool, device=power.device)
    for lattice in layers:
        mask = disks(q_row, q_col, harmonics(lattice), reach, sampling_A)
        masks.append(mask & in_band)
        taken |= mask
    taken |= disks(q_row, q_col, notch, reach, sampling_A)
    masks.append(in_band & ~taken)
    return masks


def disks(q_row, q_col, centres, reach, sampling_A):
    """Where (q_row, q_col) lies within reach Å⁻¹ of any of centres (spots, 2) Å⁻¹: every spot's disk
    at once, each tested only on the window of pixels around it, (q - g)² summed and compared exactly
    as for one spot over the whole grid, so the mask is the same bit for bit. One frame's spectrum
    has thousands of notch spots (2860 on a 1536² frame), which one disk at a time over the whole
    grid took 70 ms a frame; indices wrap, as the unshifted grid does at zero frequency."""
    rows, cols = q_row.shape[0], q_col.shape[1]
    mask = torch.zeros((rows, cols), dtype=torch.bool, device=q_row.device)
    if len(centres) == 0:
        return mask
    centres = torch.as_tensor(np.asarray(centres), device=q_row.device).to(q_row.dtype)
    windows = []
    for axis, size in enumerate((rows, cols)):
        step = 1 / (size * sampling_A[axis])
        half = int(np.ceil(reach / step)) + 1
        nearest = torch.round(centres[:, axis].double() / step).long()
        windows.append((nearest[:, None] + torch.arange(-half, half + 1, device=q_row.device)[None, :]) % size)
    row_index, col_index = windows
    inside = (q_row[row_index, 0][:, :, None] - centres[:, 0, None, None]) ** 2 + (q_col[0, col_index][:, None, :] - centres[:, 1, None, None]) ** 2 < reach**2
    mask[row_index[:, :, None].expand_as(inside)[inside], col_index[:, None, :].expand_as(inside)[inside]] = True
    return mask


def harmonics(lattice):
    """Each spot, its negative, and the sums of neighbouring first-order spots (the second order):
    all within the band carry the lattice's position."""
    first = np.concatenate([lattice, -lattice])
    second = [a + b for index, a in enumerate(first) for b in first[index + 1:] if np.linalg.norm(a + b) > 1.2 * np.linalg.norm(a)]
    return list(first) + second


def register_spectra(reference, moving, masks, sampling_A, reach_A, *, coarse_A=3.0, fine_A=5.0, upsample=100):
    """register on two prepared spectra, coarse to fine: shift_px (row, col) with moving(r) =
    reference(r - shift), parts each part's normalised correlation there (1 a perfect match, 0
    none), best each part's highest within reach, runner_up_px the best summed peak at least 1.5 Å
    away and margin how far the chosen peak stands above it, per part. Coarse: the aperiodic part
    (the last mask) at periods longer than coarse_A, the plain correlation's highest peak within reach_A; it cannot
    take a lattice or moiré image, those frequencies are cut out of it, and it says where to within
    an Å or two when the specimen's own features (edges, contamination, defects) persist between
    the frames. Fine: within fine_A of it every part's correlation, each divided by its own Σ|cross|
    so a perfect match scores 1, summed; the layers choose among lattice images 2.9 Å apart there.
    Exact: the plain correlation over the whole band near that peak, by a matrix DFT to 1/upsample px."""
    cross = moving * reference.conj()
    shape = cross.shape
    rows = torch.fft.fftfreq(shape[0], 1 / shape[0], device=cross.device)  # signed pixel shifts
    cols = torch.fft.fftfreq(shape[1], 1 / shape[1], device=cross.device)
    distance = lambda centre: ((rows[:, None] - centre[0]) * sampling_A[0]) ** 2 + ((cols[None, :] - centre[1]) * sampling_A[1]) ** 2
    q_row, q_col = frequencies(cross, sampling_A)
    slow = masks[-1] & (q_row**2 + q_col**2 < 1 / coarse_A**2)
    coarse_map = torch.fft.ifft2(cross * slow).real * (shape[0] * shape[1]) / (cross.abs() * slow).sum().clamp(min=1e-30)
    within = distance((0.0, 0.0)) < reach_A**2
    coarse_index = int(torch.argmax(masked(coarse_map, within)))
    coarse = (float(rows[coarse_index // shape[1]]), float(cols[coarse_index % shape[1]]))
    # the coarse runner-up: the best coarse peak at least 2 fine_A away, a second place the features could fit
    far = within & (distance(coarse) > (2 * fine_A) ** 2)
    coarse_second = float(masked(coarse_map, far).max())
    weights = [(cross.abs() * mask).sum().clamp(min=1e-30) for mask in masks]
    maps = [torch.fft.ifft2(cross * mask).real * (shape[0] * shape[1]) / weight for mask, weight in zip(masks, weights)]
    total = sum(maps)
    near = distance(coarse) < fine_A**2
    scored = masked(total, near)
    fine_index = int(torch.argmax(scored))
    peak = (float(rows[fine_index // shape[1]]), float(cols[fine_index % shape[1]]))
    # the fine runner-up: the best summed score at least 1.5 Å from the chosen peak within the window
    second = int(torch.argmax(masked(scored, distance(peak) > 1.5**2)))
    runner_up = (float(rows[second // shape[1]]), float(cols[second % shape[1]]))
    band = torch.stack(masks).any(0)
    shift = refine(cross * band, peak, upsample)
    names = [f"layer {index}" for index in range(len(masks) - 1)] + ["aperiodic"]
    values = {name: float(dft_at(cross * mask, shift) / weight) for name, mask, weight in zip(names, masks, weights)}
    values["coarse"] = float(coarse_map.flatten()[coarse_index])
    best = {name: float(masked(part, within).max()) for name, part in zip(names, maps)}
    best["coarse_second"] = coarse_second
    margin = float(total.flatten()[fine_index] - total.flatten()[second]) / len(masks)
    return SimpleNamespace(shift_px=shift, parts=values, best=best, runner_up_px=runner_up, margin=margin)


def masked(values, keep):
    """values where keep, -inf elsewhere: a maximum taken only over keep."""
    return torch.where(keep, values, torch.full_like(values, -torch.inf))


def dft_at(spectrum, shift):
    """Σ_q spectrum(q) exp(2πi q·shift), real part: the correlation at a sub-pixel shift (px), each part's score
    at the refined shift without an upsampled map of the whole correlation."""
    rows, cols = spectrum.shape
    k_row = torch.fft.fftfreq(rows, device=spectrum.device).double()
    k_col = torch.fft.fftfreq(cols, device=spectrum.device).double()
    phase_row = torch.exp(2j * torch.pi * k_row * shift[0])
    phase_col = torch.exp(2j * torch.pi * k_col * shift[1])
    return (phase_row @ spectrum.to(torch.complex128) @ phase_col).real


def refine(spectrum, peak, upsample, reach=1.5, coarse=10):
    """The correlation's maximum within reach px of peak, to 1/upsample px, found in two passes: every
    1/coarse px over the reach, then every 1/upsample px within 1/coarse px of the best of those. Each
    pass is the inverse transform of spectrum evaluated only on its grid, so the matrix products
    span 31 + 21 rows instead of the 301 of one 1/100 px grid over ±1.5 px (drift on the 40
    frames of 1536 x 1536: 0.91 s to 0.36 s). A true correlation peak is a pixel wide, so the coarse
    pass lands within 1/coarse px of the fine maximum and every kept pair is unchanged; only a
    flat correlation (a pair matched to a wrong region, which drift rejects) can stop on another
    local maximum. Offsets are counted in whole units of 1/upsample px, so the shift is exactly
    peak plus a multiple of 1/upsample."""
    span, step = round(reach * upsample), upsample // coarse
    units = torch.arange(-span, span + 1, step, device=spectrum.device)
    best = upsampled_max(spectrum, peak, upsample, units, units)
    best = upsampled_max(spectrum, peak, upsample, *(torch.arange(centre - step, centre + step + 1, device=spectrum.device) for centre in best))
    return (peak[0] + best[0] / upsample, peak[1] + best[1] / upsample)


def upsampled_max(spectrum, peak, upsample, row_units, col_units):
    """Where on the grid peak + (row_units, col_units) / upsample px the correlation of spectrum peaks,
    in whole units of 1/upsample px: the inverse transform evaluated only on that grid, as two matrix
    products (rows, then cols), divided by the Hann windows' own autocorrelation a(t_row) a(t_col):
    two images under one window correlate as a(t) R(t - d), whose peak a(t) pulls toward zero shift
    (0.01 px per 10 px of shift); the quotient peaks at d."""
    rows, cols = spectrum.shape
    k_row = torch.fft.fftfreq(rows, device=spectrum.device).double()
    k_col = torch.fft.fftfreq(cols, device=spectrum.device).double()
    row_kernel = torch.exp(2j * torch.pi * (peak[0] + row_units.double() / upsample)[:, None] * k_row[None, :])
    col_kernel = torch.exp(2j * torch.pi * k_col[:, None] * (peak[1] + col_units.double() / upsample)[None, :])
    fine = (row_kernel @ spectrum.to(torch.complex128) @ col_kernel).real
    power_row = torch.fft.fft(torch.hann_window(rows, periodic=False, device=spectrum.device).double()).abs().square().to(torch.complex128)
    power_col = torch.fft.fft(torch.hann_window(cols, periodic=False, device=spectrum.device).double()).abs().square().to(torch.complex128)
    fine = fine / ((row_kernel @ power_row).real[:, None] * (power_col @ col_kernel).real[None, :])
    index = int(fine.argmax())
    return int(row_units[index // len(col_units)]), int(col_units[index % len(col_units)])


def frequencies(grid, sampling=(1.0, 1.0)):
    """The spatial frequencies every Fourier-space formula here shares, so shifts and filters agree on one convention:
    q_row (rows, 1) and q_col (1, cols) of the last two axes of grid, unshifted, per unit of sampling, in grid's
    precision and beside its data."""
    rows, cols = grid.shape[-2:]
    q_row = torch.fft.fftfreq(rows, d=sampling[0], device=grid.device, dtype=grid.real.dtype)
    q_col = torch.fft.fftfreq(cols, d=sampling[1], device=grid.device, dtype=grid.real.dtype)
    return q_row[:, None], q_col[None, :]


def hann(grid):
    """The window a field is multiplied by before its spectrum is read, so its edges do not ring
    through every frequency: w(r, c) = h(r) h(c) over grid's last two axes, h the Hann window that
    is zero at both ends, in grid's precision and beside its data."""
    rows, cols = grid.shape[-2:]
    return (torch.hann_window(rows, periodic=False, dtype=grid.real.dtype, device=grid.device)[:, None]
            * torch.hann_window(cols, periodic=False, dtype=grid.real.dtype, device=grid.device))
