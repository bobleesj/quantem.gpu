"""The prepared MPS SSB state: bright-field selection, the G_qk spectra stack and its probe geometry.

``prepare_selection`` reads the selected detector columns once, transforms them to G(q, k) (Hermitian half-plane) and
caches the rotation-dependent probe geometry; every reconstruction, loss and fit then works from that state.
``retarget_prepared_rotation`` updates only the geometry when the scan rotation changes.
"""

import math
from dataclasses import dataclass

import numpy as np

from quantem.gpu.detector import fit_probe
from quantem.gpu.detector import mean as detector_mean
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.ssb.brightfield import BrightfieldDisk, disk_edge_radius
from quantem.gpu.ssb.mps.frames import (
    ArrayFrames,
    MpsBfColumnFrames,
    MpsTensorFrames,
    TensorChunkedFrames,
)
from quantem.gpu.ssb.mps.hardware import require_mlx


@dataclass
class PreparedMpsSSB:
    """Device-resident BF FFT stack and geometry for repeated SSB loss calls."""

    mx: object
    g_qk: object
    qx: object
    qy: object
    q_row: object
    q_col: object
    kx: object
    ky: object
    kx_np: np.ndarray
    ky_np: np.ndarray
    dc_value: complex
    scan_shape: tuple[int, int]
    wavelength: float
    semiangle_rad: float
    ang_y_rad: float
    ang_x_rad: float
    factor: float
    dc_mask: object
    num_bf: int
    alpha_k2: object | None
    cos2_k: object | None
    sin2_k: object | None
    aperture_k: object | None
    alpha_m2: object | None
    cos2_m: object | None
    sin2_m: object | None
    ap_m: object | None
    alpha_p2: object | None
    cos2_p: object | None
    sin2_p: object | None
    ap_p: object | None
    alpha_k2_1d: object | None = None
    cos2_k_1d: object | None = None
    sin2_k_1d: object | None = None
    aperture_k_1d: object | None = None
    bf_storage_indices_np: np.ndarray | None = None
    # thick-sample fit band (band limits, band arrays); it depends only on the q grid, so it survives rotation changes
    thick_band: tuple | None = None


def prepare_selection(
    frames,
    *,
    scan_shape: tuple[int, int],
    selection: BrightfieldDisk,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling: tuple[float, float],
    det_sampling: tuple[float, float],
    rotation_angle_deg: float,
    chunk_bf: int,
    compact_inactive: bool = False,
    dc_value_override: complex | None = None,
) -> PreparedMpsSSB:
    """Precompute BF-column FFTs and static geometry for MPS SSB fitting."""
    if scan_shape[0] != scan_shape[1]:
        raise ValueError(
            "MPS SSB requires a square scan grid; "
            f"got {scan_shape[0]}x{scan_shape[1]}."
        )
    mx = require_mlx()
    bf_row = selection.rows
    bf_col = selection.cols
    center = selection.center_row_col
    det_shape = selection.detector_shape
    wavelength = float(wavelength_A_from_kV(float(voltage_kV)))
    reciprocal_sampling = (
        det_sampling[0] * 1e-3 / wavelength,
        det_sampling[1] * 1e-3 / wavelength,
    )
    sampling = (
        1.0 / (reciprocal_sampling[0] * det_shape[0]),
        1.0 / (reciprocal_sampling[1] * det_shape[1]),
    )
    q_row_np, q_col_np = _spatial_frequencies(scan_shape, scan_sampling)

    recip_y = 1.0 / (sampling[0] * det_shape[0])
    recip_x = 1.0 / (sampling[1] * det_shape[1])
    kx_np = (bf_row.astype(np.float32) - center[0]) * recip_y
    ky_np = (bf_col.astype(np.float32) - center[1]) * recip_x
    if rotation_angle_deg:
        angle = math.radians(-float(rotation_angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        kx_np, ky_np = kx_np * cos_a + ky_np * sin_a, -kx_np * sin_a + ky_np * cos_a
    kx_np = np.asarray(kx_np, dtype=np.float32)
    ky_np = np.asarray(ky_np, dtype=np.float32)

    q_row_mx = mx.array(q_row_np, dtype=mx.float32)
    q_col_mx = mx.array(q_col_np, dtype=mx.float32)
    kx_mx = mx.array(kx_np, dtype=mx.float32)
    ky_mx = mx.array(ky_np, dtype=mx.float32)
    qx = q_row_mx[None, :, None]
    qy = q_col_mx[None, None, :]
    semiangle_rad = float(semiangle_mrad) * 1e-3
    ang_y_rad = float(det_sampling[0]) * 1e-3
    ang_x_rad = float(det_sampling[1]) * 1e-3
    alpha_k2_1d, cos2_k_1d, sin2_k_1d, aperture_k_1d = compute_geometry(
        mx,
        kx_mx,
        ky_mx,
        wavelength,
        semiangle_rad,
        ang_y_rad,
        ang_x_rad,
    )
    mx.eval(alpha_k2_1d, cos2_k_1d, sin2_k_1d, aperture_k_1d)
    bf_storage_indices_np = None
    active_mask_np = None
    if compact_inactive:
        active_mask_np = np.asarray(aperture_k_1d) > 0.0
        bf_storage_indices_np = np.flatnonzero(active_mask_np).astype(np.int32)
        if bf_storage_indices_np.size == 0:
            raise ValueError(
                "Automatic MPS SSB probe aperture contains no active BF pixels. "
                "Check the BF center, detector sampling, and semiangle."
            )

    g_chunks = []
    dc_chunks = []
    chunk_bf = max(1, int(chunk_bf))
    for start in range(0, int(bf_row.size), chunk_bf):
        stop = min(start + chunk_bf, int(bf_row.size))
        rows = bf_row[start:stop]
        cols = bf_col[start:stop]
        if compact_inactive and dc_value_override is not None:
            chunk_active = active_mask_np[start:stop]
            rows = rows[chunk_active]
            cols = cols[chunk_active]
            if rows.size == 0:
                continue
        if isinstance(frames, ArrayFrames):
            stack_mx = None
            stack_np = frames.columns(rows, cols).reshape(int(rows.size), *scan_shape).astype(np.float32, copy=False)
        else:
            # Gather into MLX-owned storage: a resident gather without an output allocates a Metal buffer that
            # nothing releases, so every preparation would strand one chunk of driver memory.
            stack_mx = mx.empty(
                (int(rows.size), *scan_shape),
                dtype=mx.float32,
            )
            mx.eval(stack_mx)
            stack_np = np.asarray(stack_mx)
            frames.columns_float32_into(
                rows,
                cols,
                out=stack_np.reshape(int(rows.size), -1),
            )
        if compact_inactive and dc_value_override is None:
            dc_chunks.append(
                stack_np.reshape(stop - start, -1).sum(
                    axis=1,
                    dtype=np.float32,
                )
            )
            # The compacted stack is a host copy and enters the FFT as a new array.
            stack_np = stack_np[active_mask_np[start:stop]]
            stack_mx = None
            if stack_np.shape[0] == 0:
                continue
        if stack_mx is not None and isinstance(frames, (MpsTensorFrames, TensorChunkedFrames)):
            # Preserve the previous array-input FFT layout and rounding. The
            # selected columns were scan-major; materialize that layout on GPU.
            stack_mx = mx.moveaxis(
                mx.contiguous(mx.moveaxis(stack_mx, 0, -1)), -1, 0,
            )
        g_chunk = _fft2_hermitian(
            mx,
            stack_mx if stack_mx is not None else mx.array(stack_np),
        )
        mx.eval(g_chunk)
        g_chunks.append(g_chunk)
    g_qk = g_chunks[0] if len(g_chunks) == 1 else mx.concatenate(g_chunks, axis=0)
    mx.eval(g_qk)
    # MLX's RFFT result is physically column-major. Materialize it once in
    # BF-major order so every repeated exact row kernel reads contiguous
    # evidence instead of paying an implicit slice-layout conversion.
    g_qk = mx.contiguous(g_qk)
    mx.eval(g_qk)
    if compact_inactive:
        if dc_value_override is None:
            dc_values = np.concatenate(dc_chunks).astype(np.complex64)
            dc_value = complex(dc_values.mean())
        else:
            dc_value = complex(np.complex64(dc_value_override))
        storage_index = mx.array(bf_storage_indices_np)
        kx_np = kx_np[bf_storage_indices_np]
        ky_np = ky_np[bf_storage_indices_np]
        kx_mx = kx_mx[storage_index]
        ky_mx = ky_mx[storage_index]
        alpha_k2_1d = alpha_k2_1d[storage_index]
        cos2_k_1d = cos2_k_1d[storage_index]
        sin2_k_1d = sin2_k_1d[storage_index]
        aperture_k_1d = aperture_k_1d[storage_index]
        mx.eval(
            kx_mx,
            ky_mx,
            alpha_k2_1d,
            cos2_k_1d,
            sin2_k_1d,
            aperture_k_1d,
        )
    else:
        dc_value = complex(np.asarray(g_qk[:, 0, 0]).mean())
    alpha_k2 = cos2_k = sin2_k = aperture_k = None
    alpha_m2 = cos2_m = sin2_m = ap_m = None
    alpha_p2 = cos2_p = sin2_p = ap_p = None
    # Cache static geometry when the selected BF set is small enough.  For the
    # held-out dataset bf_radius=5 path this is ~650 MB of float32 geometry and removes
    # repeated sqrt/aperture work from every optimizer batch.
    geometry_values = int(kx_np.size) * int(scan_shape[0]) * int(scan_shape[1])
    if geometry_values <= 32_000_000:
        kx = mx.array(kx_np, dtype=mx.float32)[:, None, None]
        ky = mx.array(ky_np, dtype=mx.float32)[:, None, None]
        alpha_k2, cos2_k, sin2_k, aperture_k = compute_geometry(
            mx, kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_m2, cos2_m, sin2_m, ap_m = compute_geometry(
            mx, qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_p2, cos2_p, sin2_p, ap_p = compute_geometry(
            mx, qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        mx.eval(
            alpha_k2, cos2_k, sin2_k, aperture_k,
            alpha_m2, cos2_m, sin2_m, ap_m,
            alpha_p2, cos2_p, sin2_p, ap_p,
        )

    dc_mask_np = np.zeros(scan_shape, dtype=bool)
    dc_mask_np[0, 0] = True
    return PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=qx,
        qy=qy,
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=kx_mx,
        ky=ky_mx,
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=dc_value,
        scan_shape=scan_shape,
        wavelength=wavelength,
        semiangle_rad=semiangle_rad,
        ang_y_rad=ang_y_rad,
        ang_x_rad=ang_x_rad,
        factor=math.pi / wavelength,
        dc_mask=mx.array(dc_mask_np),
        num_bf=selection.size,
        alpha_k2=alpha_k2,
        cos2_k=cos2_k,
        sin2_k=sin2_k,
        aperture_k=aperture_k,
        alpha_m2=alpha_m2,
        cos2_m=cos2_m,
        sin2_m=sin2_m,
        ap_m=ap_m,
        alpha_p2=alpha_p2,
        cos2_p=cos2_p,
        sin2_p=sin2_p,
        ap_p=ap_p,
        alpha_k2_1d=alpha_k2_1d,
        cos2_k_1d=cos2_k_1d,
        sin2_k_1d=sin2_k_1d,
        aperture_k_1d=aperture_k_1d,
        bf_storage_indices_np=bf_storage_indices_np,
    )


def retarget_prepared_rotation(
    prepared: PreparedMpsSSB,
    *,
    selection: BrightfieldDisk,
    rotation_angle_deg: float,
) -> None:
    """Update rotation-dependent geometry without rebuilding source FFT evidence."""

    mx = prepared.mx
    reciprocal_y = prepared.ang_y_rad / prepared.wavelength
    reciprocal_x = prepared.ang_x_rad / prepared.wavelength
    kx_np = (
        selection.rows.astype(np.float32) - selection.center_row_col[0]
    ) * reciprocal_y
    ky_np = (
        selection.cols.astype(np.float32) - selection.center_row_col[1]
    ) * reciprocal_x
    if rotation_angle_deg:
        angle = math.radians(-float(rotation_angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        kx_np, ky_np = (
            kx_np * cos_a + ky_np * sin_a,
            -kx_np * sin_a + ky_np * cos_a,
        )
    kx_np = np.asarray(kx_np, dtype=np.float32)
    ky_np = np.asarray(ky_np, dtype=np.float32)
    if prepared.bf_storage_indices_np is not None:
        indices = prepared.bf_storage_indices_np
        kx_np = kx_np[indices]
        ky_np = ky_np[indices]

    kx = mx.array(kx_np, dtype=mx.float32)
    ky = mx.array(ky_np, dtype=mx.float32)
    alpha_k2_1d, cos2_k_1d, sin2_k_1d, aperture_k_1d = compute_geometry(
        mx,
        kx,
        ky,
        prepared.wavelength,
        prepared.semiangle_rad,
        prepared.ang_y_rad,
        prepared.ang_x_rad,
    )
    arrays = [kx, ky, alpha_k2_1d, cos2_k_1d, sin2_k_1d, aperture_k_1d]

    alpha_k2 = cos2_k = sin2_k = aperture_k = None
    alpha_m2 = cos2_m = sin2_m = ap_m = None
    alpha_p2 = cos2_p = sin2_p = ap_p = None
    if prepared.alpha_k2 is not None:
        kx_grid = kx[:, None, None]
        ky_grid = ky[:, None, None]
        alpha_k2, cos2_k, sin2_k, aperture_k = compute_geometry(
            mx,
            kx_grid,
            ky_grid,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
        alpha_m2, cos2_m, sin2_m, ap_m = compute_geometry(
            mx,
            prepared.qx - kx_grid,
            prepared.qy - ky_grid,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
        alpha_p2, cos2_p, sin2_p, ap_p = compute_geometry(
            mx,
            prepared.qx + kx_grid,
            prepared.qy + ky_grid,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
        arrays.extend(
            [
                alpha_k2,
                cos2_k,
                sin2_k,
                aperture_k,
                alpha_m2,
                cos2_m,
                sin2_m,
                ap_m,
                alpha_p2,
                cos2_p,
                sin2_p,
                ap_p,
            ]
        )
    mx.eval(*arrays)

    prepared.kx = kx
    prepared.ky = ky
    prepared.kx_np = kx_np
    prepared.ky_np = ky_np
    prepared.alpha_k2_1d = alpha_k2_1d
    prepared.cos2_k_1d = cos2_k_1d
    prepared.sin2_k_1d = sin2_k_1d
    prepared.aperture_k_1d = aperture_k_1d
    prepared.alpha_k2 = alpha_k2
    prepared.cos2_k = cos2_k
    prepared.sin2_k = sin2_k
    prepared.aperture_k = aperture_k
    prepared.alpha_m2 = alpha_m2
    prepared.cos2_m = cos2_m
    prepared.sin2_m = sin2_m
    prepared.ap_m = ap_m
    prepared.alpha_p2 = alpha_p2
    prepared.cos2_p = cos2_p
    prepared.sin2_p = sin2_p
    prepared.ap_p = ap_p


def resolve_bf_selection(
    data,
    threshold: float,
    bf_radius: float | None,
    center_override: tuple[float, float] | None = None,
    *,
    mean_diffraction: np.ndarray | None = None,
    detected_radius_px: float | None = None,
) -> BrightfieldDisk:
    """Return one validated BF selection for exact or raw detector input.

    Without ``bf_radius`` the disk is ``detector.fit_probe``'s equal-area disk (pixels above mean + std); with it the
    centre is the intensity-weighted centroid. ``detected_radius_px`` is the disk edge radius the detector calibration
    uses: the caller's (``semiangle / det_sampling`` when the sampling is given, which a bright-field crop must pass
    because its pattern has no edge to measure) or ``disk_edge_radius`` of the mean pattern, the rule CUDA uses too.
    Exact BF columns carry the selection and radius they were exported with.
    """

    if isinstance(data, MpsBfColumnFrames):
        if center_override is not None or bf_radius is not None:
            raise ValueError(
                "MpsBfColumnFrames owns its exact BF selection; do not pass "
                "bf_center or bf_radius overrides."
            )
        return data.selection

    mean_pattern = detector_mean(data) if mean_diffraction is None else mean_diffraction
    if detected_radius_px is None:
        detected_radius_px = disk_edge_radius(mean_pattern)
    if bf_radius is None:
        probe_center, probe_radius = fit_probe(mean_pattern)
    mask = mean_pattern > float(mean_pattern.max()) * float(threshold)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError(
            f"No bright-field pixels found with threshold={threshold:.2f}."
        )
    if center_override is not None:
        center = (
            float(center_override[0]),
            float(center_override[1]),
        )
    elif bf_radius is None:
        center = (float(probe_center[0]), float(probe_center[1]))
    else:
        weights = mean_pattern[rows, cols].astype(np.float32, copy=False)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            center = (
                float((rows.astype(np.float32) * weights).sum() / weight_sum),
                float((cols.astype(np.float32) * weights).sum() / weight_sum),
            )
        else:
            center = (float(rows.mean()), float(cols.mean()))
    selected_radius = float(probe_radius if bf_radius is None else bf_radius)
    distance_sq = (rows.astype(np.float32) - center[0]) ** 2 + (
        cols.astype(np.float32) - center[1]
    ) ** 2
    keep = distance_sq <= selected_radius**2
    rows = rows[keep]
    cols = cols[keep]
    if rows.size == 0:
        raise ValueError(
            f"No BF pixels selected with threshold={threshold} and radius={bf_radius}."
        )
    return BrightfieldDisk(
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        center_row_col=center,
        radius_px=selected_radius,
        detected_radius_px=float(detected_radius_px),
        detector_shape=tuple(int(value) for value in mean_pattern.shape),
    )


def pk_from_prepared(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
):
    """Probe term ``p(k)`` for each selected BF pixel."""
    mx = prepared.mx
    alpha_k2 = prepared.alpha_k2_1d
    cos2_k = prepared.cos2_k_1d
    sin2_k = prepared.sin2_k_1d
    aperture_k = prepared.aperture_k_1d
    if alpha_k2 is None or cos2_k is None or sin2_k is None or aperture_k is None:
        alpha_k2, cos2_k, sin2_k, aperture_k = compute_geometry(
            mx,
            prepared.kx,
            prepared.ky,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
    chi_k = prepared.factor * alpha_k2 * (
        float(C12)
        * (
            cos2_k * math.cos(2.0 * float(phi12))
            + sin2_k * math.sin(2.0 * float(phi12))
        )
        + float(C10)
    )
    pk = aperture_k * _exp_neg_i(mx, chi_k)
    mx.eval(pk)
    return pk


def pk_batch_from_prepared(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Batched probe terms ``p(k)`` for a BF slice."""
    mx = prepared.mx
    alpha_k2 = prepared.alpha_k2_1d
    cos2_k = prepared.cos2_k_1d
    sin2_k = prepared.sin2_k_1d
    aperture_k = prepared.aperture_k_1d
    if alpha_k2 is None or cos2_k is None or sin2_k is None or aperture_k is None:
        alpha_k2, cos2_k, sin2_k, aperture_k = compute_geometry(
            mx,
            prepared.kx,
            prepared.ky,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
    bf_slice = slice(int(start), int(stop))
    alpha = alpha_k2[bf_slice][None, :]
    cos2 = cos2_k[bf_slice][None, :]
    sin2 = sin2_k[bf_slice][None, :]
    aperture = aperture_k[bf_slice][None, :]
    c10 = c10[:, None]
    c12 = c12[:, None]
    cos2phi12 = cos2phi12[:, None]
    sin2phi12 = sin2phi12[:, None]
    chi_k = prepared.factor * alpha * (
        c12 * (cos2 * cos2phi12 + sin2 * sin2phi12) + c10
    )
    pk = aperture * _exp_neg_i(mx, chi_k)
    mx.eval(pk)
    return pk


def bf_storage_chunks(
    prepared: PreparedMpsSSB,
    logical_chunk_bf: int,
):
    """Yield packed storage slices without moving logical reduction boundaries."""
    logical_chunk_bf = max(1, int(logical_chunk_bf))
    indices = prepared.bf_storage_indices_np
    if indices is None:
        for start in range(0, prepared.num_bf, logical_chunk_bf):
            yield start, min(start + logical_chunk_bf, prepared.num_bf)
        return
    indices = np.asarray(indices, dtype=np.intp)
    for logical_start in range(0, prepared.num_bf, logical_chunk_bf):
        logical_stop = min(logical_start + logical_chunk_bf, prepared.num_bf)
        storage_start = int(np.searchsorted(indices, logical_start, side="left"))
        storage_stop = int(np.searchsorted(indices, logical_stop, side="left"))
        if storage_start != storage_stop:
            yield storage_start, storage_stop


def bf_storage_chunk_packs(
    prepared: PreparedMpsSSB,
    logical_chunk_bf: int,
    max_storage_bf: int,
) -> list[list[tuple[int, int]]]:
    """Pack adjacent sparse boundaries without exceeding row scratch limits."""
    boundaries = list(bf_storage_chunks(prepared, logical_chunk_bf))
    if not boundaries:
        return []
    max_storage_bf = max(1, int(max_storage_bf))
    packs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_size = 0
    for start, stop in boundaries:
        boundary_size = int(stop) - int(start)
        if current and current_size + boundary_size > max_storage_bf:
            packs.append(current)
            current = []
            current_size = 0
        current.append((int(start), int(stop)))
        current_size += boundary_size
    if current:
        packs.append(current)
    return packs


def compute_geometry(mx, dx, dy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad):
    """Probe geometry at reciprocal vectors ``(dx, dy)``: alpha^2, cos 2phi, sin 2phi and the soft aperture.

    alpha = lambda |(dx, dy)| is the scattering angle and phi its azimuth. The aperture is
    ``(semiangle - alpha) / w + 1/2`` clipped to [0, 1], where ``w`` is the angular width of one detector pixel along
    ``(dx, dy)``: a one-pixel linear edge instead of a hard cut. The origin, which has no direction, is fully inside.
    """
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = mx.sqrt(r2)
    alpha = r * wavelength
    alpha2 = alpha * alpha
    inv_r2 = mx.where(r2 > 1e-30, 1.0 / r2, 0.0)
    cos2phi = (dx2 - dy2) * inv_r2
    sin2phi = 2.0 * dx * dy * inv_r2
    denom_num2 = (dx * ang_y_rad) ** 2 + (dy * ang_x_rad) ** 2
    inv_r = mx.where(r > 1e-15, 1.0 / r, 0.0)
    denom = mx.sqrt(denom_num2) * inv_r
    edge = mx.where(denom > 1e-15, (semiangle_rad - alpha) / denom + 0.5, 1.0)
    aperture = mx.clip(edge, 0.0, 1.0)
    return alpha2, cos2phi, sin2phi, aperture


def as_sampling(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return float(value[0]), float(value[1])


def expand_hermitian_mx(mx, g_qk, full_cols: int):
    """Expand an MLX Hermitian half-plane stack to a full Fourier grid."""
    shape = tuple(int(size) for size in g_qk.shape)
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3D G_qk, got shape {shape}.")
    full_cols = int(full_cols)
    if shape[-1] == full_cols:
        return g_qk
    expected_cols = full_cols // 2 + 1
    if shape[-1] != expected_cols:
        raise ValueError(
            f"Expected Hermitian G_qk with {expected_cols} columns or full "
            f"{full_cols} columns, got shape {shape}."
        )
    n_rows = int(shape[-2])
    mirror_rows = mx.array(((-np.arange(n_rows)) % n_rows).astype(np.int32))
    mirror_cols = mx.array(
        np.arange(full_cols - expected_cols, 0, -1, dtype=np.int32)
    )
    mirrored_rows = mx.take(g_qk, mirror_rows, axis=-2)
    mirrored = mx.take(mirrored_rows, mirror_cols, axis=-1)
    return mx.concatenate([g_qk, mx.conjugate(mirrored)], axis=-1)


def ifft2_chunked(mx, fourier_stack):
    """Run a chunked 2D inverse FFT with the faster MLX row-column schedule."""
    row_ifft = mx.fft.ifft(fourier_stack, axis=-1)
    return mx.fft.ifft(row_ifft, axis=-2)


def _fft2_hermitian(mx, real_stack):
    """Return the nonredundant FFT half-plane for a real BF stack."""
    return mx.fft.rfft2(real_stack)


def _spatial_frequencies(shape: tuple[int, int], sampling: tuple[float, float]):
    return (
        np.fft.fftfreq(shape[0], sampling[0]).astype(np.float32),
        np.fft.fftfreq(shape[1], sampling[1]).astype(np.float32),
    )


def _exp_neg_i(mx, chi):
    return mx.cos(chi) - (1j * mx.sin(chi))
