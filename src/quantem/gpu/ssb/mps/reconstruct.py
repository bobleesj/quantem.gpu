"""Phase, loss and object reconstruction on MPS from the prepared state."""

import math

import numpy as np

from quantem.gpu.ssb.mps.hardware import default_phase_col_k_bf
from quantem.gpu.ssb.mps.kernels.corrected import (
    corrected_from_cached_geometry,
    corrected_from_dynamic_geometry,
)
from quantem.gpu.ssb.mps.kernels.phase_columns import (
    phase_cols_small_scalar_loss_from_row_ifft,
    phase_cols_small_sum_from_row_ifft,
)
from quantem.gpu.ssb.mps.kernels.phase_columns_512 import (
    phase_cols512_scalar_loss_batch_from_row_ifft,
    phase_cols512_scalar_loss_from_row_ifft,
    phase_cols512_sum_from_row_ifft,
)
from quantem.gpu.ssb.mps.kernels.phase_sums import (
    phase_sum_from_complex,
    phase_sums_from_complex,
)
from quantem.gpu.ssb.mps.kernels.row_ifft import (
    row_ifft512_from_dynamic_geometry,
    row_ifft_small_from_dynamic_geometry,
)
from quantem.gpu.ssb.mps.prepared import (
    PreparedMpsSSB,
    bf_storage_chunks,
    ifft2_chunked,
)


def reconstruct_prepared(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    chunk_bf: int,
    compute_loss: bool,
    compute_object: bool,
    return_phase: bool = True,
    thick=None,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
    """Run SSB correction from a prepared BF FFT stack.

    ``thick`` = (thickness, tilt_row_rad, tilt_col_rad) applies the thick-sample depth weights inside the fused
    128/256/1024 row kernel (``thick_sample.reconstruct_thick``); None is standard SSB.
    """
    mx = prepared.mx
    if thick is not None and (
        prepared.scan_shape not in ((128, 128), (256, 256), (1024, 1024)) or compute_object
    ):
        raise ValueError("The fused thick-sample path covers 128/256/1024 phase images only.")
    accumulator = (
        mx.zeros(prepared.scan_shape, dtype=mx.complex64)
        if compute_object else None
    )
    # CUDA's fixed SSB output is the mean of per-BF phase images, not the
    # phase of the averaged complex object wave. Keep that contract for reference agreement.
    phase_sum = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    uses_scalar_512_loss = prepared.scan_shape == (512, 512)
    uses_scalar_dynamic_loss = (
        prepared.scan_shape in ((128, 128), (256, 256), (1024, 1024))
        and (prepared.alpha_k2 is None or thick is not None)
    )
    use_scalar_loss = (
        compute_loss
        and not compute_object
        and (uses_scalar_512_loss or uses_scalar_dynamic_loss)
    )
    if use_scalar_loss:
        phase_sumsq = mx.array(0.0, dtype=mx.float32)
    elif compute_loss:
        phase_sumsq = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    else:
        phase_sumsq = None
    c10_values = mx.array([float(C10)], dtype=mx.float32)
    c12_values = mx.array([float(C12)], dtype=mx.float32)
    cos2phi12_values = mx.array(
        [math.cos(2.0 * float(phi12))],
        dtype=mx.float32,
    )
    sin2phi12_values = mx.array(
        [math.sin(2.0 * float(phi12))],
        dtype=mx.float32,
    )
    chunk_bf = max(1, int(chunk_bf))
    phase_col_k_bf = default_phase_col_k_bf(prepared.scan_shape)

    for start, stop in bf_storage_chunks(prepared, chunk_bf):
        use_fused_row = (
            prepared.scan_shape in ((128, 128), (256, 256), (512, 512), (1024, 1024))
            and not compute_object
            and (prepared.alpha_k2 is None or thick is not None)
        )
        if use_fused_row:
            if prepared.scan_shape == (512, 512):
                row_ifft, active_bf = row_ifft512_from_dynamic_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                    # The column stage reads eight rows of one column per
                    # threadgroup.  Storing the row IFFT in the tiled layout
                    # keeps those eight loads inside one 64-byte granule
                    # instead of striding 4 KB apart.  Same values, permuted
                    # addresses: 22.767 to 21.055 ms on the stage pair and
                    # 404.0 to 369.5 ms end-to-end, bit-exact
                    # (experiment 20260916-ssb-mps-hotpath).
                    return_active=True,
                    tiled_output=True,
                )
                if compute_loss:
                    batch_sum, batch_sumsq = (
                        phase_cols512_scalar_loss_batch_from_row_ifft(
                            mx,
                            row_ifft[None, ...],
                            k_bf=phase_col_k_bf,
                            active_bf=active_bf,
                            tiled_input=True,
                        )
                    )
                    chunk_sum = batch_sum[0]
                    chunk_sumsq = batch_sumsq[0]
                else:
                    chunk_sum = phase_cols512_sum_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                        active_bf=active_bf,
                        tiled_input=True,
                    )
                    chunk_sumsq = None
            else:
                row_ifft = row_ifft_small_from_dynamic_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                    thick=thick,
                )
                if compute_loss:
                    chunk_sum, chunk_sumsq = phase_cols_small_scalar_loss_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                else:
                    chunk_sum = phase_cols_small_sum_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                    chunk_sumsq = None
        else:
            if prepared.alpha_k2 is not None:
                corrected = corrected_from_cached_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                )[0]
            else:
                corrected = corrected_from_dynamic_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                )[0]
            if prepared.scan_shape == (512, 512) and not compute_object:
                row_ifft = mx.fft.ifft(corrected, axis=-1)
                if compute_loss:
                    chunk_sum, chunk_sumsq = phase_cols512_scalar_loss_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                else:
                    chunk_sum = phase_cols512_sum_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                    chunk_sumsq = None
            else:
                obj_chunk = ifft2_chunked(mx, corrected)
                if compute_object:
                    accumulator = accumulator + mx.sum(obj_chunk, axis=0)
                if compute_loss:
                    chunk_sum, chunk_sumsq = phase_sums_from_complex(mx, obj_chunk)
                else:
                    chunk_sum = phase_sum_from_complex(mx, obj_chunk)
                    chunk_sumsq = None
        if compute_loss:
            phase_sum = phase_sum + chunk_sum
            phase_sumsq = phase_sumsq + chunk_sumsq
        else:
            phase_sum = phase_sum + chunk_sum
        mx.eval(
            *[
                arr for arr in (accumulator, phase_sum, phase_sumsq)
                if arr is not None
            ]
        )

    object_wave = None
    if compute_object:
        object_wave_mx = accumulator / prepared.num_bf
        mx.eval(object_wave_mx)
        object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    mean_phase_mx = phase_sum / prepared.num_bf
    loss = None
    if compute_loss:
        if use_scalar_loss:
            mean_sq = mx.mean(mean_phase_mx * mean_phase_mx)
            norm = float(
                prepared.num_bf * prepared.scan_shape[0] * prepared.scan_shape[1]
            )
            loss = float(np.asarray(phase_sumsq / norm - mean_sq))
        else:
            var_per_pixel = phase_sumsq / prepared.num_bf - mean_phase_mx * mean_phase_mx
            loss = float(np.asarray(mx.mean(var_per_pixel)))
    mean_phase = None
    if return_phase:
        mx.eval(mean_phase_mx)
        mean_phase = np.asarray(mean_phase_mx).astype(np.float32, copy=False)
    return object_wave, loss, mean_phase
