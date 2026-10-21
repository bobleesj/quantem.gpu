"""Phase, loss and object reconstruction on MPS from the prepared state."""

import math
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from quantem.gpu.ssb._execution import checkpoint, is_cancellable
from quantem.gpu.ssb.mps.hardware import default_phase_col_k_bf
from quantem.gpu.ssb.mps.kernels.corrected import (
    corrected_from_cached_geometry,
    corrected_from_dynamic_geometry,
)
from quantem.gpu.ssb.mps.kernels.phase_columns import (
    phase_cols_small_loss_partials,
    phase_cols_small_scalar_loss_from_row_ifft,
    phase_cols_small_sum_from_row_ifft,
)
from quantem.gpu.ssb.mps.kernels.phase_columns_512 import (
    continue_phase_cols512_loss,
    phase_cols512_scalar_loss_batch_from_row_ifft,
    phase_cols512_scalar_loss_from_row_ifft,
    phase_cols512_sum_from_row_ifft,
    reduce_phase_cols512_loss,
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

_DIAGNOSTIC_BF_TILE = 256
# Bright-field pixels per row IFFT of a 512 x 512 loss chunk outside a cancellable preview. A chunk larger than the
# tile goes through the tiled path of a cancellable preview: the same reduction boundaries and float32 order, so the
# same loss and phase bits, with one tile of row IFFT (2.1 GB at 1024) alive at a time. None keeps the whole chunk
# (4096 pixels and 8.6 GB on Macs with 96 GB or more), which MLX's cache then keeps between slider releases: about
# 4 % faster loss previews for 6 to 12 GB more resident memory, an opt-in for big Macs. Chunks of 1024 or fewer
# pixels (Macs below 96 GB) are not split either way.
LOSS_ROW_IFFT_TILE_BF: int | None = 1024
# The tile inside find_aberrations: its few loss and phase evaluations (the fit's final loss, both branches' phases)
# take a negligible share of the fit's time, so they run in quarter-size tiles and the fit keeps 0.5 GB of row IFFT
# instead of 2.1 GB. The same bits either way.
FIT_LOSS_ROW_IFFT_TILE_BF: int | None = 256
_loss_tile_override: ContextVar[int | None] = ContextVar("loss_row_ifft_tile_bf", default=None)


@contextmanager
def fit_loss_tiles():
    """Run the enclosed loss evaluations with ``FIT_LOSS_ROW_IFFT_TILE_BF`` tiles (``SSB.find_aberrations``)."""
    token = _loss_tile_override.set(FIT_LOSS_ROW_IFFT_TILE_BF)
    try:
        yield
    finally:
        _loss_tile_override.reset(token)


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
    128/256/512/1024 row kernel (``thick_sample.reconstruct_thick``); None is standard SSB.
    """
    mx = prepared.mx
    if thick is not None and (
        prepared.scan_shape not in ((128, 128), (256, 256), (512, 512), (1024, 1024)) or compute_object
    ):
        raise ValueError("The fused thick-sample path covers 128/256/512/1024 phase images only.")
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
        checkpoint()
        use_fused_row = (
            prepared.scan_shape in ((128, 128), (256, 256), (512, 512), (1024, 1024))
            and not compute_object
            and (prepared.alpha_k2 is None or thick is not None)
        )
        loss_tile = _loss_tile_override.get() or LOSS_ROW_IFFT_TILE_BF
        if _loss_tile_override.get() is not None and prepared.scan_shape != (512, 512):
            # Keep the fit's same row-FFT byte budget across scan sizes.
            # Up to 1,024 small planes avoid dozens of tiny dispatches.
            loss_tile = max(phase_col_k_bf, min(
                1024, loss_tile * 512 * 512 // math.prod(prepared.scan_shape),
            ))
        tiled = is_cancellable() or (loss_tile is not None and stop - start > loss_tile)
        if use_fused_row and prepared.scan_shape == (512, 512) and compute_loss and tiled:
            chunk_sum, chunk_sumsq = _cancellable_512_loss_chunk(
                prepared, start=start, stop=stop, c10=c10_values, c12=c12_values,
                cos2phi12=cos2phi12_values, sin2phi12=sin2phi12_values,
                phase_col_k_bf=phase_col_k_bf, thick=thick,
                tile_bf=_DIAGNOSTIC_BF_TILE if is_cancellable() else loss_tile,
            )
        elif use_fused_row and compute_loss and tiled:
            chunk_sum, chunk_sumsq = _tiled_small_loss_chunk(
                prepared, start=start, stop=stop, c10=c10_values, c12=c12_values,
                cos2phi12=cos2phi12_values, sin2phi12=sin2phi12_values,
                phase_col_k_bf=phase_col_k_bf, thick=thick,
                tile_bf=_DIAGNOSTIC_BF_TILE if is_cancellable() else loss_tile,
            )
        elif use_fused_row:
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
                    thick=thick,
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
                object_chunk = ifft2_chunked(mx, corrected)
                if compute_object:
                    accumulator = accumulator + mx.sum(object_chunk, axis=0)
                if compute_loss:
                    chunk_sum, chunk_sumsq = phase_sums_from_complex(mx, object_chunk)
                else:
                    chunk_sum = phase_sum_from_complex(mx, object_chunk)
                    chunk_sumsq = None
        if compute_loss:
            phase_sum = phase_sum + chunk_sum
            phase_sumsq = phase_sumsq + chunk_sumsq
        else:
            phase_sum = phase_sum + chunk_sum
        mx.eval(
            *[
                array for array in (accumulator, phase_sum, phase_sumsq)
                if array is not None
            ]
        )

    object_wave = None
    checkpoint()
    if compute_object:
        object_wave_mx = accumulator / prepared.num_bright_field_pixels
        mx.eval(object_wave_mx)
        object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    mean_phase_mx = phase_sum / prepared.num_bright_field_pixels
    loss = None
    if compute_loss:
        if use_scalar_loss:
            mean_sq = mx.mean(mean_phase_mx * mean_phase_mx)
            norm = float(
                prepared.num_bright_field_pixels * prepared.scan_shape[0] * prepared.scan_shape[1]
            )
            loss = float(np.asarray(phase_sumsq / norm - mean_sq))
        else:
            var_per_pixel = phase_sumsq / prepared.num_bright_field_pixels - mean_phase_mx * mean_phase_mx
            loss = float(np.asarray(mx.mean(var_per_pixel)))
    mean_phase = None
    if return_phase:
        mx.eval(mean_phase_mx)
        mean_phase = np.asarray(mean_phase_mx).astype(np.float32, copy=False)
    return object_wave, loss, mean_phase


def _tiled_small_loss_chunk(
    prepared, *, start, stop, c10, c12, cos2phi12, sin2phi12,
    phase_col_k_bf, thick, tile_bf,
):
    """Bound the row FFT stack while retaining every original BF reduction group."""
    mx = prepared.mx
    # Never split a column kernel's sequential BF sum. Concatenate its group
    # outputs before the unchanged whole-chunk reduction, including the tail.
    tile_bf = max(phase_col_k_bf, tile_bf // phase_col_k_bf * phase_col_k_bf)
    sums, squares = [], []
    for first in range(start, stop, tile_bf):
        checkpoint()
        row_ifft = row_ifft_small_from_dynamic_geometry(
            prepared, start=first, stop=min(first + tile_bf, stop),
            c10=c10, c12=c12, cos2phi12=cos2phi12, sin2phi12=sin2phi12,
            thick=thick,
        )
        partial, partial_sq = phase_cols_small_loss_partials(
            mx, row_ifft, k_bf=phase_col_k_bf,
        )
        mx.eval(partial, partial_sq)
        sums.append(partial)
        squares.append(partial_sq)
    partial = sums[0] if len(sums) == 1 else mx.concatenate(sums, axis=0)
    partial_sq = squares[0] if len(squares) == 1 else mx.concatenate(squares, axis=0)
    total = partial[0] if partial.shape[0] == 1 else mx.sum(partial, axis=0)
    return total, mx.sum(partial_sq)


def _cancellable_512_loss_chunk(
    prepared, *, start, stop, c10, c12, cos2phi12, sin2phi12, phase_col_k_bf, thick,
    tile_bf=_DIAGNOSTIC_BF_TILE,
):
    """Tile independent BF groups without changing any reduction boundary.

    A diagnostic can yield between completed row/column FFT batches. Partial
    groups retain their original order and are joined before the unchanged
    reduction, so neither BF coverage nor float32 accumulation changes.
    ``tile_bf`` pixels go through one row IFFT, and each tile is evaluated
    before the next is built: left lazy, MLX holds every tile's row IFFT at
    once, as much memory as the whole chunk.
    """
    mx = prepared.mx
    sums, squares = [], []
    for group in range(start, stop, phase_col_k_bf):
        group_stop = min(group + phase_col_k_bf, stop)
        partial = mx.zeros((1, 1, 512, 512), dtype=mx.float32)
        partial_sq = mx.zeros_like(partial)
        for first in range(group, group_stop, tile_bf):
            checkpoint()
            last = min(first + tile_bf, group_stop)
            row_ifft, active_bf = row_ifft512_from_dynamic_geometry(
                prepared, start=first, stop=last,
                c10=c10, c12=c12, cos2phi12=cos2phi12, sin2phi12=sin2phi12,
                return_active=True, tiled_output=True, thick=thick,
            )
            partial, partial_sq = continue_phase_cols512_loss(
                mx, row_ifft, active_bf, partial, partial_sq,
                finalize=last == group_stop,
            )
            mx.eval(partial, partial_sq)
        sums.append(partial)
        squares.append(partial_sq)
    checkpoint()
    partial = sums[0] if len(sums) == 1 else mx.concatenate(sums, axis=1)
    partial_sq = squares[0] if len(squares) == 1 else mx.concatenate(squares, axis=1)
    total, total_sq = reduce_phase_cols512_loss(mx, partial, partial_sq)
    return total[0], total_sq[0]
