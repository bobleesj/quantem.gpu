"""The exact full-BF phase-variance loss for batches of candidate aberrations on MPS (the fit's objective)."""

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from quantem.gpu.ssb.mps.hardware import (
    default_phase_col_k_bf,
    exact_pair_row_allocation_bf_512,
    exact_pair_row_policy_512,
)
from quantem.gpu.ssb.mps.kernels.phase_columns import (
    phase_cols_small_scalar_loss_batch_from_row_ifft,
)
from quantem.gpu.ssb.mps.kernels.phase_columns_512 import (
    phase_cols512_pack_loss_batch_from_row_ifft,
    phase_cols512_scalar_loss_batch_from_row_ifft,
)
from quantem.gpu.ssb.mps.kernels.row_ifft import (
    row_ifft512_batch_from_dynamic_geometry,
    row_ifft_small_batch_from_dynamic_geometry,
)
from quantem.gpu.ssb.mps.prepared import (
    PreparedMpsSSB,
    bf_storage_chunk_packs,
    bf_storage_chunks,
    pk_batch_from_prepared,
)
from quantem.gpu.ssb.mps.reconstruct import reconstruct_prepared

# Keep the logical reduction boundaries used by the exact 512 pair path
# independent of the high-memory phase-loss chunk default.  On 96+ GB Macs
# that default is 4096 BF; with compact inactive-plane storage a single such
# boundary can still contain more than the retained 288/320-plane row
# allocation classes.  The pair kernel may pack adjacent 512-boundary slices,
# but must not merge a larger logical boundary or change reduction order.
_EXACT_PAIR_LOGICAL_BOUNDARY_BF_512 = 512
# single-candidate 512 calls keep this many packs in flight (no reduction boundary changes)
_EXACT_SCALAR_ROW_PACK_DEPTH_512 = 5



def reconstruct_prepared_batch_exact_loss(
    prepared: PreparedMpsSSB,
    *,
    C10: np.ndarray,
    C12: np.ndarray,
    phi12: np.ndarray,
    chunk_bf: int,
) -> np.ndarray:
    """Evaluate the exact full-BF phase-variance loss for candidate batches."""
    c10_np = np.asarray(C10, dtype=np.float32).reshape(-1)
    c12_np = np.asarray(C12, dtype=np.float32).reshape(-1)
    phi_np = np.asarray(phi12, dtype=np.float32).reshape(-1)
    if c10_np.size == 0:
        return np.empty((0,), dtype=np.float32)
    if c12_np.size != c10_np.size or phi_np.size != c10_np.size:
        raise ValueError("C10, C12, and phi12 must have matching lengths.")

    batch = int(c10_np.size)
    if (
        prepared.scan_shape in ((128, 128), (256, 256), (1024, 1024))
        and prepared.alpha_k2 is None
        and batch == 2
    ):
        return _reconstruct_prepared_small_batch_exact_loss_fused(
            prepared,
            c10_np=c10_np,
            c12_np=c12_np,
            phi_np=phi_np,
            chunk_bf=chunk_bf,
            packed_columns=prepared.scan_shape == (256, 256),
        )
    if (
        prepared.scan_shape in ((128, 128), (256, 256))
        and prepared.alpha_k2 is None
        and batch > 1
    ):
        executor = _small_exact_executor()
        futures = [
            executor.submit(
                _small_exact_loss_worker,
                prepared,
                c10,
                c12,
                phi,
                chunk_bf,
            )
            for c10, c12, phi in zip(c10_np, c12_np, phi_np)
        ]
        return np.asarray(
            [future.result() for future in futures],
            dtype=np.float32,
        )

    if prepared.scan_shape != (512, 512) or prepared.alpha_k2 is not None:
        losses = []
        for c10, c12, phi in zip(c10_np, c12_np, phi_np):
            _object_wave, loss, _phase = reconstruct_prepared(
                prepared,
                C10=float(c10),
                C12=float(c12),
                phi12=float(phi),
                chunk_bf=chunk_bf,
                compute_loss=True,
                compute_object=False,
                return_phase=False,
            )
            if loss is None:
                raise RuntimeError("Exact MPS SSB objective did not return a loss.")
            losses.append(float(loss))
        return np.asarray(losses, dtype=np.float32)

    # The 512 kernels fuse two candidates while sharing the large G_qk read.
    # Wider fused batches increase threadgroup storage and register pressure;
    # the real 2,464-plane benchmark made batch 4 about 2.8x slower per
    # candidate than executing two fused pairs. Keep the public batch API, but
    # tile it through the measured pair topology.
    if batch > 2:
        return np.concatenate(
            [
                reconstruct_prepared_batch_exact_loss(
                    prepared,
                    C10=c10_np[start : start + 2],
                    C12=c12_np[start : start + 2],
                    phi12=phi_np[start : start + 2],
                    chunk_bf=chunk_bf,
                )
                for start in range(0, batch, 2)
            ]
        )

    mx = prepared.mx
    phase_sum = mx.zeros((batch, *prepared.scan_shape), dtype=mx.float32)
    phase_sumsq = mx.zeros((batch,), dtype=mx.float32)
    c10_values = mx.array(c10_np, dtype=mx.float32)
    c12_values = mx.array(c12_np, dtype=mx.float32)
    cos2phi12_values = mx.array(np.cos(2.0 * phi_np).astype(np.float32))
    sin2phi12_values = mx.array(np.sin(2.0 * phi_np).astype(np.float32))
    chunk_bf = _effective_exact_batch_chunk_bf(chunk_bf, prepared.scan_shape, batch)
    phase_col_k_bf = default_phase_col_k_bf(prepared.scan_shape)
    storage_num_bf = int(prepared.g_qk.shape[0])
    pk_all = pk_batch_from_prepared(
        prepared,
        start=0,
        stop=storage_num_bf,
        c10=c10_values,
        c12=c12_values,
        cos2phi12=cos2phi12_values,
        sin2phi12=sin2phi12_values,
    )
    active_bf_all = (mx.abs(pk_all[0]) > 0.0).astype(mx.uint8)
    mx.eval(active_bf_all)
    pair_logical_chunk_bf = min(
        chunk_bf,
        _EXACT_PAIR_LOGICAL_BOUNDARY_BF_512,
    )
    row_pack_limit, row_storage_classes = exact_pair_row_policy_512(batch)
    storage_packs = bf_storage_chunk_packs(
        prepared,
        pair_logical_chunk_bf,
        max_storage_bf=row_pack_limit,
    )

    for pack_index, pack in enumerate(storage_packs):
        start = pack[0][0]
        stop = pack[-1][1]
        row_ifft = row_ifft512_batch_from_dynamic_geometry(
            prepared,
            start=start,
            stop=stop,
            c10=c10_values,
            c12=c12_values,
            cos2phi12=cos2phi12_values,
            sin2phi12=sin2phi12_values,
            pk_override=pk_all[:, start:stop],
            # Reuse bounded allocation sizes instead of retaining every real
            # sparse-pack shape. A pack that is wider than every retained
            # class still runs at its own width because one logical boundary
            # is never split; only the written prefix is consumed.
            storage_bf=(
                exact_pair_row_allocation_bf_512(
                    stop - start,
                    row_storage_classes,
                )
                if batch == 2
                else None
            ),
        )
        relative_ranges = tuple(
            (boundary_start - start, boundary_stop - start)
            for boundary_start, boundary_stop in pack
        )
        if batch == 2 and len(relative_ranges) > 1:
            chunk_sums, chunk_sumsqs = (
                phase_cols512_pack_loss_batch_from_row_ifft(
                    mx,
                    row_ifft,
                    bf_ranges=relative_ranges,
                    active_bf=active_bf_all[start:stop],
                )
            )
        else:
            chunk_sums = []
            chunk_sumsqs = []
            for boundary_start, boundary_stop in pack:
                chunk_sum, chunk_sumsq = (
                    phase_cols512_scalar_loss_batch_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                        active_bf=active_bf_all[start:stop],
                        tiled_input=batch <= 2,
                        bf_start=boundary_start - start,
                        bf_stop=boundary_stop - start,
                    )
                )
                chunk_sums.append(chunk_sum)
                chunk_sumsqs.append(chunk_sumsq)
        for chunk_sum, chunk_sumsq in zip(chunk_sums, chunk_sumsqs):
            phase_sum = phase_sum + chunk_sum
            phase_sumsq = phase_sumsq + chunk_sumsq
        # Scalar calls benefit from two in-flight packs without increasing the
        # paired optimizer's working set or changing any reduction boundary.
        if (
            batch != 1
            or (pack_index + 1) % _EXACT_SCALAR_ROW_PACK_DEPTH_512 == 0
            or pack_index + 1 == len(storage_packs)
        ):
            mx.eval(phase_sum, phase_sumsq)

    mean_phase = phase_sum / prepared.num_bf
    mean_sq = mx.mean(mean_phase * mean_phase, axis=(1, 2))
    norm = float(prepared.num_bf * prepared.scan_shape[0] * prepared.scan_shape[1])
    losses = phase_sumsq / norm - mean_sq
    mx.eval(losses)
    return np.asarray(losses).astype(np.float32, copy=False)


def _reconstruct_prepared_small_batch_exact_loss_fused(
    prepared: PreparedMpsSSB,
    *,
    c10_np: np.ndarray,
    c12_np: np.ndarray,
    phi_np: np.ndarray,
    chunk_bf: int,
    packed_columns: bool = False,
) -> np.ndarray:
    """Evaluate a small-scan exact pair while sharing row-stage inputs."""
    mx = prepared.mx
    batch = int(c10_np.size)
    c10_values = mx.array(c10_np, dtype=mx.float32)
    c12_values = mx.array(c12_np, dtype=mx.float32)
    # Match the scalar exact path's Python-float trig followed by float32 cast.
    cos2phi12_values = mx.array(
        [math.cos(2.0 * float(phi)) for phi in phi_np],
        dtype=mx.float32,
    )
    sin2phi12_values = mx.array(
        [math.sin(2.0 * float(phi)) for phi in phi_np],
        dtype=mx.float32,
    )
    phase_sum = mx.zeros((batch, *prepared.scan_shape), dtype=mx.float32)
    phase_sumsq = mx.zeros((batch,), dtype=mx.float32)
    phase_col_k_bf = default_phase_col_k_bf(prepared.scan_shape)
    storage_num_bf = int(prepared.g_qk.shape[0])
    pk_all = pk_batch_from_prepared(
        prepared,
        start=0,
        stop=storage_num_bf,
        c10=c10_values,
        c12=c12_values,
        cos2phi12=cos2phi12_values,
        sin2phi12=sin2phi12_values,
    )

    storage_chunks = list(bf_storage_chunks(prepared, chunk_bf))
    submission_depth = 1 if prepared.scan_shape == (1024, 1024) else 6
    for chunk_index, (start, stop) in enumerate(storage_chunks):
        row_ifft = row_ifft_small_batch_from_dynamic_geometry(
            prepared,
            start=start,
            stop=stop,
            c10=c10_values,
            c12=c12_values,
            cos2phi12=cos2phi12_values,
            sin2phi12=sin2phi12_values,
            pk_override=pk_all[:, start:stop],
            rows_per_group=1,
        )
        chunk_sum, chunk_sumsq = (
            phase_cols_small_scalar_loss_batch_from_row_ifft(
                mx,
                row_ifft,
                k_bf=phase_col_k_bf,
                packed_pair=packed_columns,
            )
        )
        phase_sum = phase_sum + chunk_sum
        phase_sumsq = phase_sumsq + chunk_sumsq
        # Keep sequential float32 additions while bounding live row storage.
        # A 1024 pair row buffer is about 8 GiB, so submit each chunk before
        # constructing the next; smaller sizes retain the measured depth six.
        if (
            chunk_index % submission_depth == submission_depth - 1
            or chunk_index + 1 == len(storage_chunks)
        ):
            mx.eval(phase_sum, phase_sumsq)

    mean_phase = phase_sum / prepared.num_bf
    mean_sq = mx.stack(
        [
            mx.mean(mean_phase[candidate] * mean_phase[candidate])
            for candidate in range(batch)
        ]
    )
    norm = float(prepared.num_bf * prepared.scan_shape[0] * prepared.scan_shape[1])
    losses = phase_sumsq / norm - mean_sq
    mx.eval(losses)
    return np.asarray(losses).astype(np.float32, copy=False)


def _effective_exact_batch_chunk_bf(
    chunk_bf: int,
    scan_shape: tuple[int, int],
    batch: int,
) -> int:
    """Choose an MPS exact-loss chunk that leaves room for candidate batches."""
    requested = max(1, int(chunk_bf))
    batch = max(1, int(batch))
    if batch <= 1:
        return requested
    if tuple(scan_shape) == (512, 512):
        if batch >= 8:
            return min(requested, 256)
        if batch >= 4:
            return min(requested, 512)
        return min(requested, 1024)
    return requested


@lru_cache(maxsize=1)
def _small_exact_executor() -> ThreadPoolExecutor:
    """Persistent workers whose MLX streams are created in their own threads."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="mps-ssb-exact")


class _WorkerStream(threading.local):
    """The MLX stream of one exact-loss worker thread, created on that thread's first loss."""

    stream = None


_worker_stream = _WorkerStream()


def _small_exact_loss_worker(
    prepared: PreparedMpsSSB,
    c10: float,
    c12: float,
    phi12: float,
    chunk_bf: int,
) -> float:
    """Submit one unchanged exact loss path on a worker-local MLX stream."""
    mx = prepared.mx
    if _worker_stream.stream is None:
        _worker_stream.stream = mx.new_stream(mx.gpu)
    with mx.stream(_worker_stream.stream):
        _object_wave, loss, _phase = reconstruct_prepared(
            prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=chunk_bf,
            compute_loss=True,
            compute_object=False,
            return_phase=False,
        )
    if loss is None:
        raise RuntimeError("Exact MPS SSB objective did not return a loss.")
    return float(loss)
