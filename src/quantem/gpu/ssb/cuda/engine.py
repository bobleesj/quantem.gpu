"""The CUDA engine for single-sideband ptychography.

The engine holds the bright-field Fourier evidence ``G_qk`` on the device and
evaluates corrected phases, the phase-variance loss and (through
``ThickSample``) the thick-sample model for square scans of 128, 256, 512 or
1024 positions per side.
"""

import math
import pathlib
import time
from types import TracebackType
from typing import Self

import cupy as cp
import numpy as np

from quantem.gpu.io.qem import save_streamed
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.ssb.brightfield import BrightfieldDisk
from quantem.gpu.ssb.contract import SSBExportState
from quantem.gpu.ssb.cuda.kernels import get_fft_kernel
from quantem.gpu.ssb.cuda.kernels.engine import (
    accumulate_phase_moments,
    pk_kernel,
    sum_sumsq_phase_kernel,
)
from quantem.gpu.ssb.cuda.thick import ThickSample

# Above this many bytes of corrected planes (num_bright_field_pixels x scan_row x scan_col complex64) the full-BF staging buffer is never
# allocated: the paths reduce bright-field chunks whose staging buffer targets _CHUNK_TARGET_BYTES instead.
_STAGING_LIMIT_BYTES = 6 * 1024 ** 3
_CHUNK_TARGET_BYTES = 2 * 1024 ** 3


class _PreparedCudaBfSubset:
    """Reusable CUDA-owned state for an explicitly approximate drag preview.

    ``engine`` is the :class:`SSBEngine` whose full-BF state this subset
    swaps out while it is entered and restores on exit. What does not depend
    on the rotation (the subset's rows of G, detector indices and work
    buffers) is made once; the per-pixel geometry is taken from the engine's
    current rotation on entry, so a ``set_rotation`` between two uses is
    followed rather than undone on exit.
    """

    def __init__(self, engine, num_bright_field_pixels: int) -> None:
        self._engine = engine
        full_num_bf = engine.num_bright_field_pixels
        self.num_bright_field_pixels = max(1, min(int(num_bright_field_pixels), full_num_bf))
        self._active = False
        self._full = {}
        self._subset = {}
        # the engine geometry the subset's cache was taken from
        self._geometry_source = None
        if self.num_bright_field_pixels == full_num_bf:
            # All BF pixels: the "subset" is the session itself and nothing is swapped. Copying G and a full result
            # buffer would duplicate ~2 x (BF x scan) complex64 (37 GB at 512 x 512) for an identical preview.
            return
        step = max(1, full_num_bf // self.num_bright_field_pixels)
        self._indices = cp.arange(0, full_num_bf, step, dtype=cp.int64)[:self.num_bright_field_pixels]
        ny, nx = engine.scan_shape
        self._subset = {
            "G_qk": engine.G_qk[self._indices],
            "bf_inds_row": engine.bf_inds_row[self._indices],
            "bf_inds_col": engine.bf_inds_col[self._indices],
            "pk_buffer": cp.empty((self.num_bright_field_pixels,), dtype=cp.complex64),
            "result_buffer": cp.empty((self.num_bright_field_pixels, ny, nx), dtype=cp.complex64),
            "mean_phase_buffer": cp.empty((ny, nx), dtype=cp.float32),
        }

    def __enter__(self) -> Self:
        if self._active:
            raise RuntimeError("The prepared SSB BF subset is already active.")
        self._active = True
        if not self._subset:
            return self
        engine = self._engine
        if self._geometry_source is not engine._cache:
            cache = dict(engine._cache)
            for key in ("kx_bf", "ky_bf", "alpha_k2_1d", "cos2phi_k_1d", "sin2phi_k_1d", "aperture_k_1d"):
                cache[key] = cp.ascontiguousarray(engine._cache[key][self._indices])
            cache["num_bright_field_pixels"] = self.num_bright_field_pixels
            self._subset["cache"] = cache
            self._geometry_source = engine._cache
        self._full = {
            "G_qk": engine.G_qk,
            "bf_inds_row": engine.bf_inds_row,
            "bf_inds_col": engine.bf_inds_col,
            "cache": engine._cache,
            "pk_buffer": engine._pk_buffer,
            "result_buffer": engine._result_buffer,
            "mean_phase_buffer": engine._mean_phase_buffer,
        }
        subset = self._subset
        engine.G_qk = subset["G_qk"]
        engine.bf_inds_row = subset["bf_inds_row"]
        engine.bf_inds_col = subset["bf_inds_col"]
        engine._cache = subset["cache"]
        engine._pk_buffer = subset["pk_buffer"]
        engine._result_buffer = subset["result_buffer"]
        engine._corrected_buffer = None
        engine._mean_phase_buffer = subset["mean_phase_buffer"]
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        engine = self._engine
        if not self._active:
            return
        self._active = False
        if not self._subset:
            return
        self._subset["result_buffer"] = engine._result_buffer
        self._subset["mean_phase_buffer"] = engine._mean_phase_buffer
        full = self._full
        engine.G_qk = full["G_qk"]
        engine.bf_inds_row = full["bf_inds_row"]
        engine.bf_inds_col = full["bf_inds_col"]
        engine._cache = full["cache"]
        engine._pk_buffer = full["pk_buffer"]
        engine._result_buffer = full["result_buffer"]
        engine._corrected_buffer = None
        engine._mean_phase_buffer = full["mean_phase_buffer"]

    def close(self) -> None:
        """Release the subset buffers after restoring full-BF state."""

        if self._active:
            self.__exit__(None, None, None)
        self._subset.clear()
        self._full.clear()


class SSBEngine:
    """
    CuPy-accelerated SSB computation with fused CUDA kernels.

    All computation is done on GPU using CuPy arrays.
    Pre-computes rotation-dependent quantities and caches them for
    fast aberration-dependent gamma factor computation.
    """

    def __init__(
        self,
        G_qk: cp.ndarray,
        bf_inds_row: cp.ndarray,
        bf_inds_col: cp.ndarray,
        bright_field_center_px: tuple[float, float] | None,
        dc_value: complex,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        q_row: cp.ndarray,
        q_col: cp.ndarray,
        wavelength: float,
        semiangle_cutoff: float,
        angular_sampling: tuple[float, float],
        bright_field_radius_px: float | None = None,
    ):
        n_bf = int(G_qk.shape[0])
        n_row = int(bf_inds_row.shape[0])
        n_col = int(bf_inds_col.shape[0])
        if n_bf != n_row or n_bf != n_col:
            raise ValueError(
                "G_qk first dimension must match bf_inds_row and bf_inds_col "
                f"lengths; got G_qk.shape[0]={n_bf}, "
                f"len(row)={n_row}, len(col)={n_col}."
            )
        self.G_qk = G_qk
        self.bf_inds_row = bf_inds_row
        self.bf_inds_col = bf_inds_col
        if bright_field_center_px is None:
            bright_field_center_px = ((gpts[0] - 1) * 0.5, (gpts[1] - 1) * 0.5)
        self.bright_field_center_px = bright_field_center_px
        self.bright_field_radius_px = bright_field_radius_px
        self._dc_value_host = complex(dc_value)
        self.gpts = gpts
        self.sampling = sampling
        self.q_row = q_row
        self.q_col = q_col
        self.wavelength = wavelength
        self.semiangle_cutoff = semiangle_cutoff
        self.angular_sampling = angular_sampling
        # Pre-computed cache
        self._cached_scan_detector_rotation_rad = None
        self._cache = {}
        # Pre-compute factor for aberration phase
        self._factor = float(math.pi / wavelength)  # = (2*pi/wl) * 0.5
        # Work buffers (sized in cache_rotation)
        self._result_buffer = None
        self._corrected_buffer = None
        self._mean_phase_buffer = None
        self._pk_buffer = None
        self._sum_buffer = None
        self._sumsq_buffer = None
        self._partial_sum = None
        self._partial_sumsq = None
        # Custom FFT (initialized in cache_rotation)
        self._custom_fft = None
        self._bf_source_path: pathlib.Path | None = None
        self._bf_source_dtype: np.dtype | None = None
        self._bf_source_write_seconds: float | None = None
        self._colvar_group = 32
        self.thick = ThickSample(self)

    def gqk_is_half_plane(self) -> bool:
        """Return True when ``G_qk`` stores only the Hermitian half-plane."""
        if self.G_qk.ndim != 3:
            return False
        nx = int(self.q_col.shape[1])
        return int(self.G_qk.shape[2]) == nx // 2 + 1

    @property
    def num_bright_field_pixels(self) -> int:
        """Number of bright-field pixels."""
        return int(self.bf_inds_row.size)

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Reconstruction grid shape in public ``(row, col)`` order."""

        return int(self.q_row.shape[0]), int(self.q_row.shape[1])

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector grid shape (n_k_row, n_k_col)."""

        return int(self.gpts[0]), int(self.gpts[1])

    @property
    def bf_source_write_seconds(self) -> float:
        """Duration of the most recent exact BF-source write."""

        if self._bf_source_write_seconds is None:
            raise RuntimeError("No exact BF source has been written yet.")
        return self._bf_source_write_seconds

    def export_state(self) -> SSBExportState:
        """Return backend-neutral host metadata for a WebGPU consumer."""

        cache = self._cache
        rows = cp.asnumpy(self.bf_inds_row).astype(np.int32, copy=False)
        cols = cp.asnumpy(self.bf_inds_col).astype(np.int32, copy=False)
        center = (float(self.bright_field_center_px[0]), float(self.bright_field_center_px[1]))
        distance_sq = (rows.astype(np.float64) - center[0]) ** 2
        distance_sq += (cols.astype(np.float64) - center[1]) ** 2
        # Replaying a fit must select the same edge pixels. A padded radius
        # inferred from selected pixels can admit their unselected neighbours.
        radius = self.bright_field_radius_px
        if radius is None:
            radius = float(np.sqrt(distance_sq).max()) + 1e-3
        detector_sampling = 0.5 * (
            float(self.angular_sampling[0]) + float(self.angular_sampling[1])
        )
        # the disk radius this calibration implies: detector_sampling_mrad = semiangle / radius
        detected_radius = float(self.semiangle_cutoff) / detector_sampling
        selection = BrightfieldDisk(
            rows=rows,
            cols=cols,
            bright_field_center_px=center,
            bright_field_radius_px=radius,
            bright_field_edge_radius_px=detected_radius,
            detector_shape=self.detector_shape,
        )

        return SSBExportState(
            backend="cuda",
            scan_shape=self.scan_shape,
            brightfield=selection,
            kx_bf=cp.asnumpy(cache["kx_bf"]),
            ky_bf=cp.asnumpy(cache["ky_bf"]),
            qx_1d=cp.asnumpy(cache["qx_1d"]),
            qy_1d=cp.asnumpy(cache["qy_1d"]),
            aperture_k=cp.asnumpy(cache["aperture_k_1d"]),
            alpha_k2=cp.asnumpy(cache["alpha_k2_1d"]),
            cos2phi_k=cp.asnumpy(cache["cos2phi_k_1d"]),
            sin2phi_k=cp.asnumpy(cache["sin2phi_k_1d"]),
            wavelength_A=float(self.wavelength),
            semiangle_rad=float(cache["semiangle_rad"]),
            angular_sampling_rad=(
                float(cache["ang_y_rad"]),
                float(cache["ang_x_rad"]),
            ),
            sampling_A=(float(self.sampling[0]), float(self.sampling[1])),
            dc_value=complex(self._dc_value_host),
            bf_source_path=self._bf_source_path,
            bf_source_dtype=self._bf_source_dtype,
        )

    def write_exact_bf_source(
        self,
        data,
        path_stem: str | pathlib.Path,
    ) -> pathlib.Path:
        """Save the selected, unchanged BF counts in a lossless ANS QEM file.

        Encoding uses the shared CUDA acquisition codec in bounded scan chunks.
        The QEM detector axis enumerates the selected BF pixels; their original
        row/column coordinates remain in the SSB calibration.
        """
        started = time.perf_counter()
        data = cp.asarray(data)
        if data.ndim != 4 or tuple(data.shape[:2]) != self.scan_shape:
            raise ValueError("BF export requires the same scan shape as the SSB session.")
        if data.dtype not in {cp.dtype(cp.uint8), cp.dtype(cp.uint16)}:
            raise TypeError("Lossless BF export requires uint8 or uint16 detector counts.")
        path = pathlib.Path(path_stem).with_suffix(".qem").expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"{path} already exists; choose a new BF export path.")
        with cp.cuda.Device(data.device.id):
            source = StreamedCounts((*self.scan_shape, 1, self.num_bright_field_pixels), data.dtype)
            flat = data.reshape(-1, *self.detector_shape)
            try:
                for first in range(0, flat.shape[0], 512):
                    selected = cp.ascontiguousarray(
                        flat[first:first + 512, self.bf_inds_row, self.bf_inds_col]
                    ).reshape(-1, 1, self.num_bright_field_pixels)
                    source.append(selected)
                save_streamed(path, source, {"selection": "SSB bright-field pixels"})
            finally:
                source.release()
        self._bf_source_path = path
        self._bf_source_dtype = np.dtype(data.dtype)
        self._bf_source_write_seconds = time.perf_counter() - started
        return path

    def prepare_bf_subset(self, num_bright_field_pixels: int) -> _PreparedCudaBfSubset:
        """Prepare a reusable deterministic BF subset for drag previews."""

        return _PreparedCudaBfSubset(self, num_bright_field_pixels)

    def cache_rotation(
        self,
        scan_detector_rotation_rad: float,
        force: bool = False,
    ) -> None:
        """Pre-compute all rotation-dependent quantities."""
        if self._cached_scan_detector_rotation_rad == scan_detector_rotation_rad and not force:
            return
        # Compute detector k-space coordinates centered on the BF disk.
        recip_row = 1.0 / (self.sampling[0] * self.gpts[0])
        recip_col = 1.0 / (self.sampling[1] * self.gpts[1])
        row_offsets = cp.arange(self.gpts[0], dtype=cp.float32) - self.bright_field_center_px[0]
        col_offsets = cp.arange(self.gpts[1], dtype=cp.float32) - self.bright_field_center_px[1]
        kxa = row_offsets[:, None] * recip_row
        kya = col_offsets[None, :] * recip_col
        # Passive rotation
        if scan_detector_rotation_rad is not None:
            cos_a = math.cos(-scan_detector_rotation_rad)
            sin_a = math.sin(-scan_detector_rotation_rad)
            kxa_rot = kxa * cos_a + kya * sin_a
            kya_rot = -kxa * sin_a + kya * cos_a
            kxa, kya = kxa_rot, kya_rot
        kx_bf = kxa[self.bf_inds_row, self.bf_inds_col]
        ky_bf = kya[self.bf_inds_row, self.bf_inds_col]
        num_bright_field_pixels = int(kx_bf.shape[0])
        ny, nx = self.q_row.shape
        # Soft aperture (constant, doesn't depend on aberrations)
        semiangle_rad = self.semiangle_cutoff * 1e-3
        ang_y, ang_x = self.angular_sampling
        # Extract 1D q-space coordinates (separable from meshgrid)
        qx_1d = cp.ascontiguousarray(self.q_row[:, 0].astype(cp.float32))
        qy_1d = cp.ascontiguousarray(self.q_col[0, :].astype(cp.float32))
        # Probe at k positions
        k_mag = cp.sqrt(kx_bf**2 + ky_bf**2)
        phi_k = cp.arctan2(ky_bf, kx_bf)
        alpha_k = k_mag * self.wavelength
        alpha_k2 = alpha_k**2
        cos_phi_k = cp.cos(phi_k)
        sin_phi_k = cp.sin(phi_k)
        denom_k = cp.sqrt(
            (cos_phi_k * ang_y * 1e-3)**2 +
            (sin_phi_k * ang_x * 1e-3)**2
        )
        aperture_k = cp.clip((semiangle_rad - alpha_k) / denom_k + 0.5, 0, 1).astype(cp.float32)
        cos2phi_k = cos_phi_k * cos_phi_k - sin_phi_k * sin_phi_k
        sin2phi_k = 2.0 * sin_phi_k * cos_phi_k
        del cos_phi_k, sin_phi_k, denom_k, phi_k
        # the kernels read the per-pixel geometry as contiguous float32 arrays
        cache = {
            "num_bright_field_pixels": num_bright_field_pixels,
            "ny": ny,
            "nx": nx,
            "alpha_k2_1d": cp.ascontiguousarray(alpha_k2.astype(cp.float32)),
            "cos2phi_k_1d": cp.ascontiguousarray(cos2phi_k.astype(cp.float32)),
            "sin2phi_k_1d": cp.ascontiguousarray(sin2phi_k.astype(cp.float32)),
            "aperture_k_1d": cp.ascontiguousarray(aperture_k),
            "kx_bf": cp.ascontiguousarray(kx_bf.astype(cp.float32)),
            "ky_bf": cp.ascontiguousarray(ky_bf.astype(cp.float32)),
            "qx_1d": qx_1d,
            "qy_1d": qy_1d,
            "wavelength": float(self.wavelength),
            "semiangle_rad": float(semiangle_rad),
            "ang_y_rad": float(ang_y * 1e-3),
            "ang_x_rad": float(ang_x * 1e-3),
        }
        self._cache = cache
        self._cached_scan_detector_rotation_rad = scan_detector_rotation_rad
        if self._custom_fft is None:
            if ny != nx:
                raise ValueError(
                    "CUDA SSB requires a square scan grid; "
                    f"got {ny}x{nx}."
                )
            self._custom_fft = get_fft_kernel(ny)
            self._colvar_group = self._custom_fft._colvar_group
        # Coherent reconstruction does not need per-BF image staging.
        # Other paths size it on demand; retain an existing buffer across
        # rotations so repeated phase/loss previews do not reallocate it.
        self._corrected_buffer = None
        self._mean_phase_buffer = None
        self._pk_buffer = cp.empty((num_bright_field_pixels,), dtype=cp.complex64)
        self._sum_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._sumsq_buffer = cp.empty((ny, nx), dtype=cp.float32)
        cp.get_default_memory_pool().free_all_blocks()

    # =====================================================================
    #  Reconstruction
    # =====================================================================

    def _fill_probe(self, C10: float, C12: float, phi12: float) -> tuple[float, float]:
        """Write the probe at every bright-field pixel, P(k) = A(k) exp(-i chi(k)), into ``_pk_buffer``.

        Every correction path multiplies G_qk by conj(P(k)) and re-evaluates P(q -/+ k) on the fly from the same two
        trigonometric factors, so this returns (cos 2 phi12, sin 2 phi12) for those kernels.
        """
        cache = self._cache
        probe = self._probe_buffer(int(cache["num_bright_field_pixels"]))
        cos2phi12 = math.cos(2.0 * phi12)
        sin2phi12 = math.sin(2.0 * phi12)
        pk_kernel(
            cache["alpha_k2_1d"], cache["cos2phi_k_1d"], cache["sin2phi_k_1d"], cache["aperture_k_1d"],
            cp.float32(C10), cp.float32(C12), cp.float32(cos2phi12), cp.float32(sin2phi12),
            cp.float32(self._factor), probe,
        )
        return cos2phi12, sin2phi12

    def _run_correction_pipeline(self, C10: float, C12: float, phi12: float) -> None:
        """Run the aberration-correction pipeline, populating _corrected_buffer."""
        self._size_result_buffer(*self._bf_grid())
        cos2phi12, sin2phi12 = self._fill_probe(C10, C12, phi12)
        self._custom_fft.ifft2_inplace_fused_pk(
            self._result_buffer,
            self.G_qk,
            self._cache,
            self._pk_buffer,
            C10,
            C12,
            cos2phi12,
            sin2phi12,
            self._factor,
            self._dc_value_host,
        )
        self._corrected_buffer = self._result_buffer

    # =====================================================================
    #  Fused reconstruct + variance loss (single pipeline pass)
    # =====================================================================

    def reconstruct_with_loss(self, C10: float, C12: float, phi12: float) -> tuple[cp.ndarray, float]:
        """Mean phase and the full-IFFT phase-variance loss in one pass: (mean_phase (ny, nx), loss).

        Large scans (> 6 GB of corrected planes) run the chunked fused path, which writes one partial plane per
        reduction group and merges them in a fixed order: fits, results and previews get the same bits every time.
        """
        num_bright_field_pixels, ny, nx = self._bf_grid()
        if num_bright_field_pixels * ny * nx * 8 > _STAGING_LIMIT_BYTES:
            return self._fused_chunked_core(C10, C12, phi12)
        self._run_correction_pipeline(C10, C12, phi12)
        return self._mean_phase_and_loss_of_corrected(num_bright_field_pixels, ny, nx)

    def _fused_chunked_core(
        self,
        C10: float, C12: float, phi12: float,
        *,
        chunk_bf: int | None = None,
    ) -> tuple[cp.ndarray, float]:
        """``reconstruct_with_loss`` of a large scan by chunks, with the fused col-FFT + phase accumulate.

        Returns (mean phase, loss). Every chunk writes one partial plane per 32-BF group and the planes are summed in a
        fixed order, so the result does not depend on the order the device runs the groups in (atomic accumulation would).
        ``chunk_bf`` (BF pixels per chunk) is chosen from the scan size and free memory unless given.
        """
        num_bright_field_pixels, ny, nx = self._bf_grid()

        cos2phi12, sin2phi12 = self._fill_probe(C10, C12, phi12)

        bytes_per_bf = ny * nx * 8
        if chunk_bf is None or not 0 < chunk_bf < num_bright_field_pixels:
            if ny == 512 and nx == 512 and num_bright_field_pixels > 64:
                # Keep the staged row-IFFT producer/consumer working set small.
                # For full-BF 512 SSB, 64 BF chunks avoid the slow 18+ GB
                # write-then-read pass while preserving exact phase/loss accumulation.
                chunk_bf = 64
            elif ny == 1024 and nx == 1024 and num_bright_field_pixels > 1024:
                # Native 1024 exact phase/loss has the same redraw time at a
                # smaller staging chunk but avoids the 60+ GB transient pool.
                # This is a memory-footprint default, not a FPS breakthrough.
                chunk_bf = 1024
            else:
                free_bytes = cp.cuda.runtime.memGetInfo()[0]
                if self._result_buffer is not None:
                    free_bytes += int(self._result_buffer.nbytes)
                target_bytes = min(int(free_bytes * 0.45), 24 * 1024 ** 3)
                chunk_bf = max(1, _CHUNK_TARGET_BYTES // bytes_per_bf)
                chunk_bf = min(num_bright_field_pixels, max(chunk_bf, max(1, target_bytes // bytes_per_bf)))
        self._size_result_buffer(chunk_bf, ny, nx)

        k_bf = self._colvar_group
        max_groups = (chunk_bf + k_bf - 1) // k_bf
        partial_shape = (max_groups, ny, nx)
        if self._partial_sum is None or self._partial_sum.shape != partial_shape:
            self._partial_sum = cp.empty(partial_shape, dtype=cp.float32)
        if self._partial_sumsq is None or self._partial_sumsq.shape != partial_shape:
            self._partial_sumsq = cp.empty(partial_shape, dtype=cp.float32)

        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        phase_sumsq = cp.zeros((ny, nx), dtype=cp.float32)

        for bf_start in range(0, num_bright_field_pixels, chunk_bf):
            bf_end = min(bf_start + chunk_bf, num_bright_field_pixels)
            chunk = bf_end - bf_start
            chunk_cache = self._chunk_cache(bf_start, bf_end)
            chunk_planes = self._result_buffer[:chunk]
            n_groups = (chunk + k_bf - 1) // k_bf
            self._custom_fft.ifft2_fused_pk_col_accumulate(
                chunk_planes,
                self.G_qk[bf_start:bf_end],
                chunk_cache,
                self._pk_buffer[bf_start:bf_end],
                C10, C12, cos2phi12, sin2phi12,
                self._factor, self._dc_value_host,
                self._partial_sum[:n_groups],
                self._partial_sumsq[:n_groups],
                k_bf,
            )
            if n_groups <= 2:
                accumulate_phase_moments(
                    self._partial_sum, self._partial_sumsq,
                    np.int32(n_groups), np.int32(ny * nx),
                    phase_sum, phase_sumsq, phase_sum, phase_sumsq,
                )
            else:
                phase_sum += self._partial_sum[:n_groups].sum(axis=0)
                phase_sumsq += self._partial_sumsq[:n_groups].sum(axis=0)

        mean_phase = phase_sum / float(num_bright_field_pixels)
        var_per_pixel = phase_sumsq / float(num_bright_field_pixels) - mean_phase ** 2
        loss = float(cp.mean(var_per_pixel))
        return self._row_col_phase(mean_phase), loss

    def _row_col_phase(self, mean_phase: cp.ndarray) -> cp.ndarray:
        """Return a chunked-core phase image in the public (row, col) scan order.

        The 512 fused column-IFFT kernels accumulate each pixel at ``col * 512 + row``, so
        their summed plane is the transpose of the scan-frame image that the small-scan pipeline, the thick-sample path
        and the 128/256/1024 kernels return. The loss is a mean over pixels and does not depend on
        the order, so only the phase image is transposed back. Without this, previews of 512 x 512 scans appear transposed
        next to the fitted result.
        """
        if self._custom_fft._size == 512:
            return cp.ascontiguousarray(mean_phase.T)
        return mean_phase

    def release_reconstruction_buffers(self) -> None:
        """Drop the full-BF reconstruction buffers; ``free`` returns their memory."""
        self._result_buffer = None
        self._corrected_buffer = None

    # =====================================================================
    #  Buffers and launches shared by the reconstruction paths
    # =====================================================================

    def _bf_grid(self) -> tuple[int, int, int]:
        """Return ``(num_bright_field_pixels, ny, nx)`` of the cached rotation geometry, the shape of the stack of corrected planes."""
        cache = self._cache
        return int(cache["num_bright_field_pixels"]), int(cache["ny"]), int(cache["nx"])

    def _probe_buffer(self, num_bright_field_pixels: int) -> cp.ndarray:
        """Return ``_pk_buffer`` sized to ``num_bright_field_pixels`` probe values, reallocated only when the BF count changes."""
        if self._pk_buffer is None or self._pk_buffer.shape != (num_bright_field_pixels,):
            self._pk_buffer = cp.empty((num_bright_field_pixels,), dtype=cp.complex64)
        return self._pk_buffer

    def _size_result_buffer(self, count: int, ny: int, nx: int) -> None:
        """Size ``_result_buffer`` to ``count`` corrected planes.

        A larger buffer is dropped before the new one is allocated, so a full-BF staging buffer and the chunk buffer
        that replaces it on a large scan never hold device memory at the same time.
        """
        if self._result_buffer is not None and self._result_buffer.shape[0] > count:
            self._result_buffer = None
        shape = (count, ny, nx)
        if self._result_buffer is None or self._result_buffer.shape != shape:
            self._result_buffer = cp.empty(shape, dtype=cp.complex64)

    def _chunk_cache(self, bf_start: int, bf_end: int) -> dict:
        """The geometry cache with its per-pixel k vectors cut to bright-field pixels ``bf_start:bf_end``.

        The fused kernels read every other entry (q grids, wavelength, angles) unchanged, so a chunk shares them.
        """
        cache = self._cache
        return {**cache, "kx_bf": cache["kx_bf"][bf_start:bf_end], "ky_bf": cache["ky_bf"][bf_start:bf_end]}

    def _mean_phase_and_loss_of_corrected(self, num_bright_field_pixels: int, ny: int, nx: int) -> tuple[cp.ndarray, float]:
        """Mean phase and phase-variance loss of ``_corrected_buffer`` from one pass over the BF axis.

        The kernel writes the per-pixel sum and sum of squares of the BF phases: mean = sum / N, variance per pixel =
        sumsq / N - mean^2, and the loss is the mean variance over scan pixels.
        """
        if self._mean_phase_buffer is None:
            self._mean_phase_buffer = cp.empty((ny, nx), dtype=cp.float32)
        if self._sum_buffer is None:
            self._sum_buffer = cp.empty((ny, nx), dtype=cp.float32)
        if self._sumsq_buffer is None:
            self._sumsq_buffer = cp.empty((ny, nx), dtype=cp.float32)
        block = 256
        grid = (ny * nx + block - 1) // block
        sum_sumsq_phase_kernel(
            (grid,), (block,),
            (self._corrected_buffer, self._sum_buffer, self._sumsq_buffer, np.int32(num_bright_field_pixels), np.int32(ny), np.int32(nx)),
        )
        cp.divide(self._sum_buffer, float(num_bright_field_pixels), out=self._mean_phase_buffer)
        var_per_pixel = self._sumsq_buffer / float(num_bright_field_pixels) - self._mean_phase_buffer ** 2
        loss = float(cp.mean(var_per_pixel))
        return self._mean_phase_buffer, loss
