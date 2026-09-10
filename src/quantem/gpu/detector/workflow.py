"""Virtual detectors (bright / annular-dark / dark field) for 4D-STEM.

Primary API - place a virtual detector on 4D-STEM data and get its image, with
collection angles in **mrad**::

    from quantem.gpu.io import load
    from quantem.widget import Show2D
    data = load("master.h5")
    Show2D(bf(data))                       # bright field (the bright disk)
    Show2D(adf(data))                      # annular dark field (auto band)
    Show2D(adf(data, inner=50, outer=180)) # collection angles in mrad
    Show2D(df(data))                       # outside the bright disk

``bf`` / ``adf`` / ``df`` are thin geometry over the shared compute backend: they
build a boolean detector mask and call :func:`masked_sum` - the same fast
reduction that Show4DSTEM and live Browse use. The probe (disk center + size)
auto-fits from the mean diffraction pattern. MacBook (MPS) runs the raw-Metal
masked-sum over chunked uint8/uint16/uint32 buffers; CUDA runs the CuPy
RawKernel reducer for resident uint8/uint16/uint32 arrays; Torch/NumPy are
fallback paths for small or unsupported arrays. **No binning** on either GPU
path.

The lower-level :func:`virtual` function (below) is mode-based
(DP/BF/ABF/ADF/HAADF/DF, bands measured in the auto-detected disk radius) and is
mainly the reference path the parity tests pin; ``ds.bf()`` etc. are the API.
"""

from __future__ import annotations

import numpy as np

from quantem.gpu.io.uint4 import is_packed_uint4


class DetectorSession:
    """Prepared, cache-owning detector compute session.

    Widget and live callers use this backend-neutral object instead of
    importing CUDA or Metal implementation modules.
    """

    def __init__(self, data) -> None:
        self._backend = _resolve_backend(data)

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Scan shape in public ``(row, col)`` order."""

        return tuple(int(value) for value in self._backend.scan_shape)

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector shape in public ``(row, col)`` order."""

        return tuple(int(value) for value in self._backend.det_shape)

    @property
    def num_frames(self) -> int:
        """Number of scan positions per acquisition in the prepared data."""

        return int(self._backend.n_frames)

    @property
    def series_shape(self) -> tuple[int, ...]:
        """Leading acquisition dimensions; empty for a single 4D acquisition."""
        return tuple(getattr(self._backend, "series_shape", ()))

    @property
    def backend_metadata(self) -> dict:
        """Scientific format and implementation identity, when supplied."""
        return dict(getattr(self._backend, "backend_metadata", {}))

    @property
    def timings(self) -> dict:
        """Completed backend timings for the last request, excluding display."""
        return dict(getattr(self._backend, "last", {}))

    @property
    def detector_validity(self) -> np.ndarray | None:
        """Copy the stored detector-validity mask when the source provides it."""
        valid = getattr(self._backend, "valid_pixels", None)
        return None if valid is None else np.array(valid, dtype=bool, copy=True)

    def frame(self, index: int, *, output: str = "numpy", out=None, wait: bool = True):
        """Return one detector frame.

        Parameters
        ----------
        index
            Flat scan index in row-major order. A compact series decodes this
            position across every acquisition in one calculation.
        output
            ``"numpy"`` preserves the host-returning default. ``"native"``
            requests a supported device-native result without a host copy.
        out
            Optional contiguous device buffer, only with ``output="native"``.
            Point patterns retain uint8/uint16 with leading ``series_shape``.
            The operation finishes before returning; do not reuse ``out`` while
            a consumer still reads it. Without ``out``, the result owns storage.

        Examples
        --------
        >>> patterns = session.frame(256 * 512 + 256, output="native")
        """
        _check_output(output, out)
        index = int(index)
        if not 0 <= index < self.num_frames:
            raise IndexError(
                f"Scan index {index} is outside {self.num_frames} frames; "
                "use a nonnegative row-major index within the loaded scan."
            )
        native = getattr(self._backend, "frame_native", None)
        if native is not None:
            result = native(index, out=out) if wait else native(index, out=out, wait=False)
            return result if output == "native" else result.get()
        if output == "native":
            raise NotImplementedError("This backend has no native frame output; use output='numpy'.")
        return np.array(self._backend.frame(index), copy=True)

    def reduce_frames(self, indices, mode: str = "mean") -> np.ndarray:
        """Reduce selected scan frames with ``mean``, ``sum``, or ``max``."""

        return np.asarray(self._backend.reduce_frames(indices, reduce=mode))

    def reduce_frames_exact(self, indices) -> np.ndarray:
        """Return the exact uint64 sum of selected scan frames."""

        reducer = getattr(self._backend, "reduce_frames_exact", None)
        if reducer is None:
            raise NotImplementedError(
                "This compute backend has no exact selected-frame reducer."
            )
        return _exact_to_numpy(reducer(indices)).reshape(self.detector_shape)

    def reduce_frames_max(self, indices) -> np.ndarray:
        """Return the exact integer maximum of selected scan frames."""

        reducer = getattr(self._backend, "reduce_frames_max", None)
        if reducer is None:
            raise NotImplementedError(
                "This compute backend has no exact selected-frame maximum."
            )
        return _exact_to_numpy(reducer(indices)).reshape(self.detector_shape)

    def mean_dp(self) -> np.ndarray:
        """Return the float32 mean diffraction pattern."""

        return _reduced_to_numpy(self._backend.mean_dp())

    def finish(self) -> dict:
        """Wait for the oldest query launched with ``wait=False`` and return its timings.

        Raises ``ValueError`` if an encoded stream failed exact decoding, in which
        case that result must not be used.
        """
        return self._backend.finish()

    def masked_sum(self, mask, *, output: str = "numpy", out=None, wait: bool = True, block_stride: int = 1):
        """Return a float32 virtual-detector image for one detector mask.

        Parameters
        ----------
        mask
            Detector mask in ``(row, col)`` order. Compact sources support binary
            masks and preserve the stored detector-validity semantics.
        output
            ``"numpy"`` retains the float32 host default. ``"native"`` for a
            count series returns exact uint32/uint64 counts, shape
            ``(*series_shape, *scan_shape)``, without host conversion.
        out
            Optional contiguous native output array on the source device.
            Only valid with ``output="native"``. The result is complete on
            return. Finish reading it before reusing this buffer. Without
            ``out``, each native result owns separate storage.
        wait
            ``False`` (native output on a streamed CUDA series only) returns as
            soon as the kernels are queued; the result is complete after
            :meth:`finish`, which also raises for a malformed stream. Several
            queries may be in flight so the host plans while the device works.
        block_stride
            Paired native series only. ``k > 1`` sums every k-th 512-scan block
            (every k-th scan row of a 512-wide raster) and leaves the other rows
            of ``out`` untouched, at about ``1/k`` of the device time: a viewer's
            preview of a moving mask on its tiles. The values written are exact.
            Consecutive queries at the same stride build on each other
            incrementally; a change of stride starts from a full plan.

        Examples
        --------
        >>> images = session.masked_sum(mask, output="native")
        """
        _check_output(output, out)
        native = getattr(self._backend, "masked_sum_native", None)
        if block_stride != 1 and (native is None or output != "native" or not hasattr(self._backend, "block_stride")):
            raise ValueError("block_stride needs a paired native series with output='native'.")
        if native is not None:
            extra = {} if block_stride == 1 else dict(block_stride=block_stride)
            result = native(mask, out=out, **extra) if wait else native(mask, out=out, wait=False, **extra)
        elif output == "native":
            raise NotImplementedError("This backend has no native detector output; use output='numpy'.")
        else:
            result = self._backend.masked_sum(mask)
        if output == "native":
            return result
        return _reduced_to_numpy(result).reshape((*self.series_shape, *self.scan_shape))

    def masked_sum_exact(self, mask, *, output: str = "numpy", out=None):
        """Return an exact uint64 virtual-detector image for one mask.

        Parameters
        ----------
        mask
            Full-resolution detector mask. Compact masks must be binary.
        output
            ``"numpy"`` retains the exact uint64 host default. ``"native"``
            returns exact uint32/uint64 counts with leading ``series_shape``.
            The prepared fixed-shape series fits uint32; general CUDA series
            select the accumulator from the native dtype and detector size.
        out
            Optional native buffer with the backend's exact sum dtype and
            shape, following the ownership and completion rules documented
            by :meth:`masked_sum`.

        Examples
        --------
        >>> exact_images = session.masked_sum_exact(mask, output="native")
        """
        _check_output(output, out)
        values = np.asarray(mask)
        if values.shape != self.detector_shape:
            raise ValueError(
                f"Detector mask shape {values.shape} does not match "
                f"{self.detector_shape}; provide one mask value for each "
                "detector (row, column)."
            )
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(
                "Exact detector masks must be binary; provide only 0/1 or "
                "False/True values. Weighted masks require a weighted reducer."
            )
        native = getattr(self._backend, "masked_sum_native", None)
        if native is not None:
            result = native(mask, out=out)
        elif output == "native":
            raise NotImplementedError("This backend has no native exact output; use output='numpy'.")
        else:
            result = self._backend.masked_sum_exact(mask)
        if output == "native":
            return result
        return _exact_to_numpy(result).reshape((*self.series_shape, *self.scan_shape))

    def masked_sum_batch_exact(self, mask) -> np.ndarray:
        """Return all acquisition images as one exact integer batch.

        The supported compact owner returns uint32 with shape
        ``(acquisition, scan_row, scan_col)``. This performs one backend batch
        operation; it never iterates the acquisitions through 4D reducers.

        Parameters
        ----------
        mask
            Full-resolution binary detector mask in ``(row, col)`` order.

        Returns
        -------
        numpy.ndarray
            Independently owned, immutable uint32 image batch.

        Examples
        --------
        >>> images = prepare(resident_data).masked_sum_batch_exact(mask)
        """
        operation = getattr(self._backend, "masked_sum_batch_exact", None)
        if operation is None:
            raise NotImplementedError("This detector backend has no exact all-acquisition batch operation.")
        return operation(mask)

    def frame_batch(self, index: int) -> np.ndarray:
        """Return exact point DPs across all acquisitions.

        Parameters
        ----------
        index
            Flat native scan index, ``scan_row * scan_columns + scan_col``.

        Returns
        -------
        numpy.ndarray
            Immutable uint16 ``(acquisition, det_row, det_col)`` batch.

        Examples
        --------
        >>> patterns = prepare(resident_data).frame_batch(scan_row * 512 + scan_col)
        """
        operation = getattr(self._backend, "frame_batch", None)
        if operation is None:
            raise NotImplementedError("This detector backend has no all-acquisition point-DP operation.")
        return operation(index)

    def center_of_mass(self, mask=None) -> tuple[np.ndarray, np.ndarray]:
        """Return mean-subtracted detector CoM in ``(row, col)`` order."""

        com_col, com_row = self._backend.center_of_mass(mask)
        row = _reduced_to_numpy(com_row).reshape(self.scan_shape)
        col = _reduced_to_numpy(com_col).reshape(self.scan_shape)
        return row, col

    @property
    def supports_fast(self) -> bool:
        """Whether this session provides an accelerated interaction sidecar."""

        return "fast_sidecar" in self._backend.capabilities

    @property
    def fast_ready(self) -> bool:
        """Whether the interaction sidecar is ready."""

        return self.supports_fast and bool(self._backend.has_fast)

    @property
    def fast_bin(self) -> int:
        """Detector binning used only by the optional interaction sidecar."""

        return int(self._backend.fast_bin) if self.supports_fast else 1

    def prepare_fast(self, *, verbose: bool = False) -> bool:
        """Prepare the optional interaction sidecar."""

        if not self.supports_fast:
            return False
        return bool(self._backend.ensure_fast_sidecar(verbose=verbose))

    def cache_fast_presets(self, masks: dict[str, np.ndarray]) -> dict:
        """Cache named interaction masks when supported."""

        if not self.supports_fast:
            return {}
        return self._backend.cache_fast_presets(masks)

    def radial_ready(self, center: tuple[float, float]) -> bool:
        """Return whether exact radial reductions are prepared at ``center``."""

        if "radial_cache" not in self._backend.capabilities:
            return False
        return bool(self._backend.radial_cache_ready(*center))

    def prepare_radial(
        self,
        center: tuple[float, float],
        *,
        idle_delay_s: float = 0.75,
    ) -> None:
        """Prepare exact radial reductions around ``(row, col)``."""

        if "radial_cache" not in self._backend.capabilities:
            return
        self._backend.ensure_radial_cache(
            *center,
            idle_delay_s=float(idle_delay_s),
        )

    def radial_sum(
        self,
        center: tuple[float, float],
        outer_radius: float,
        *,
        inner_radius: float = 0.0,
        build: bool = False,
    ) -> np.ndarray | None:
        """Return an exact cached annular sum, or ``None`` while unavailable."""

        if "radial_cache" not in self._backend.capabilities:
            return None
        result = self._backend.radial_masked_sum(
            center_row=float(center[0]),
            center_col=float(center[1]),
            outer_radius=float(outer_radius),
            inner_radius=float(inner_radius),
            build=bool(build),
        )
        return None if result is None else np.asarray(result, dtype=np.float32)

    @property
    def radial_building(self) -> bool:
        """Whether exact radial cache construction is active."""

        if "radial_cache" not in self._backend.capabilities:
            return False
        return bool(self._backend.radial_building)

    @property
    def radial_error(self) -> str | None:
        """Return the radial cache error, if any."""

        if "radial_cache" not in self._backend.capabilities:
            return None
        return self._backend.radial_error

    def close(self) -> None:
        """Release this session's ownership of backend caches."""

        self._backend = None


def _check_output(output: str, out) -> None:
    if output not in ("numpy", "native"):
        raise ValueError(f"Use output='numpy' or 'native'; got {output!r}.")
    if out is not None and output != "native":
        raise ValueError("out is a device buffer; use output='native' with it.")


def prepare(data) -> DetectorSession:
    """Prepare detector queries over one source or a list of CUDA acquisitions.

    Parameters
    ----------
    data
        A loaded source, array, or list of equally shaped uint8/uint16 CUDA
        acquisitions. A list retains its complete acquisition axis and uses
        one joint kernel launch per native detector or point-pattern query.

    Returns
    -------
    DetectorSession
        A session borrowing the supplied sources and owning query workspaces.

    Examples
    --------
    >>> session = prepare([first_loaded, second_loaded])  # doctest: +SKIP
    >>> images = session.masked_sum(mask, output="native")  # doctest: +SKIP
    """

    return DetectorSession(data)


def _is_cupy_array(data) -> bool:
    return type(data).__module__.split(".", 1)[0] == "cupy"


def _is_torch_tensor(data) -> bool:
    return type(data).__module__.split(".", 1)[0] == "torch"


def _unwrap_core_4dstem(data):
    """Return numeric data from LoadResult or quantem.core Dataset4dstem."""
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        return data.data
    if is_packed_uint4(data):
        return data
    if _is_cupy_array(data) or _is_torch_tensor(data) or isinstance(data, np.ndarray):
        return data
    tensor = getattr(data, "_tensor", None)
    if tensor is not None:
        return tensor
    array = getattr(data, "_array", None)
    if array is not None:
        return array
    array = getattr(data, "array", None)
    if array is not None and not callable(array):
        return array
    return data


def _reduced_to_numpy(data) -> np.ndarray:
    """Convert a small reduced product to float32 NumPy for widget display."""
    if _is_cupy_array(data):
        data = data.get()
    elif _is_torch_tensor(data):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float32)


def _exact_to_numpy(data) -> np.ndarray:
    """Copy a reduced integer product to host without changing its values."""
    if _is_cupy_array(data):
        data = data.get()
    elif _is_torch_tensor(data):
        data = data.detach().cpu().numpy()
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            "Exact detector sums require integer detector data; "
            f"the backend returned {array.dtype}."
        )
    return array.astype(np.uint64, copy=False)


def _flatten_scan(data):
    data = _unwrap_core_4dstem(data)
    if data.ndim == 4:
        return data.reshape(-1, *data.shape[-2:]), (int(data.shape[0]), int(data.shape[1]))
    if data.ndim == 3:
        n = int(data.shape[0])
        side = round(n ** 0.5)
        scan_shape = (side, n // side) if side * side == n else (n,)
        return data, scan_shape
    raise ValueError(
        f"Expected 3D or 4D 4D-STEM data, got {data.ndim}D with shape {data.shape}."
    )


class _ArrayComputeBackend:
    """Small array backend for public detector products."""

    def __init__(self, data):
        self.data = _unwrap_core_4dstem(data)
        self.flat, self.scan_shape = _flatten_scan(self.data)
        self.n_frames = int(self.flat.shape[0])
        self.det_shape = tuple(int(value) for value in self.flat.shape[-2:])
        self.capabilities = ()

    def mean_dp(self):
        if _is_cupy_array(self.flat):
            import cupy as cp

            return self.flat.sum(axis=0, dtype=cp.uint64).astype(cp.float32) / self.n_frames
        if _is_torch_tensor(self.flat):
            import torch

            return self.flat.to(torch.float32).mean(dim=0)
        return (
            np.asarray(self.flat)
            .sum(axis=0, dtype=np.uint64)
            .astype(np.float32)
            / self.n_frames
        )

    def frame(self, index: int) -> np.ndarray:
        frame = self.flat[int(index)]
        if _is_cupy_array(frame):
            return frame.get()
        if _is_torch_tensor(frame):
            return frame.detach().cpu().numpy()
        return np.asarray(frame)

    def reduce_frames(self, scan_indices, reduce: str = "mean") -> np.ndarray:
        selected = self.flat[np.asarray(scan_indices, dtype=np.intp)]
        if reduce == "mean":
            return _reduced_to_numpy(selected.mean(axis=0))
        if reduce == "sum":
            return _reduced_to_numpy(selected.sum(axis=0, dtype=np.uint64))
        if reduce == "max":
            return _reduced_to_numpy(selected.max(axis=0))
        raise ValueError(f"Unknown frame reduction {reduce!r}; use mean, sum, or max.")

    def reduce_frames_exact(self, scan_indices) -> np.ndarray:
        indices = np.asarray(scan_indices, dtype=np.intp).reshape(-1)
        if _is_cupy_array(self.flat):
            from quantem.gpu.detector.backends.cuda.kernels import (
                cuda_selected_frame_sum_uint64,
            )

            out = cuda_selected_frame_sum_uint64(self.data, indices)
            if out is not None:
                return out
        selected = self.flat[indices]
        if _is_torch_tensor(selected):
            import torch

            if torch.is_floating_point(selected):
                raise TypeError("Exact scan ROI sums require integer detector data.")
            return selected.to(torch.int64).sum(dim=0)
        if _is_cupy_array(selected):
            import cupy as cp

            if selected.dtype.kind not in "ui":
                raise TypeError("Exact scan ROI sums require integer detector data.")
            return selected.sum(axis=0, dtype=cp.uint64)
        selected = np.asarray(selected)
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("Exact scan ROI sums require integer detector data.")
        return selected.sum(axis=0, dtype=np.uint64)

    def reduce_frames_max(self, scan_indices) -> np.ndarray:
        indices = np.asarray(scan_indices, dtype=np.intp).reshape(-1)
        if _is_cupy_array(self.flat):
            from quantem.gpu.detector.backends.cuda.kernels import (
                cuda_selected_frame_max_uint32,
            )

            out = cuda_selected_frame_max_uint32(self.data, indices)
            if out is not None:
                return out
        selected = self.flat[indices]
        if _is_torch_tensor(selected):
            import torch

            if torch.is_floating_point(selected):
                raise TypeError("Exact scan ROI maxima require integer detector data.")
            return selected.to(torch.int64).max(dim=0).values
        if _is_cupy_array(selected):
            if selected.dtype.kind not in "ui":
                raise TypeError("Exact scan ROI maxima require integer detector data.")
            return selected.max(axis=0)
        selected = np.asarray(selected)
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("Exact scan ROI maxima require integer detector data.")
        return selected.max(axis=0).astype(np.uint32, copy=False)

    def masked_sum(self, det_mask):
        mask_np = np.asarray(det_mask, dtype=bool)
        if mask_np.shape != tuple(int(x) for x in self.flat.shape[-2:]):
            raise ValueError(
                f"det_mask shape {mask_np.shape} does not match detector shape "
                f"{tuple(int(x) for x in self.flat.shape[-2:])}."
            )
        if _is_cupy_array(self.flat):
            import cupy as cp

            from quantem.gpu.detector.backends.cuda.kernels import cuda_masked_sum

            out = cuda_masked_sum(self.data, mask_np)
            if out is not None:
                return out
            mask = cp.asarray(mask_np.reshape(-1))
            selected = cp.where(mask)[0]
            flat = self.flat.reshape(self.n_frames, -1)
            return (
                flat[:, selected]
                .sum(axis=1, dtype=cp.uint64)
                .astype(cp.float32)
                .reshape(self.scan_shape)
            )
        if _is_torch_tensor(self.flat):
            import torch

            mask = torch.as_tensor(mask_np.reshape(-1), device=self.flat.device, dtype=torch.bool)
            flat = self.flat.reshape(self.n_frames, -1).to(torch.float32)
            return flat[:, mask].sum(dim=1).reshape(self.scan_shape)
        flat = np.asarray(self.flat).reshape(self.n_frames, -1)
        return (
            flat[:, mask_np.reshape(-1)]
            .sum(axis=1, dtype=np.uint64)
            .astype(np.float32)
            .reshape(self.scan_shape)
        )

    def masked_sum_exact(self, det_mask):
        mask_np = np.asarray(det_mask, dtype=bool)
        if mask_np.shape != tuple(int(x) for x in self.flat.shape[-2:]):
            raise ValueError(
                f"det_mask shape {mask_np.shape} does not match detector shape "
                f"{tuple(int(x) for x in self.flat.shape[-2:])}."
            )
        if _is_cupy_array(self.flat):
            import cupy as cp

            from quantem.gpu.detector.backends.cuda.kernels import (
                cuda_selected_sum_uint64,
            )

            indices = cp.asarray(np.flatnonzero(mask_np.reshape(-1)), dtype=cp.int32)
            out = cuda_selected_sum_uint64(self.data, indices)
            if out is not None:
                return out
            return self.flat.reshape(self.n_frames, -1)[:, indices].sum(
                axis=1, dtype=cp.uint64
            ).reshape(self.scan_shape)
        if _is_torch_tensor(self.flat):
            import torch

            mask = torch.as_tensor(
                mask_np.reshape(-1), device=self.flat.device, dtype=torch.bool
            )
            values = self.flat.reshape(self.n_frames, -1)[:, mask]
            if torch.is_floating_point(values):
                raise TypeError("Exact detector sums require integer detector data.")
            return values.to(torch.int64).sum(dim=1).reshape(self.scan_shape)
        flat = np.asarray(self.flat).reshape(self.n_frames, -1)
        if not np.issubdtype(flat.dtype, np.integer):
            raise TypeError("Exact detector sums require integer detector data.")
        return flat[:, mask_np.reshape(-1)].sum(
            axis=1, dtype=np.uint64
        ).reshape(self.scan_shape)

    def center_of_mass(self, det_mask=None):
        mask = np.ones(self.det_shape, dtype=bool)
        if det_mask is not None:
            mask = np.asarray(det_mask, dtype=bool)
            if mask.shape != self.det_shape:
                raise ValueError(
                    f"det_mask shape {mask.shape} does not match {self.det_shape}."
                )
        flat = np.asarray(self.flat, dtype=np.float64)
        weighted = flat * mask
        denominator = np.maximum(weighted.sum(axis=(1, 2)), 1e-10)
        rows = np.arange(self.det_shape[0], dtype=np.float64)[None, :, None]
        cols = np.arange(self.det_shape[1], dtype=np.float64)[None, None, :]
        com_row = (weighted * rows).sum(axis=(1, 2)) / denominator
        com_col = (weighted * cols).sum(axis=(1, 2)) / denominator
        return com_col.astype(np.float32), com_row.astype(np.float32)


def _resolve_backend(data):
    """Return the array compute backend for this data."""
    if isinstance(data, list):
        from .backends.cuda.series import CudaSeriesCompute

        from quantem.gpu._compact.interaction import StreamedSeriesCompute
        from quantem.gpu._compact.streamed import StreamedCounts

        sources = [item.data if hasattr(item, "_fields") and "data" in item._fields else item for item in data]
        from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts, CudaPackedResidentCounts

        from quantem.gpu._compact.paired import PairedCounts, PairedSeriesCompute

        if sources and all(isinstance(source, PairedCounts) for source in sources):
            return PairedSeriesCompute(data)
        if sources and all(isinstance(source, (StreamedCounts, CudaANSResidentCounts, CudaPackedResidentCounts)) for source in sources):
            return StreamedSeriesCompute(data)
        return CudaSeriesCompute(data)
    from .backends.packed import PackedDetectorCompute, is_packed_source
    from .backends.counts import CountDetectorCompute, is_count_source

    data = _unwrap_core_4dstem(data)
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    from quantem.gpu.io._resident import is_cuda_resident, _ResidentBackend

    if is_cuda_resident(data):
        return _ResidentBackend(data)
    from quantem.gpu._compact.streamed import StreamedCounts
    from quantem.gpu._compact.interaction import StreamedSeriesCompute

    if isinstance(data, StreamedCounts):
        from quantem.gpu._compact.paired import PairedCounts, PairedSeriesCompute

        result = (PairedSeriesCompute if isinstance(data, PairedCounts) else StreamedSeriesCompute)([data])
        result.series_shape = ()
        result.valid_pixels = result.valid_pixels[0]
        result.backend_metadata["series_shape"] = ()
        return result
    if is_count_source(data):
        return CountDetectorCompute(data)
    if is_packed_source(data):
        return PackedDetectorCompute(data)
    from quantem.gpu._compact.source import CompactSeries

    if isinstance(data, CompactSeries):
        return data
    if is_packed_uint4(data):
        from quantem.gpu.detector.backends.dispatch import compute_backend

        return compute_backend(data)
    if _is_cupy_array(data) or _is_torch_tensor(data):
        from quantem.gpu.detector.backends.dispatch import compute_backend

        return compute_backend(data)
    if hasattr(data, "chunks"):
        from quantem.gpu.detector.backends.dispatch import compute_backend
        from quantem.gpu.detector.backends.mps.kernels import ChunkedFrames

        # MetalRawBackend needs the ChunkedFrames contract (``.vi``); raw
        # loader results (MPSChunked4DSTEM) carry ``_is_gpu_frames`` but not
        # ``vi``, so duck-type on the attribute the backend actually uses.
        if not hasattr(data, "vi"):
            data = ChunkedFrames(data)
        return compute_backend(data)
    return _ArrayComputeBackend(data)


def _scan_shape(data, backend) -> tuple[int, int]:
    del data
    return tuple(int(value) for value in backend.scan_shape)


def _semiangle_mrad(data):
    if hasattr(data, "_fields") and "metadata" in getattr(data, "_fields", ()):
        meta = data.metadata or {}
        return meta.get("semiangle_mrad") or meta.get("semi_angle_mrad")
    meta = getattr(data, "metadata", None)
    if isinstance(meta, dict):
        return meta.get("semiangle_mrad") or meta.get("semi_angle_mrad")
    return getattr(data, "semiangle_mrad", None)


def mean_dp(data) -> np.ndarray:
    """Mean diffraction pattern for array/load/core-dataset/MPS inputs."""
    return _reduced_to_numpy(_resolve_backend(data).mean_dp())


def masked_sum(data, det_mask) -> np.ndarray:
    """Masked detector sum over scan positions.

    This is the small public helper for code that needs the shared widget/live
    masked-sum compute path without constructing a widget-local dataset object.
    """
    return prepare(data).masked_sum(det_mask)


def auto_probe(mean_dp):
    """Detect the probe (BF disk) from the mean diffraction pattern.

    Threshold at ``mean + std``, take the centroid of the bright disk for the
    center, and ``radius = sqrt(area / pi)``. Matches Show4DSTEM.auto_detect_center.
    Returns ``((center_row, center_col), bf_radius)``.
    """
    dp = np.asarray(mean_dp, dtype=np.float32)
    thr = float(dp.mean()) + float(dp.std())
    mask = dp > thr
    total = int(mask.sum())
    if total == 0:
        h, w = dp.shape
        return (h / 2.0, w / 2.0), min(h, w) * 0.25
    rows = np.arange(dp.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(dp.shape[1], dtype=np.float32)[None, :]
    cy = float((rows * mask).sum() / total)
    cx = float((cols * mask).sum() / total)
    radius = float(np.sqrt(total / np.pi))
    return (cy, cx), radius


def detector_mask(center, lo_px, hi_px, det_shape, *, dtype=np.float32) -> np.ndarray:
    """THE virtual-detector geometry primitive: boolean ``(det_row, det_col)`` mask
    of pixels whose distance from ``center`` (row, col) is in ``[lo_px, hi_px]``
    detector pixels. Every detector everywhere - ``ds.bf/adf/df``, the standalone
    ``virtual``, and the Show4DSTEM viewer's circle/annular ROIs - builds its mask
    here, so a viewer ROI and ``ds.adf()`` are pixel-identical by construction.

    ``dtype=np.float64`` uses inclusive float64 Euclidean distances for native
    compact-series interactions. The default float32 calculation is unchanged.

    Examples
    --------
    >>> mask = detector_mask((95.5, 95.5), 40, 80, (192, 192), dtype=np.float64)
    """
    cy, cx = center
    precision = np.dtype(dtype)
    if precision not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Use float32 or float64 detector geometry; got {dtype!r}.")
    rows = np.arange(det_shape[0], dtype=precision)[:, None]
    cols = np.arange(det_shape[1], dtype=precision)[None, :]
    dist = (np.hypot(rows - cy, cols - cx) if precision == np.dtype(np.float64)
            else np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2))
    return (dist >= lo_px) & (dist <= hi_px)


def _detector_mask(mode, center, bf_radius, det_shape, inner, outer):
    """Mode-based mask (BF/ABF/ADF/HAADF/DF, bands in disk-radius units) for the
    standalone :func:`virtual`. Resolves the band to pixel radii, then defers to
    :func:`detector_mask` - the one geometry primitive."""
    r = float(max(1.0, bf_radius))
    bands = {
        "BF": (0.0, r),
        "ABF": (0.5 * r, r),
        "ADF": (r, 2.0 * r),
        "HAADF": (2.0 * r, 4.0 * r),
        "DF": (r, np.inf),
    }
    if mode == "ANNULAR":
        lo, hi = (inner if inner is not None else 0.0) * r, (outer if outer is not None else np.inf) * r
    else:
        lo, hi = bands[mode]
    return detector_mask(center, lo, hi, det_shape)


# --- virtual detectors: thin geometry over the shared compute backend ---
# bf/adf/df build a boolean detector mask, then call the dataset's masked-sum
# (the single fast reduction in kernels/compute - the same one Show4DSTEM and any
# GUI use). Stateless: the probe auto-fits per call (override via center/radius),
# nothing is cached. Re-execute to rerun; cache at the edges (viewer/browser/caller).


def _mrad_to_px(data, mrad: float, radius: float) -> float:
    """Collection angle in mrad -> detector pixel radius. The bright disk radius
    spans ``semiangle_mrad``, so a mrad angle maps to
    ``mrad / semiangle_mrad * radius``."""
    semiangle_mrad = _semiangle_mrad(data)
    if not semiangle_mrad:
        raise ValueError(
            "inner / outer are collection angles in mrad, but the convergence "
            "semi-angle is unknown for this data. Store semiangle_mrad in metadata "
            "or pass detector pixels instead: adf(data, inner=..., outer=..., unit='px').")
    return float(mrad) / float(semiangle_mrad) * radius


def _to_px(data, value: float, unit: str, radius: float) -> float:
    """A collection-angle radius -> detector pixels. ``unit='mrad'`` (default)
    converts via the convergence semi-angle; ``unit='px'`` is already pixels
    (calibration-free, exact)."""
    unit = str(unit).lower()
    if unit in ("px", "pixel", "pixels"):
        return float(value)
    if unit == "mrad":
        return _mrad_to_px(data, value, radius)
    raise ValueError(f"unit must be 'mrad' or 'px', got {unit!r}")


def _probe(data, center=None, radius=None):
    if center is not None and radius is not None:
        return (float(center[0]), float(center[1])), float(radius)
    auto_center, auto_radius = auto_probe(mean_dp(data))
    center = (float(center[0]), float(center[1])) if center is not None else auto_center
    radius = float(radius) if radius is not None else auto_radius
    return center, radius


def _detector_image(data, center, lo_px: float, hi_px: float) -> np.ndarray:
    """Masked-sum image over the annulus ``lo_px .. hi_px`` detector pixels.
    Stateless - builds the mask via :func:`detector_mask` and runs the
    shared-backend masked-sum each call."""
    backend = _resolve_backend(data)
    mask = detector_mask(center, lo_px, hi_px, backend.det_shape)
    return _reduced_to_numpy(backend.masked_sum(mask)).reshape(backend.scan_shape)


def bf(data, center=None, radius=None) -> np.ndarray:
    """Bright-field image of ``data``: the bright disk (the unscattered probe).
    Probe auto-fits unless ``center``/``radius`` (detector pixels) are given."""
    center, radius = _probe(data, center, radius)
    return _detector_image(data, center, 0.0, radius)


def adf(data, inner: float | None = None, outer: float | None = None,
        unit: str = "mrad", center=None, radius=None) -> np.ndarray:
    """Annular-dark-field image of ``data``, collected between ``inner`` and
    ``outer``. ``unit='mrad'`` (default, needs ``ds.semiangle_mrad``) or
    ``unit='px'`` (raw detector pixels). Omit either for the automatic band:
    ``inner`` = the bright-disk edge, ``outer`` = twice that. Probe auto-fits
    unless ``center``/``radius`` (detector pixels) are given."""
    center, radius = _probe(data, center, radius)
    lo_px = radius if inner is None else _to_px(data, inner, unit, radius)
    hi_px = 2.0 * radius if outer is None else _to_px(data, outer, unit, radius)
    return _detector_image(data, center, lo_px, hi_px)


def df(data, inner: float | None = None, unit: str = "mrad",
       center=None, radius=None) -> np.ndarray:
    """Dark-field image of ``data``: everything collected beyond ``inner``.
    ``unit='mrad'`` (default, needs ``ds.semiangle_mrad``) or ``unit='px'``.
    Omit ``inner`` for everything outside the bright disk. Probe auto-fits
    unless ``center``/``radius`` (detector pixels) are given."""
    center, radius = _probe(data, center, radius)
    lo_px = radius if inner is None else _to_px(data, inner, unit, radius)
    return _detector_image(data, center, lo_px, np.inf)


def virtual(data, mode="BF", *, center=None, bf_radius=None, inner=None, outer=None):
    """Virtual image for ``mode`` with automatic probe fitting. See module docstring.

    ``mode`` is case-insensitive (DP/BF/ABF/ADF/HAADF/DF/annular). ``center`` and
    ``bf_radius`` override the auto-detected probe; ``inner``/``outer`` (BF-radius
    units) define a custom band when ``mode="annular"``. Returns a 2D float array
    (detector-space for DP, scan-space otherwise) for ``Show2D``.
    """
    dp = mean_dp(data)
    mode = str(mode).strip().upper()
    if mode == "DP":
        return dp
    if center is None or bf_radius is None:
        c_auto, r_auto = auto_probe(dp)
        center = center if center is not None else c_auto
        bf_radius = bf_radius if bf_radius is not None else r_auto
    mask = _detector_mask(mode, center, bf_radius, dp.shape, inner, outer)
    return masked_sum(data, mask)
