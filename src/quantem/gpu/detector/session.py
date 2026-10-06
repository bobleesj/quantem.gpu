"""Prepared detector sessions: one backend chosen for where the data resides.

``prepare`` picks the backend that computes on the data in place (encoded counts
on CUDA or MPS, scaled-precision residents, float ANS residents, CuPy arrays,
Metal frame chunks, Torch tensors or NumPy arrays) and wraps it in a
``DetectorSession``. Widget, live and remote callers use the session instead of
importing CUDA or Metal implementation modules; every result returns to the host
as NumPy unless the caller asks for native device output.
"""

import sys

import numpy as np

from quantem.gpu.detector.counts import CountDetectorCompute, CountSeriesCompute
from quantem.gpu.detector.cuda.dense import CudaKernelCompute
from quantem.gpu.detector.cuda.paired_series import PairedSeriesCompute
from quantem.gpu.detector.cuda.series import CudaSeriesCompute
from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute
from quantem.gpu.detector.float_ans import FloatANSDetectorCompute
from quantem.gpu.detector.mps.dense import MetalRawBackend
from quantem.gpu.detector.tensors import ArrayBackend, TorchBackend
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.cuda.paired import PairedCounts
from quantem.gpu.resident.float_ans import FloatANSResident
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.frames import ChunkedFrames
from quantem.gpu.resident.mps.precision import PrecisionSource as MPSPrecisionSource


class DetectorSession:
    """Prepared, cache-owning detector compute session.

    Every backend answers the operations of ``resident.queries.DetectorQueries``;
    an operation the data cannot support raises ``NotImplementedError``.
    """

    def __init__(self, data) -> None:
        self._backend = resolve_backend(data)

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
        return tuple(self._backend.series_shape)

    @property
    def backend_metadata(self) -> dict:
        """Scientific format and implementation identity, when supplied."""
        return dict(self._backend.backend_metadata)

    @property
    def detector_validity(self) -> np.ndarray | None:
        """Detector pixels the source counts, or None when every pixel counts.

        Encoded acquisitions record the pixels their file flagged (hot or dead)
        and every exact product reads those as 0; a viewer that builds its own
        masks and sums from this copy agrees with them. Bool, shape
        ``(*series_shape, detector_row, detector_col)``; dense arrays carry no
        flags and give None.

        Examples
        --------
        >>> bright_field = session.masked_sum_exact(mask & session.detector_validity)
        """
        valid = self._backend.valid_pixels
        return None if valid is None else np.array(valid, dtype=bool, copy=True)

    @property
    def timings(self) -> dict:
        """Completed backend timings for the last request, excluding display."""
        return dict(self._backend.last)

    def frame(self, index: int, *, output: str = "numpy", out=None, wait: bool = True):
        """Return one detector frame.

        Parameters
        ----------
        index
            Flat scan index in row-major order. A series decodes this
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
        if output == "numpy":
            return np.array(self._backend.frame(index), copy=True)
        if wait:
            return self._backend.frame_native(index, out=out)
        return self._backend.frame_native(index, out=out, wait=False)

    def reduce_frames(self, indices, mode: str = "mean") -> np.ndarray:
        """Reduce selected scan frames with ``mean``, ``sum``, or ``max``.

        Integer counts give exact uint64 ``sum`` and ``max``; float sources
        (scaled precision, float ANS, bounded float reads) give float32. The
        ``mean`` is float32 on every source: an integer total is divided in
        float64 and rounded once, as for :meth:`mean_dp`.
        """
        return np.asarray(self._backend.reduce_frames(indices, reduce=mode))

    def reduce_frames_exact(self, indices) -> np.ndarray:
        """Return the exact uint64 sum of selected scan frames."""
        return _exact_to_numpy(self._backend.reduce_frames_exact(indices)).reshape(self.detector_shape)

    def reduce_frames_max(self, indices) -> np.ndarray:
        """Return the exact integer maximum of selected scan frames."""
        return _exact_to_numpy(self._backend.reduce_frames_max(indices)).reshape(self.detector_shape)

    def mean_dp(self, *, output: str = "numpy"):
        """Return the float32 mean diffraction pattern on the host or device.

        Integer counts sum exactly; the total is divided by the number of scan
        positions in float64 and rounded once to float32 on every backend.
        """
        _check_output(output, None)
        if output == "native":
            return self._backend.mean_dp_native()
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
        if block_stride != 1 and (output != "native" or not isinstance(self._backend, StreamedSeriesCompute)):
            raise ValueError("block_stride needs a paired native series with output='native'.")
        if output == "numpy":
            return _reduced_to_numpy(self._backend.masked_sum(mask)).reshape((*self.series_shape, *self.scan_shape))
        # Only streamed series take the queueing and stride options; other backends keep the plain call.
        options = {} if block_stride == 1 else {"block_stride": block_stride}
        if not wait:
            options["wait"] = False
        return self._backend.masked_sum_native(mask, out=out, **options)

    def masked_sum_exact(self, mask, *, output: str = "numpy", out=None):
        """Return an exact uint64 virtual-detector image for one mask.

        Parameters
        ----------
        mask
            Full-resolution detector mask. Compact masks must be binary.
        output
            ``"numpy"`` retains the exact uint64 host default. ``"native"``
            returns exact counts on the CUDA device of a count series or an
            encoded CUDA acquisition, with leading ``series_shape``, in the
            sum dtype the native dtype and detector size need (uint32 or
            uint64). Float sources (scaled precision, float ANS, bounded
            float reads) raise ``TypeError`` for either output; other sources
            have no native exact output.
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
        if output == "native":
            return self._backend.masked_sum_exact_native(mask, out=out)
        result = _exact_to_numpy(self._backend.masked_sum_exact(mask))
        return result.reshape((*self.series_shape, *self.scan_shape))

    def weighted_sum_exact(self, weights) -> np.ndarray:
        """Return exact uint64 per-scan sums of integer detector weights times counts.

        Parameters
        ----------
        weights
            Nonnegative integer weight for each detector ``(row, col)`` pixel,
            for example the detector row index for a centre-of-mass moment.
            Flagged detector pixels weigh nothing.

        Returns
        -------
        numpy.ndarray
            uint64 sums with shape ``(*series_shape, *scan_shape)``.

        Examples
        --------
        >>> rows, cols = np.indices(session.detector_shape)
        >>> row_moment = session.weighted_sum_exact(rows)
        """
        result = _exact_to_numpy(self._backend.weighted_sum_exact(weights))
        return result.reshape((*self.series_shape, *self.scan_shape))

    def detector_total(self) -> np.ndarray:
        """Return the exact uint64 sum of every scan position's pattern.

        Encoded acquisitions decode each detector pixel's streams once, which
        is far faster than ``reduce_frames_exact`` over every scan index.
        Flagged detector pixels hold 0. A series returns one total per
        acquisition, shape ``(*series_shape, *detector_shape)``.

        Examples
        --------
        >>> mean_dp = (session.detector_total() / session.num_frames).astype(np.float32)
        """
        result = _exact_to_numpy(self._backend.detector_total())
        return result.reshape((*self.series_shape, *self.detector_shape))

    def masked_sums_exact(self, masks) -> np.ndarray:
        """Return exact uint64 virtual-detector images for several masks.

        ``masks`` is an array in ``(mask, detector_row, detector_column)``
        order; the result is ``(mask, *series_shape, *scan_shape)``.
        """
        values = np.asarray(masks)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or values.shape[1:] != self.detector_shape:
            raise ValueError(
                "Detector masks must have shape "
                f"(mask, {self.detector_shape[0]}, {self.detector_shape[1]})."
            )
        if len(values) < 1 or not np.all((values == 0) | (values == 1)):
            raise ValueError("Exact detector masks must be binary.")
        return np.stack([self.masked_sum_exact(mask) for mask in values], axis=0)

    def center_of_mass(self, mask=None) -> tuple[np.ndarray, np.ndarray]:
        """Return the count-weighted detector centre of each scan position.

        Encoded counts on CUDA and MPS give the same float32 values: exact
        integer moments over the valid pixels of ``mask`` divided by the exact
        total in float64, and 0 where a pattern holds no counts.

        Returns
        -------
        tuple of numpy.ndarray
            Detector ``(row, col)`` centres in pixels, each with ``scan_shape``.
            They are not mean-subtracted; ``dpc.center_of_mass`` does that.

        Examples
        --------
        >>> com_row, com_col = session.center_of_mass()
        """
        com_col, com_row = self._backend.center_of_mass(mask)
        row = _reduced_to_numpy(com_row).reshape(self.scan_shape)
        col = _reduced_to_numpy(com_col).reshape(self.scan_shape)
        return row, col

    @property
    def supports_fast(self) -> bool:
        """Whether this session provides an accelerated interaction sidecar."""
        return isinstance(self._backend, MetalRawBackend)

    @property
    def fast_ready(self) -> bool:
        """Whether the interaction sidecar is ready."""
        return self.supports_fast and self._backend.has_fast

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

    def close(self) -> None:
        """Release this session's ownership of backend caches."""
        self._backend = None


def prepare(data) -> DetectorSession:
    """Prepare detector queries over one source or a list of acquisitions.

    Parameters
    ----------
    data
        A loaded source, array, or list of equally shaped uint8/uint16
        acquisitions on one CUDA device or on MPS. A list retains its complete
        acquisition axis; CUDA uses one joint kernel launch per native
        detector or point-pattern query, MPS queries each acquisition in turn.

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


def resolve_backend(data):
    """Choose the one backend that computes on ``data`` where it already resides.

    CuPy, Torch and Metal are imported only by the backends that use them, so a
    Mac never imports CuPy and a CUDA machine never imports Metal.
    """
    # quantem.widget's bounded readers mark themselves; quantem.gpu cannot import the widget.
    if getattr(data, "_bounded_detector_source", False):
        # The bounded reader needs Torch, which only the widget's readers bring.
        from quantem.gpu.detector.bounded import BoundedDetectorCompute

        owner = data._detector_source
        encoded = isinstance(owner, Dataset4dstemGPU) and owner.representation == "encoded"
        return BoundedDetectorCompute(data, DetectorSession(owner) if encoded else None)
    if isinstance(data, list):
        sources = [item.data if isinstance(item, Dataset4dstemGPU) else item for item in data]
        if sources and all(isinstance(source, MPSStreamedCounts) for source in sources):
            return CountSeriesCompute(sources)
        if sources and all(isinstance(source, PairedCounts) for source in sources):
            return PairedSeriesCompute(data)
        if sources and all(isinstance(source, StreamedCounts) for source in sources):
            return StreamedSeriesCompute(data)
        return CudaSeriesCompute(data)
    data = _unwrap_core_4dstem(data)
    # Scaled-precision residents answer detector queries on their own encoded data.
    if isinstance(data, MPSPrecisionSource) or _is_loaded_instance(
        data, "quantem.gpu.resident.cuda.precision", "PrecisionSource"
    ):
        return data
    if isinstance(data, FloatANSResident):
        return FloatANSDetectorCompute(data)
    if isinstance(data, StreamedCounts):
        # One acquisition is a series of one, presented without the series axis.
        result = (PairedSeriesCompute if isinstance(data, PairedCounts) else StreamedSeriesCompute)([data])
        result.series_shape = ()
        result.valid_pixels = result.valid_pixels[0]
        result.backend_metadata["series_shape"] = ()
        return result
    if isinstance(data, MPSStreamedCounts):
        return CountDetectorCompute(data)
    if _is_loaded_instance(data, "cupy", "ndarray"):
        return CudaKernelCompute(data)
    # MPS SSB's exact BF-column frames expose ``chunks`` too; ssb sits above
    # detector, so that type cannot be named here.
    if isinstance(data, ChunkedFrames) or hasattr(data, "chunks"):
        return MetalRawBackend(data)
    if _is_loaded_instance(data, "torch", "Tensor"):
        # A CPU tensor is host data: view it as NumPy for the float64 host reference.
        return ArrayBackend(data.detach().numpy()) if data.device.type == "cpu" else TorchBackend(data)
    return ArrayBackend(data)


def _unwrap_core_4dstem(data):
    """Return the numeric data of a Dataset4dstemGPU or a quantem.core Dataset4dstem.

    quantem.core is optional, so its dataset is recognized by the attributes
    that hold its array rather than by importing the class.
    """
    if isinstance(data, Dataset4dstemGPU):
        return data.data
    if isinstance(data, np.ndarray) or _is_loaded_instance(data, "cupy", "ndarray") or _is_loaded_instance(
        data, "torch", "Tensor"
    ):
        return data
    for name in ("_tensor", "_array"):
        array = getattr(data, name, None)
        if array is not None:
            return array
    array = getattr(data, "array", None)
    if array is not None and not callable(array):
        return array
    return data


def _is_loaded_instance(value, module: str, name: str) -> bool:
    """Type check against a class of an optional library without importing it.

    An instance of the class can exist only once its module was imported, so
    an unloaded module means ``value`` is not one.
    """
    loaded = sys.modules.get(module)
    return loaded is not None and isinstance(value, getattr(loaded, name))


def _check_output(output: str, out) -> None:
    """Reject an ``out`` buffer for host output: it would be silently ignored."""
    if output not in ("numpy", "native"):
        raise ValueError(f"Use output='numpy' or 'native'; got {output!r}.")
    if out is not None and output != "native":
        raise ValueError("out is a device buffer; use output='native' with it.")


def _reduced_to_numpy(data) -> np.ndarray:
    """Convert a small reduced product to float32 NumPy for widget display."""
    return np.asarray(_to_host(data), dtype=np.float32)


def _exact_to_numpy(data) -> np.ndarray:
    """Copy a reduced integer product to host without changing its values."""
    array = _to_host(data)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            "Exact detector sums require integer detector data; "
            f"the backend returned {array.dtype}."
        )
    return array.astype(np.uint64, copy=False)


def _to_host(data) -> np.ndarray:
    """Download a CuPy or Torch result; NumPy and Metal results convert in place."""
    if _is_loaded_instance(data, "cupy", "ndarray"):
        return data.get()
    if _is_loaded_instance(data, "torch", "Tensor"):
        return data.detach().cpu().numpy()
    return np.asarray(data)
