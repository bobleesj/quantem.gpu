"""Backend-neutral loaded data and ownership contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .representation import DataRepresentation


@dataclass(frozen=True)
class MasterReadiness:
    """Header-only readiness report for one 4D-STEM master.

    Parameters
    ----------
    ready
        Whether every selected detector source is readable, internally
        consistent, and contains the expected number of stored frames.
    reason
        Concise description of the observed state.
    action
        Corrective next step when ``ready`` is ``False``.
    source_kind
        ``"inline"`` or ``"external"`` according to the selected
        ``entry/data`` source layout, or ``"unavailable"`` when inspection
        could not identify a data source.
    actual_frames
        Total stored frame count across the selected datasets, when known.
    expected_frames
        Frame count derived from an explicit ``scan_shape`` or discoverable
        master metadata, when available.
    detector_shape
        Common detector shape ``(row, col)``, when known.
    dtype
        Common NumPy dtype string, when known.
    source_signature
        JSON-serializable file-stat and dataset-header fingerprint. Callers can
        compare this dictionary across polls without reading detector pixels.
    """

    ready: bool
    reason: str
    action: str
    source_kind: str
    actual_frames: int | None
    expected_frames: int | None
    detector_shape: tuple[int, int] | None
    dtype: str | None
    source_signature: dict[str, Any]


def _release_owned_storage(
    data: object, *, failure: BaseException | None = None
) -> None:
    """Release the resident owner without masking an active workflow failure."""
    for name in ("release_resident_storage", "release", "free"):
        release = getattr(data, name, None)
        if callable(release):
            try:
                release()
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Resident cleanup also failed: {cleanup_error}")
            return


class FourDSTEMData(NamedTuple):
    """Loaded 4D-STEM data with backend-neutral representation metadata.

    Attributes
    ----------
    data
        Backend-native detector data by default. CUDA returns a CuPy array;
        MPS may return a chunk-backed Metal frame source. With
        ``output="torch"``, this is a Torch tensor. Shape is normally
        ``(scan_row, scan_col, detector_row, detector_col)`` when the scan
        shape is known, otherwise ``(frame, detector_row, detector_col)``.
    metadata
        Acquisition and detector metadata from the HDF5 source. The mapping
        includes these normalized fields:

        **Derived, named fields** (always present; value is ``None`` when
        the source field is missing):

        - ``scan_shape`` : ``(H, W)`` or ``None``
            Auto-derived from ``ntrigger`` assuming a square scan.
        - ``n_frames`` : ``int`` or ``None``
            Total frame count.
        - ``dwell_time_us`` : ``float`` or ``None``
            Per-frame dwell in microseconds.
        - ``detector_shape`` : ``(H, W)`` or ``None``
            Detector pixel count.
        - ``detector_name`` : ``str`` or ``None``
            Human-readable detector description.
        - ``saturation`` : ``int`` or ``None``
            ADU ceiling before the detector saturates.

        **Raw HDF5 scalars**: every scalar dataset in the file keyed by its
        full HDF5 path (e.g. ``metadata["entry/instrument/detector/count_time"]``),
        as an escape hatch for fields not in the derived layer.

        .. note::

            Scope-side parameters (``voltage_kV``, ``semiangle``,
            ``scan_sampling``, ``camera_length``, ``rotation``) are NOT in
            the h5 master - pass them to ``ssb()`` explicitly.

    Examples
    --------
    ```python
    data, meta = load("scan_master.h5")
    data.shape             # (512, 512, 192, 192)
    meta["scan_shape"]     # (512, 512)
    meta["dwell_time_us"]  # 99.6
    meta["detector_name"]  # detector model string
    ```
    """

    data: Any
    metadata: dict[str, Any]

    @property
    def representation(self) -> DataRepresentation:
        """Return how the complete logical array is encoded."""
        value = self.metadata.get("representation", DataRepresentation.DENSE.value)
        return DataRepresentation.parse(value)

    @property
    def residency(self) -> str:
        """Return where the representation remains available after loading."""
        return str(self.metadata.get("residency", "device"))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return logical ``(scan row, scan column, detector row, detector column)`` shape."""
        value = self.metadata.get("working_shape")
        if value is None:
            value = getattr(self.data, "shape", ())
        return tuple(int(item) for item in value)

    @property
    def dtype(self) -> np.dtype:
        """Return the scientific working dtype exposed by the representation."""
        value = self.metadata.get("working_dtype", self.metadata.get("dtype"))
        if value is None:
            value = getattr(self.data, "dtype", None)
        if value is None:
            raise AttributeError("Loaded data did not report a scientific working dtype.")
        token = str(value).removeprefix("torch.")
        return np.dtype(np.uint8 if token == "uint4" else token)

    @property
    def logical_bytes(self) -> int:
        """Return bytes required by an equivalent dense working tensor."""
        value = self.metadata.get("working_logical_tensor_bytes")
        if value is not None:
            return int(value)
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def resident_bytes(self) -> int | None:
        """Return measured physical resident bytes when the backend reports them."""
        value = self.metadata.get("physical_resident_bytes")
        if value is None:
            value = getattr(self.data, "nbytes", None)
        return int(value) if value is not None else None

    @property
    def lossless(self) -> bool:
        """Return whether source-to-working exactness is established by metadata.

        False also means unverified, not necessarily that values were lost.
        The declared detector-mask policy remains part of the working array.
        """
        return bool(self.metadata.get("lossless_exact", False))

    def close(self) -> None:
        """Release owned resident storage after the final scientific consumer.

        Examples
        --------
        >>> loaded = load("scan-lossless.h5", backend="mps")
        >>> loaded.close()
        """
        _release_owned_storage(self.data)

    def to_representation(self, representation: DataRepresentation | str) -> FourDSTEMData:
        """Return an exact independently owned conversion when supported.

        The source remains usable and caller-owned. Requesting its current
        representation returns this same object, not a second ownership lease.
        Unsupported directions fail before hidden materialization or CPU work.
        Conversion readiness is backend-specific during this integration.
        """
        from ._ans_dispatch import _convert_resident

        return _convert_resident(self, representation)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

# Retained public result spelling.
LoadResult = FourDSTEMData
