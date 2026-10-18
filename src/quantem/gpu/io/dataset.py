"""Backend-neutral loaded data and ownership contracts."""

import warnings
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from quantem.gpu.formats.qem.metadata import (
    DETECTOR_SAMPLING_KEYS,
    _microscopy_quantity,
)
from quantem.gpu.io.read import read, resident_device
from quantem.gpu.io.representation import DataRepresentation

if TYPE_CHECKING:
    import pandas as pd


@dataclass(eq=False, frozen=True, slots=True)
class Dataset4dstemGPU:
    """Loaded acquisition with metadata and GPU tensor indexing.

    Owns the resident storage and decoder. Selecting scan/detector regions
    returns ordinary PyTorch tensors; calibration remains in ``metadata``.
    The read-only ``sampling``, ``units`` and ``origin`` properties expose
    that metadata in axis order without decoding data or keeping another copy.

    ``data`` is a GPU tensor or array, or an encoded resident owner from any
    backend; algorithm packages may also wrap their own arrays. The
    shape, dtype and size queries therefore read ``metadata`` first and fall
    back to whatever the owner reports.
    """

    data: object
    metadata: dict[str, object]

    def metadata_table(self) -> "pd.DataFrame":
        """Inspect acquisition calibration before choosing an algorithm.

        The table describes the loaded metadata, including saved calibration
        overrides. Missing values remain explicitly unrecorded; an algorithm's
        defaults or later keyword overrides do not rewrite this table. No
        detector values are read. Complete source records remain in ``metadata``.

        Returns
        -------
        pandas.DataFrame
            Acquisition geometry and calibration with units and provenance.

        Examples
        --------
        >>> data = io.load("acquisition.qem")
        >>> data.metadata_table()
        """
        import pandas as pd

        from quantem.gpu.formats.qem.snapshot import (
            FLOAT_CODEC,
            INTEGER_CODEC,
            SCALED_CODEC,
        )
        from quantem.gpu.io.metadata_display import calibration_rows

        rows = {
            "Source": {"Value": self.metadata.get("source_path", "In-memory acquisition")},
            "Format": {"Value": self.metadata.get("container", self.metadata.get("source_kind", "Not recorded"))},
            "Shape (scan, detector)": {"Value": " × ".join(map(str, self.shape))},
            "Data dtype": {"Value": str(self.dtype)},
        }
        source = Path(str(self.metadata.get("source_path", "")))
        if source.suffix.lower() == ".qem" and source.is_file():
            rows["QEM file size"] = {
                "Value": source.stat().st_size / 1e9, "Unit": "GB", "Metadata": "On disk",
            }
        ans = self.representation is DataRepresentation.ENCODED and (
            self.metadata.get("resident_codec") == "ans"
            or self.metadata.get("resident_profile") in (INTEGER_CODEC, SCALED_CODEC, FLOAT_CODEC)
        )
        resident_bytes = self.resident_bytes
        rows["ANS resident size" if ans else "Resident size"] = {
            "Value": "Not reported" if resident_bytes is None else resident_bytes / 1e9,
            "Unit": "GB", "Metadata": "Loaded storage" if resident_bytes is not None else "Not reported",
        }
        rows["Uncompressed size"] = {
            "Value": self.logical_bytes / 1e9, "Unit": "GB", "Metadata": "Shape × dtype",
        }
        rows.update(calibration_rows(self.metadata))
        return pd.DataFrame.from_dict(rows, orient="index").fillna("").rename_axis("Property")

    def _repr_html_(self) -> str:
        return self.metadata_table().to_html(escape=True)

    def __reduce_ex__(self, protocol):
        raise TypeError(
            "Save resident acquisitions with quantem.gpu.io.save(path, data); device handles cannot be pickled."
        )

    def __array__(self, dtype=None, copy=None):
        """Return the dense CPU reference as NumPy; reject implicit full-acquisition copies from a GPU."""
        if isinstance(self.data, np.ndarray):
            return np.array(self.data, dtype=dtype, copy=copy)
        raise TypeError(
            "Acquisition data stays on the GPU. Select a bounded region first, "
            "then use data[row, column].cpu().numpy() for a NumPy array. "
            "Access acquisition metadata through data.metadata."
        )

    def __repr__(self) -> str:
        """Summarize the acquisition without decoding detector values."""
        return self.metadata_table().to_string()

    @property
    def ndim(self) -> int:
        """Return the number of logical array axes."""
        return len(self.shape)

    @property
    def name(self) -> str:
        """Name of the acquisition: ``metadata['name']``, else its source file.

        A master is named without ``_master.h5``, another file without its
        extension. quantem core datasets carry a ``name``; a loaded acquisition
        reports one the same way, so code that titles or labels data reads it
        from either. Empty when the metadata names no source.

        Examples
        --------
        >>> load("scan_001_master.h5").name
        'scan_001'
        """
        name = self.metadata.get("name")
        if name:
            return str(name)
        file = Path(str(self.metadata.get("source_path", ""))).name
        return file.removesuffix("_master.h5") if file.endswith("_master.h5") else Path(file).stem

    @property
    def size(self) -> int:
        """Return the logical element count without decoding any values."""
        return prod(self.shape)

    @property
    def representation(self) -> DataRepresentation:
        """Return how the complete logical array is encoded."""
        value = self.metadata.get("representation", DataRepresentation.DENSE.value)
        return DataRepresentation.parse(value)

    @property
    def device(self):
        """Return the normalized device without decoding detector values."""
        return resident_device(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return logical ``(scan row, scan column, detector row, detector column)`` shape."""
        value = self.metadata.get("working_shape")
        if value is None:
            value = getattr(self.data, "shape", ())
        return tuple(int(item) for item in value)

    def _axis_values(self, name: str) -> list:
        """Read one metadata value per logical axis, retaining unknown values."""
        value = self.metadata.get(name)
        if value is None or np.isscalar(value):
            return [value] * self.ndim
        if len(value) != self.ndim:
            raise ValueError(
                f"metadata[{name!r}] must have {self.ndim} axis values; "
                f"got {len(value)}. Match the order in data.shape."
            )
        return list(value)

    def _axis_calibration(self) -> tuple[tuple, tuple]:
        """Use the reader's effective calibration ahead of generic axis metadata."""
        sampling = self._axis_values("sampling")
        units = self._axis_values("units")
        if self.ndim == 4:
            for start, key, unit in (
                (0, "scan_sampling_A", "angstrom"),
                *((2, key, unit) for unit, key in DETECTOR_SAMPLING_KEYS.items()),
            ):
                value = self.metadata.get(key)
                if value is not None:
                    if np.isscalar(value) or len(value) != 2:
                        raise ValueError(f"metadata[{key!r}] must be a [row, col] pair; got {value!r}.")
                    sampling[start:start + 2] = list(value)
                    units[start:start + 2] = [unit] * 2
        return (
            tuple(None if value is None else float(value) for value in sampling),
            tuple(units),
        )

    @property
    def sampling(self) -> tuple[float | None, ...]:
        """Pixel spacing along each logical axis, with unknown values as None.

        For a 4D acquisition the order is scan row, scan col, detector row,
        detector col. Scan spacing is in angstrom; detector spacing uses
        ``units``. Effective reader calibration takes precedence over generic
        axis metadata, including when a saved calibration override is present.

        Examples
        --------
        >>> data = load("gold.qem")
        >>> scan_sampling = data.sampling[:2]
        """
        return self._axis_calibration()[0]

    @property
    def units(self) -> tuple[str | None, ...]:
        """Units for each axis's sampling and origin; None means unspecified.

        An uncalibrated axis is not silently assigned angstrom or reciprocal
        units. These values describe the acquisition, not a selected tensor.

        Examples
        --------
        >>> data = load("gold.qem")
        >>> detector_units = data.units[2:]
        """
        return self._axis_calibration()[1]

    @property
    def origin(self) -> tuple[float | None, ...]:
        """Coordinate of the first pixel on each axis, in the axis's units.

        Unknown coordinates remain None. No detector center or absolute scan
        position is inferred. Explicit origins come from ``metadata['origin']``
        in ``metadata['units']`` when recorded, otherwise the effective units.

        Examples
        --------
        >>> data = load("gold.qem")
        >>> scan_origin = data.origin[:2]
        """
        origin = [
            None if value is None else float(value)
            for value in self._axis_values("origin")
        ]
        for axis, (source_unit, unit) in enumerate(zip(
            self._axis_values("units"), self.units,
        )):
            if (origin[axis] is None or source_unit is None or unit is None
                    or source_unit == unit):
                continue
            kind = "scan" if axis < 2 else "detector"
            source = _microscopy_quantity({"value": 1, "unit": source_unit}, kind)
            target = _microscopy_quantity({"value": 1, "unit": unit}, kind)
            if source["unit"] != target["unit"]:
                raise ValueError(
                    f"Origin axis {axis} uses {source_unit!r}, but sampling uses "
                    f"{unit!r}. Supply origin and sampling in compatible units."
                )
            origin[axis] *= source["value"] / target["value"]
        return tuple(origin)

    @property
    def dtype(self) -> np.dtype:
        """Return the scientific working dtype exposed by the representation."""
        value = self.metadata.get("working_dtype", self.metadata.get("dtype"))
        if value is None:
            value = getattr(self.data, "dtype", None)
        if value is None:
            raise AttributeError(
                "Loaded data did not report a scientific working dtype."
            )
        return np.dtype(str(value).removeprefix("torch."))

    @property
    def logical_bytes(self) -> int:
        """Return bytes required by an equivalent dense working tensor."""
        value = self.metadata.get("working_logical_tensor_bytes")
        if value is not None:
            return int(value)
        return self.size * self.dtype.itemsize

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
        # Encoded residents and callers' own owners name it release; plain arrays
        # have none and are freed by Python.
        release = getattr(self.data, "release", None)
        if callable(release):
            release()

    def release_rows_before(self, row: int) -> None:
        """Free the stored scan rows above ``row``, for a consumer that reads the scan once from the top.

        A merge that walks down the scan frees each part of its inputs once it
        is past it, so the inputs shrink while the result grows. Only whole
        stored chunks are freed, so a few rows above ``row`` may remain.
        Reading a freed row raises. Integer and float ANS sources only; a
        scaled-precision source raises ``TypeError``.
        """
        self.data.release_scans_before(row * self.shape[1])

    def __getitem__(self, key):
        """Decode a basic array selection into a tensor on the source GPU.

        Axes are scan row, scan column, detector row, detector column.
        Integers remove axes; slices and one ellipsis follow Python indexing,
        including negative indices and steps. Boolean masks, index arrays and
        new axes are not supported. A strided selection decodes its bounding
        region before selecting values; it does not interpolate or bin them.
        Metadata remains available through ``metadata``.
        """
        if len(self.shape) != 4:
            raise ValueError(
                "Index a single 4D acquisition; select a series member first."
            )
        keys = key if isinstance(key, tuple) else (key,)
        if sum(item is Ellipsis for item in keys) > 1:
            raise IndexError("Use at most one ellipsis.")
        if any(item is Ellipsis for item in keys):
            position = next(axis for axis, item in enumerate(keys) if item is Ellipsis)
            keys = (
                keys[:position]
                + (slice(None),) * (5 - len(keys))
                + keys[position + 1 :]
            )
        if len(keys) > 4:
            raise IndexError("A 4D acquisition accepts at most four indices.")
        keys += (slice(None),) * (4 - len(keys))
        regions, selection, reverse, output_shape = [], [], [], []
        empty = False
        for axis, (item, size) in enumerate(zip(keys, self.shape)):
            if isinstance(item, (bool, np.bool_)):
                raise TypeError("Use integer indices or slices, not boolean masks.")
            if isinstance(item, (int, np.integer)):
                index = int(item)
                if index < 0:
                    index += size
                if not 0 <= index < size:
                    raise IndexError(
                        f"Index {item} is outside axis {axis} with size {size}."
                    )
                regions.extend((index, index + 1))
                selection.append(0)
            elif isinstance(item, slice):
                indices = range(*item.indices(size))
                count = len(indices)
                output_shape.append(count)
                empty |= count == 0
                if count:
                    regions.extend(
                        (
                            min(indices[0], indices[-1]),
                            max(indices[0], indices[-1]) + 1,
                        )
                    )
                else:
                    regions.extend((0, 1))
                selection.append(slice(None, None, abs(indices.step)))
                if indices.step < 0:
                    reverse.append(axis)
            else:
                raise TypeError(
                    "Use integer indices, slices or ellipsis; "
                    "index arrays and new axes are unsupported."
                )
        if empty:
            # Obtain the backend's scientific tensor dtype/device with one pixel.
            return self.read(
                scan_region=(0, 1, 0, 1), detector_region=(0, 1, 0, 1)
            ).new_empty(output_shape)
        values = self.read(
            scan_region=tuple(regions[:4]), detector_region=tuple(regions[4:])
        )
        if reverse:
            values = values.flip(reverse)
        return values[tuple(selection)]

    def read(
        self,
        *,
        scan_region: tuple[int, int, int, int] | None = None,
        detector_region: tuple[int, int, int, int] | None = None,
    ):
        """Read one bounded logical region as a Torch tensor on the source GPU.

        Regions use ``(row_start, row_stop, col_start, col_stop)`` with
        exclusive stops. The resident representation, decoding, and transfer
        scheduling remain automatic. A complete read is allowed only when its
        dense tensor fits the accelerator's current working memory.
        """
        return read(
            self,
            scan_region=scan_region,
            detector_region=detector_region,
        )

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for row in range(len(self)):
            yield self[row]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def resident_metadata(shape, dtype, resident_bytes: int, *, backend: str, representation: str = "encoded") -> dict:
    """Return the metadata fields every resident loader records about what it holds.

    ``Dataset4dstemGPU`` answers ``shape``, ``dtype`` and ``resident_bytes``
    from these fields without decoding, so every loading route must report
    them with the same keys and meaning. A complete acquisition is held: the
    working shape is the source shape, without binning or cropping.
    """
    shape = tuple(shape)
    dtype = np.dtype(dtype)
    return dict(
        backend=backend,
        representation=representation,
        residency="device",
        source_shape=shape,
        working_shape=shape,
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        n_frames=prod(shape[:2]),
        dtype=dtype.name,
        working_dtype=dtype.name,
        working_logical_tensor_bytes=prod(shape) * dtype.itemsize,
        physical_resident_bytes=resident_bytes,
        scan_bin=1,
        detector_bin=1,
        crop=None,
    )


def drop_unread_calibration(loaded, sources):
    """Remove the calibration attributes of an HDF5 master that this version does not read, and warn once per file.

    A master's attributes reach the dataset metadata as stored. The unit-less detector pair of earlier masters
    (``detector_sampling`` with ``detector_sampling_unit``) and a sampling stored as one number are not the
    ``[row, col]`` pairs ``scan_sampling_A``, ``detector_sampling_mrad`` and ``detector_sampling_inv_A`` this
    version reads, and no value is converted. Left in the metadata they made those axes uncalibrated without a
    word, then raised when the sampling was read or the acquisition was saved as ``.qem``.

    ``loaded`` is one dataset or a list, ``sources`` the files they came from; ``loaded`` is returned. The message
    is a ``UserWarning``, not progress output: a calibration that is in the file and is not used must be seen
    whatever ``verbose`` is, also by ``io.convert``, which loads quietly.
    """
    for item, source in zip(loaded if isinstance(loaded, list) else [loaded], sources):
        unread = {name: name for name in ("detector_sampling", "detector_sampling_unit") if name in item.metadata}
        unread.update({
            name: f"{name} (not a [row, col] pair)"
            for name in ("scan_sampling_A", "detector_sampling_mrad", "detector_sampling_inv_A")
            if item.metadata.get(name) is not None and (np.isscalar(item.metadata[name]) or len(item.metadata[name]) != 2)
        })
        for name in unread:
            del item.metadata[name]
        if unread:
            warnings.warn(
                f"Calibration not read: {Path(source).name} holds {', '.join(unread.values())}, which this version "
                "does not read; those axes are uncalibrated. It reads scan_sampling_A, detector_sampling_mrad and "
                "detector_sampling_inv_A, each a [row, col] pair.",
                UserWarning,
                # the caller of io.load or of PairedLoader.stream, where the file is named
                stacklevel=3,
            )
    return loaded
