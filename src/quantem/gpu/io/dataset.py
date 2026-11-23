"""Backend-neutral loaded data and ownership contracts."""

import json
import warnings
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from weakref import ref

import numpy as np
import torch

from quantem.gpu.formats.qem.metadata import (
    DETECTOR_SAMPLING_KEYS,
    _microscopy_quantity,
)
from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.io.read import read, resident_device
from quantem.gpu.io.representation import DataRepresentation

if TYPE_CHECKING:
    import pandas as pd

    from quantem.gpu.detector import BrightFieldDisk


def _browse_region(scan_shape, location, region_shape):
    """Place a named region inside the field, leaving room for surrounding positions."""
    quarters = {
        "top left": (1, 1),
        "top right": (1, 3),
        "middle": (2, 2),
        "bottom left": (3, 1),
        "bottom right": (3, 3),
    }
    if location not in quarters:
        raise ValueError(f"Choose scan_region from {', '.join(quarters)}, or give its row and column bounds.")
    if (
        not isinstance(region_shape, (tuple, list)) or len(region_shape) != 2
        or region_shape[0] != region_shape[1]
        or type(region_shape[0]) is not int or region_shape[0] < 2 or region_shape[0] % 2
        or region_shape[0] > min(scan_shape)
    ):
        raise ValueError(f"Choose a square, even scan_region_shape (rows, cols) within the {tuple(scan_shape)} scan.")

    bounds = []
    for edge, length, quarter in zip(scan_shape, region_shape, quarters[location], strict=True):
        margin = min(length, edge // 2)
        center = max(margin, min(edge - margin, edge * quarter // 4))
        start = center - length // 2
        bounds.extend((start, start + length))
    return tuple(bounds)


class Calibration:
    """Microscope calibration of a loaded acquisition, read and set by name.

    Reading a field returns the value the analysis uses: a recorded value, or
    one you set, else None. Setting a field records your value beside the
    file's record, which stays unchanged in ``metadata['scientific_metadata']``;
    ``io.save`` keeps both. Recreate a reconstruction session after a change.

    Examples
    --------
    >>> data = io.load("scan_master.h5")  # doctest: +SKIP
    >>> data.calibration.voltage_kV = 300  # doctest: +SKIP
    >>> data.calibration.semiangle_mrad = 30  # doctest: +SKIP
    >>> data.calibration.scan_sampling_A = (0.264, 0.264)  # doctest: +SKIP
    >>> fit = data.calibration.detector_sampling()  # doctest: +SKIP
    >>> data.calibration  # doctest: +SKIP
    """

    __slots__ = ("_dataset",)
    # each field, the unit its value is in, and whether it is a (row, col) pair
    FIELDS = {
        "voltage_kV": ("kV", False),
        "semiangle_mrad": ("mrad", False),
        "scan_sampling_A": ("angstrom", True),
        "detector_sampling_mrad": ("mrad", True),
        "detector_sampling_inv_A": ("1/angstrom", True),
    }

    # the bright-field disk detector_sampling records: read here, never set by hand (give a centre or a radius to
    # Direct ptychography or Show4DSTEM instead)
    FITTED = {
        "bright_field_center_px": ("px", True),
        "bright_field_edge_radius_px": ("px", False),
    }

    def __init__(self, dataset: "Dataset4dstem") -> None:
        object.__setattr__(self, "_dataset", dataset)

    def __getattr__(self, name: str):
        if name in Calibration.FITTED:
            value = (self._dataset.metadata.get(qem_metadata.BRIGHT_FIELD_DISK_KEY) or {}).get(name)
        elif name in Calibration.FIELDS:
            value = self._dataset.metadata.get(name)
        else:
            raise AttributeError(_unknown_calibration(name))
        return tuple(value) if isinstance(value, (list, tuple)) else value

    def __setattr__(self, name: str, value) -> None:
        if name in Calibration.FITTED:
            raise AttributeError(
                f"{name} is fitted by data.calibration.detector_sampling(); to use another value, give "
                f"{name.replace('_edge', '')}= to DirectPtycho(...) or Show4DSTEM(...)."
            )
        if name not in Calibration.FIELDS:
            raise AttributeError(_unknown_calibration(name))
        self._dataset._record_calibration(**{name: value})

    def __dir__(self):
        return [*Calibration.FIELDS, *Calibration.FITTED, "detector_sampling", "table"]

    def detector_sampling(self) -> "BrightFieldDisk":
        """Calibrate the detector from the bright-field disk and the recorded convergence semiangle, and record it.

        The bright-field disk is the image of the probe-forming aperture, so its edge sits at the convergence
        semiangle: ``detector_sampling_mrad = semiangle_mrad / bright_field_edge_radius_px`` on both axes
        (``detector.bright_field_disk``). The mean pattern is computed on the encoded acquisition, with no dense
        copy. The fit records the detector sampling (source ``Fitted (bright-field disk)``), the disk centre and its
        edge radius here; direct ptychography, the viewers and ``io.save`` then use them instead of fitting again. A detector sampling
        you set by hand is kept, and one line says so. Fit again after changing ``semiangle_mrad``.

        Returns
        -------
        BrightFieldDisk
            ``bright_field_center_px`` (row, col), ``bright_field_edge_radius_px``, ``detector_sampling_mrad``
            (row, col) mrad per pixel, ``method``, and ``show()`` for the mean pattern with the fitted edge.

        Raises
        ------
        ValueError
            When no convergence semiangle is recorded, or the mean pattern has no bright-field disk.

        Examples
        --------
        >>> data.calibration.semiangle_mrad = 30  # doctest: +SKIP
        >>> fit = data.calibration.detector_sampling()  # doctest: +SKIP
        >>> fit.show()  # doctest: +SKIP
        """
        from quantem.gpu import detector

        semiangle_mrad = self.semiangle_mrad
        if semiangle_mrad is None:
            raise ValueError(
                "Fitting the detector sampling needs the convergence semiangle, which this acquisition does not "
                "record: set data.calibration.semiangle_mrad = ... (mrad) first."
            )
        fit = detector.bright_field_disk(self._dataset, semiangle_mrad)
        self._dataset._record_bright_field_disk(fit)
        return fit

    def table(self) -> "pd.DataFrame":
        """Value, unit and source (recorded, overridden or not recorded) of each calibration.

        Rows, units and sources are the one label set of ``io.metadata_display`` that ``metadata_table()`` and
        ``DirectPtycho.calibration()`` use too.
        """
        from quantem.gpu.io.metadata_display import calibration_table

        return calibration_table(self._dataset.metadata)

    def __repr__(self) -> str:
        return self.table().to_string()

    def _repr_html_(self) -> str:
        return self.table().to_html(escape=True)


def _unknown_calibration(name: str) -> str:
    """The error for a calibration field that does not exist, naming the closest one."""
    import difflib

    fields = [*Calibration.FIELDS, *Calibration.FITTED]
    close = difflib.get_close_matches(name, fields, n=1, cutoff=0.6)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"Calibration has no field {name!r}.{hint} Fields: {', '.join(fields)}."


class _Metadata(dict):
    """Dictionary with a notebook view; never keep detector storage alive."""

    def __init__(self, values=(), dataset=None):
        super().__init__(values)
        self._dataset = None if dataset is None else ref(dataset)

    def _repr_html_(self) -> str | None:
        dataset = None if self._dataset is None else self._dataset()
        return None if dataset is None else dataset.metadata_table().to_html(escape=True)

    def __reduce_ex__(self, protocol):
        # Copies and serialization contain metadata only, without a resident owner.
        return dict, (dict(self),)


@dataclass(eq=False, frozen=True, slots=True, weakref_slot=True)
class Dataset4dstem:
    """Loaded acquisition with metadata and GPU tensor indexing.

    Owns the resident storage and decoder. Selecting scan/detector regions
    returns ordinary PyTorch tensors; calibration remains in ``metadata``.
    The read-only ``sampling``, ``units`` and ``origin`` properties expose
    that metadata in axis order without decoding data or keeping another copy.

    ``data`` is a GPU tensor or array, or an encoded resident owner from any
    backend; algorithm packages may also wrap their own arrays. The
    shape, dtype and size queries therefore read ``metadata`` first and fall
    back to whatever the owner reports.

    ``metadata`` is a dictionary that displays the acquisition table in Jupyter.
    Its outer mapping is copied on construction; detector storage and nested
    provenance are not copied. Edit metadata through ``data.metadata``.
    """

    data: object
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _Metadata(self.metadata, self))

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
        >>> data.metadata
        >>> data.metadata["voltage_kV"]
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

    @classmethod
    def from_blocks(
        cls, source, *, dtype: str = "float32",
        device: str | torch.device | None = None, path: str | Path | None = None,
        verbose: bool = False,
    ) -> "Dataset4dstem":
        """Store ordered scientific results without a complete decoded copy.

        Algorithms can construct an output through their supplied dataset's
        class without importing a reader or codec.

        Parameters
        ----------
        source
            Object with ``shape``, ``dtype`` and ``blocks()`` yielding ordered
            Torch tensors of shape ``(frames, detector_row, detector_col)``.
            Optional ``save_metadata`` is retained by the ordinary IO path.
        dtype : str
            ``"float32"`` preserves computed values; ``"scaled_uint16"``
            explicitly accepts calibrated rounding with a precision report.
        device : str or torch.device, optional
            Destination device. Defaults to ``source.device``.
        path : str or pathlib.Path, optional
            Write bounded blocks to this HDF5 file and reopen the saved acquisition.
        verbose : bool
            Show frame progress and finalization while writing.

        Returns
        -------
        Dataset4dstem
            Encoded result on the chosen device.

        Examples
        --------
        >>> merged = type(acquisition).from_blocks(regions, dtype="scaled_uint16")
        """
        from quantem.gpu.io.load import load
        from quantem.gpu.io.save import save

        device = str(source.device if device is None else device)
        if path is None:
            return load(source, dtype=dtype, device=device, verbose=False)
        save(path, source, dtype=dtype, backend=device.split(":")[0], verbose=verbose)
        return load(
            path, dtype="native" if dtype == "float32" else dtype,
            device=device, verbose=False,
        )

    def browse(
        self,
        *,
        scan_region: tuple[int, int, int, int] | Literal[
            "top left", "top right", "middle", "bottom left", "bottom right"
        ] | None = None,
        scan_region_shape: tuple[int, int] = (64, 64),
        interactive: bool = False,
        size: str = "medium",
    ):
        """Show the scan and optionally box a region for reconstruction.

        For a named region, ``scan_region_shape`` gives an even square in scan positions.
        A tuple ``scan_region=(row_start, row_stop, col_start, col_stop)`` sets the bounds
        directly, with exclusive stops.

        interactive=False (quantem.widget's ShowDirect): the virtual bright field and annular dark field and the mean
        diffraction pattern with the bright-field disk drawn (detector.bright_field_disk on the mean pattern), the box
        a square to drag. interactive=True (Show4DSTEM): every pattern, a virtual detector to drag on the pattern, the
        box as its Rect on the virtual image. Either way the box, as dragged, is ``acquisition.scan_region`` in the next
        cell. Nothing is loaded again. Why: a scientist looks at where the region sits before direct ptychography or
        any reconstruction, and picks it by hand.

        >>> acquisition.browse(scan_region="bottom right", scan_region_shape=(64, 64))
        >>> acquisition.browse(scan_region=(224, 288, 224, 288), interactive=True)"""
        if isinstance(scan_region, str):
            scan_region = _browse_region(self.shape[:2], scan_region, scan_region_shape)
        if interactive:
            from quantem.widget import Show4DSTEM

            view = Show4DSTEM(self)
            view.show_direct = False  # browsing only: direct ptychography is its own step
            if scan_region is not None:  # the Rect's centre and size, set so selected_scan_region() gives scan_region back exactly
                row_start, row_stop, col_start, col_stop = (int(value) for value in scan_region)
                view.vi_roi_mode = "rect"
                view.vi_roi_center_row, view.vi_roi_center_col = (row_start + row_stop - 1) / 2, (col_start + col_stop - 1) / 2
                view.vi_roi_center = [view.vi_roi_center_row, view.vi_roi_center_col]
                view.vi_roi_height, view.vi_roi_width = row_stop - row_start - 1, col_stop - col_start - 1
            self.scan_region = view.selected_scan_region() if scan_region is not None else None
            view.observe(lambda change: setattr(self, "scan_region", view.selected_scan_region() if view.vi_roi_mode == "rect" else None),
                         names=["vi_roi_mode", "vi_roi_center_row", "vi_roi_center_col", "vi_roi_height", "vi_roi_width"])
            return view
        from quantem.gpu import detector
        from quantem.widget import ShowDirect

        disk = detector.bright_field_disk(self, semiangle_mrad=getattr(self, "semiangle_mrad", None))
        center, radius, edge = disk.bright_field_center_px, disk.bright_field_radius_px, disk.bright_field_edge_radius_px
        bright = detector.bf(self, bright_field_center_px=center, bright_field_radius_px=radius)
        dark = detector.adf(self, bright_field_center_px=center, bright_field_edge_radius_px=edge)
        step = getattr(self, "scan_sampling_A", None)
        step = float(step[0] if isinstance(step, (tuple, list)) else step) if step is not None else 1.0
        sampling = disk.detector_sampling_mrad
        name = str(self.metadata.get("source_path", "")) if isinstance(getattr(self, "metadata", None), dict) else ""
        self.scan_region = scan_region
        view = ShowDirect(bright, dark, scan_sampling_A=step, region=self.scan_region,
                          editable=True, patterns=[disk.mean_pattern], pattern_labels=["mean pattern"], beam_center_px=center,
                          disk_radius_px=edge, mrad_per_px=0.0 if sampling is None else float(sampling[0]),
                          title=Path(name).name.removesuffix("_master.h5") if name else "", size=size)
        view.observe(lambda change: setattr(self, "scan_region", None if change.new is None else tuple(int(value) for value in change.new)), names=["region"])
        return view

    def mean(self, dim: tuple[int, int]) -> torch.Tensor:
        """Return a float32 scan image or mean diffraction pattern as a tensor.

        Cached encoded-count reductions are reused without decoding the scan.
        Integer totals are rounded only after division.

        Parameters
        ----------
        dim : tuple of int
            ``(0, 1)`` averages scan positions; ``(2, 3)`` averages detector
            pixels. Other reductions are not supported by this method.

        Returns
        -------
        torch.Tensor
            Two-dimensional mean on the acquisition's device.

        Examples
        --------
        >>> mean_pattern = acquisition.mean(dim=(0, 1))
        >>> scan_image = acquisition.mean(dim=(2, 3))
        """
        dim = tuple(dim)
        if dim not in ((0, 1), (2, 3)):
            raise ValueError("Use dim=(0, 1) for the pattern or dim=(2, 3) for the scan image.")
        if self.ndim != 4:
            raise ValueError(f"Select one 4D-STEM acquisition before mean(); got {self.shape}.")
        if self.device is None or self.device.type == "cpu":
            rows, columns, *detector_shape = self.shape
            result_t = torch.zeros(
                detector_shape if dim == (0, 1) else (rows, columns), dtype=torch.float64,
            )
            for row in range(0, rows, 16):
                stop = min(row + 16, rows)
                values_t = self.read(scan_region=(row, stop, 0, columns)).to(torch.float64)
                if dim == (0, 1):
                    result_t += values_t.sum(dim=dim)
                else:
                    result_t[row:stop] = values_t.mean(dim=dim)
            if dim == (0, 1):
                result_t /= rows * columns
            return result_t.to(torch.float32)

        from quantem.gpu.detector import prepare
        from quantem.gpu.io.read import torch_value

        method = "mean_dp_device" if dim == (0, 1) else "detector_mean_device"
        cached_mean = getattr(self.data, method, None)
        if callable(cached_mean):
            return torch_value(cached_mean())
        session = prepare(self)
        if dim == (0, 1):
            return torch_value(session.mean_pattern(output="native"))
        total_t = torch_value(
            session.masked_sum(np.ones(session.detector_shape, dtype=bool), output="native")
        )
        if not total_t.is_floating_point():
            total_t = total_t.to(torch.float64)
        return (total_t / prod(session.detector_shape)).to(torch.float32)

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
    def calibration(self) -> Calibration:
        """Microscope calibration: read or set ``voltage_kV``, ``semiangle_mrad``, ``scan_sampling_A``, ...

        Displays as a table of value, unit and source. See ``Calibration``.

        Examples
        --------
        >>> data = io.load("scan_master.h5")  # doctest: +SKIP
        >>> data.calibration.voltage_kV = 300  # doctest: +SKIP
        >>> data.calibration  # doctest: +SKIP
        """
        return Calibration(self)

    def _record_calibration(
        self, *, voltage_kV: float | None = None, semiangle_mrad: float | None = None,
        scan_sampling_A: float | tuple[float, float] | None = None,
        detector_sampling_mrad: float | tuple[float, float] | None = None,
        detector_sampling_inv_A: float | tuple[float, float] | None = None,
        evidence: str = "Stated by the user",
        provenance: str = qem_metadata.USER_OVERRIDE,
    ) -> None:
        """Record user calibration beside, never in place of, the acquisition's own record.

        A single number for a sampling applies to both (row, col) axes. ``provenance`` is
        ``qem_metadata.USER_OVERRIDE`` for a value set by hand, ``qem_metadata.BRIGHT_FIELD_DISK_FIT``
        for the detector sampling ``detector_sampling`` records.
        """
        scan_sampling_A, detector_sampling_mrad, detector_sampling_inv_A = (
            None if value is None else (value, value) if np.ndim(value) == 0 else value
            for value in (scan_sampling_A, detector_sampling_mrad, detector_sampling_inv_A)
        )
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Provide nonempty evidence describing the calibration.")
        if detector_sampling_mrad is not None and detector_sampling_inv_A is not None:
            raise ValueError("Provide detector sampling in mrad or inverse angstrom, not both.")
        scientific = qem_metadata.acquisition_metadata(self.shape, self.metadata)
        # Stored scientific metadata uses JSON values. HDF5 tags can still be
        # NumPy arrays in memory; normalize their representation without
        # dropping tags or changing their values before retaining this record.
        scientific = json.loads(json.dumps(scientific, default=qem_metadata.json_metadata))
        overrides = scientific.setdefault("calibration_overrides", {})

        def record(path, value, unit):
            overrides[path] = _microscopy_quantity(dict(
                value=value, unit=unit, provenance=provenance, evidence=evidence.strip()
            ), path)

        for path, value, unit in (
            ("electron_source/accelerating_voltage", voltage_kV, "kV"),
            ("illumination_system/semi_convergence_angle", semiangle_mrad, "mrad"),
        ):
            if value is not None:
                record(path, value, unit)
        for paths, values, unit in (
            (qem_metadata.SCAN_PIXEL_SIZE, scan_sampling_A, "angstrom"),
            (qem_metadata.DETECTOR_PIXEL_SIZE, detector_sampling_mrad, "mrad"),
            (qem_metadata.DETECTOR_PIXEL_SIZE, detector_sampling_inv_A, "1/angstrom"),
        ):
            if values is None:
                continue
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError("Provide sampling as a (row, col) pair.")
            for path, value in zip(paths, values, strict=True):
                record(path, value, unit)
        updated = qem_metadata.effective_metadata(self.metadata, scientific)
        # Commit only after all values, units and paired axes pass validation.
        updated["scientific_metadata"] = scientific
        for key in qem_metadata.FLAT_KEYS:
            self.metadata.pop(key, None)
        self.metadata.pop("calibration_overrides", None)
        self.metadata.update(updated)

    def _record_bright_field_disk(self, fit: "BrightFieldDisk") -> None:
        """Record a bright-field disk fit: its centre and edge radius, and its detector sampling unless set by hand."""
        overrides = self.metadata.get("scientific_metadata", {}).get("calibration_overrides", {})
        by_hand = overrides.get(qem_metadata.DETECTOR_PIXEL_SIZE[0], {}).get("provenance") == qem_metadata.USER_OVERRIDE
        if by_hand:
            key = next(key for key in qem_metadata.DETECTOR_SAMPLING_KEYS.values() if self.metadata.get(key) is not None)
            unit = "mrad" if key == "detector_sampling_mrad" else "1/angstrom"
            print(f"Calibration: kept the detector sampling set by hand, {tuple(self.metadata[key])} {unit} per pixel; "
                  f"the bright-field disk gives {fit.detector_sampling_mrad[0]:.6g} mrad per pixel.")
        else:
            self._record_calibration(
                detector_sampling_mrad=fit.detector_sampling_mrad, provenance=qem_metadata.BRIGHT_FIELD_DISK_FIT,
                evidence=(f"semiangle {fit.semiangle_mrad:g} mrad / bright-field edge radius "
                          f"{fit.bright_field_edge_radius_px:.6g} px ({fit.method})"),
            )
        self.metadata[qem_metadata.BRIGHT_FIELD_DISK_KEY] = {
            "schema": qem_metadata.BRIGHT_FIELD_DISK_SCHEMA, "method": fit.method,
            "bright_field_center_px": list(fit.bright_field_center_px),
            "bright_field_edge_radius_px": fit.bright_field_edge_radius_px,
        }

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
        >>> loaded = load("scan-lossless.h5", device="mps")
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
        output: str = "torch",
    ):
        """Read one logical region on its GPU, preserving every measurement bit.

        Regions use ``(row_start, row_stop, col_start, col_stop)`` with
        exclusive stops. The resident representation, decoding, and transfer
        scheduling remain automatic. A complete read is allowed only when its
        dense tensor fits the accelerator's current working memory.

        ``output='torch'`` returns a Torch tensor. ``output='native'`` returns
        a CuPy array on CUDA or an owned Metal array with ``get()``/``release()``
        on MPS. Use native output for float64 Metal data, since Torch MPS has
        no float64 tensors. Close a Metal output after its consumer finishes.
        """
        return read(
            self,
            scan_region=scan_region,
            detector_region=detector_region,
            output=output,
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


# The microscope values a user states are read and set on the acquisition itself, ``data.semiangle_mrad = 30``: the
# same record ``data.calibration`` keeps, with its units, sources and checks. The dataclass stays frozen for its own
# fields (``data``, ``metadata``); only these names pass through.
MICROSCOPE = ("voltage_kV", "semiangle_mrad", "scan_sampling_A")


def _microscope_value(name: str) -> property:
    return property(lambda self: getattr(self.calibration, name),
                    doc=f"``calibration.{name}``, read and set on the acquisition: ``data.{name} = ...``.")


def _set_microscope_or_frozen(self, name: str, value) -> None:
    if name in MICROSCOPE:
        setattr(self.calibration, name, value)
    elif name == "scan_detector_rotation_deg":
        # a rotation known for the session (written down at the microscope): recorded with the acquisition, so
        # DirectPtycho takes it as recorded and never estimates or moves it; None forgets it
        if value is None:
            self.metadata.pop("scan_detector_rotation_deg", None)
        else:
            self.metadata["scan_detector_rotation_deg"] = float(value) % 360.0
    elif name == "scan_region":
        # the region picked by hand in browse(): its box keeps this up to date as it is dragged; None forgets it
        if value is None:
            self.metadata.pop("scan_region", None)
        else:
            self.metadata["scan_region"] = tuple(int(edge) for edge in value)
    else:
        _frozen_setattr(self, name, value)


for _name in MICROSCOPE:
    setattr(Dataset4dstem, _name, _microscope_value(_name))
Dataset4dstem.scan_detector_rotation_deg = property(
    lambda self: self.metadata.get("scan_detector_rotation_deg"),
    doc="Scan-to-detector rotation in degrees when known, else None: ``data.scan_detector_rotation_deg = 169``.")
Dataset4dstem.scan_region = property(
    lambda self: self.metadata.get("scan_region"),
    doc="The region picked in browse(), (row start, row stop, col start, col stop), else None: the box as last dragged.")
_frozen_setattr = Dataset4dstem.__setattr__
Dataset4dstem.__setattr__ = _set_microscope_or_frozen


def resident_metadata(shape, dtype, resident_bytes: int, *, backend: str, representation: str = "encoded") -> dict:
    """Return the metadata fields every resident loader records about what it holds.

    ``Dataset4dstem`` answers ``shape``, ``dtype`` and ``resident_bytes``
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
