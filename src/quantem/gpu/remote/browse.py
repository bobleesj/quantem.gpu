"""Exact detector products of the served acquisitions for native 4D-STEM viewers.

:class:`BrowseService` joins the catalog of the served folder to the GPU
residency of its acquisitions and computes, from exact integer counts, every
virtual image, custom detector and diffraction pattern a plan asks for.
"""

import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from quantem.gpu import detector, dpc
from quantem.gpu.remote.catalog import Catalog
from quantem.gpu.remote.plan import BrowsePlan
from quantem.gpu.remote.residency import (
    CACHE_FRACTION,
    Residency,
    ResidentAcquisition,
    open_gpus,
)

PROTOCOL_NAME = "quantem-gpu-browse"
PROTOCOL_VERSION = 1
DETECTOR_MODES = {"BF", "ABF", "ADF", "HAADF", "DF"}
CENTER_OF_MASS_MODES = {"CoMx", "CoMy", "CoMmag", "DPC", "iCoM"}
SCAN_BINS = {1, 2, 4, 8, 16}
MAX_IMAGE_ENTRIES = 64


class BrowseService:
    """Serve one data folder's acquisitions from a pool of CUDA GPUs.

    Parameters
    ----------
    data_folder : str or os.PathLike
        Root of the served acquisitions.
    gpus : sequence of int or "auto"
        CUDA device indices of the pool, or ``"auto"`` for every visible
        device. A device that cannot initialize is left out and reported in
        :meth:`capabilities`.

    Examples
    --------
    >>> service = BrowseService("/data", gpus="auto")
    >>> path, plan = service.plan("detector/session", "scan_master.h5", det_bin=2, scan_bin=1, scan_region=None)
    >>> image = service.compute(path, lambda entry: service.virtual_image(entry, plan, mode="BF", inner=0, outer=1))
    """

    def __init__(
        self,
        data_folder: str | os.PathLike[str],
        *,
        gpus: Sequence[int] | str = (0,),
    ) -> None:
        if isinstance(gpus, str):
            if gpus != "auto":
                raise ValueError("gpus must be 'auto' or a sequence of CUDA indices")
            requested = None
        else:
            requested = tuple(dict.fromkeys(int(value) for value in gpus))
            if not requested or any(value < 0 for value in requested):
                raise ValueError("CUDA device indices must be zero or greater")
        self.data_folder = Path(data_folder).expanduser().resolve()
        self.catalog = Catalog(self.data_folder)
        pool, self.device_error = open_gpus(requested)
        self.residency = Residency(self.catalog, pool)
        self._image_lock = threading.Lock()
        self._images: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def capabilities(self) -> dict[str, object]:
        """Describe the protocol, the GPU pool with live capacity, and the features."""
        gpus = list(self.residency.gpus.values())
        devices = self.residency.devices()
        return {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "backend": "cuda" if gpus else None,
            "device_name": gpus[0].name if gpus else None,
            "device_error": self.device_error,
            "browse_gpu": gpus[0].index if gpus else None,
            "browse_gpus": [gpu.index for gpu in gpus],
            "cache_fraction": CACHE_FRACTION,
            "cache_budget_bytes": max((gpu.cache_budget_bytes for gpu in gpus), default=0),
            "aggregate_cache_budget_bytes": sum(gpu.cache_budget_bytes for gpu in gpus),
            "largest_device_memory_bytes": max(
                (gpu.total_memory_bytes for gpu in gpus), default=None
            ),
            "devices": devices,
            "data_folders": [str(self.data_folder)],
            "features": {
                "catalog_refresh": True,
                "selected_diffraction": True,
                "virtual_detectors": True,
                "custom_detector": True,
                "acquisition_events": True,
                "exact_integer_images": True,
                "multi_gpu_residency": len(devices) > 1,
            },
        }

    def plan(
        self,
        session: str,
        filename: str,
        *,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
    ) -> tuple[Path, BrowsePlan]:
        """Resolve the master and check a bin and crop request before any residency change.

        A request that cannot be served fails here, so it never evicts or
        reloads the acquisition the client is working on.
        """
        path = self.catalog.resolve_master(session, filename)
        if scan_bin not in SCAN_BINS:
            raise HTTPException(400, "scan_bin must be 1, 2, 4, 8, or 16")
        inspection = self.catalog.inspect(path)
        if inspection.scan_shape is None or inspection.detector_shape is None:
            raise HTTPException(422, "The master does not report a usable 4D shape.")
        scan_rows, scan_cols = (int(value) for value in inspection.scan_shape)
        detector_rows, detector_cols = (int(value) for value in inspection.detector_shape)
        det_bin = max(1, int(det_bin))
        if detector_rows % det_bin or detector_cols % det_bin:
            raise HTTPException(
                400,
                f"det_bin={det_bin} does not divide the {detector_rows} x "
                f"{detector_cols} detector; choose a bin that divides both sizes.",
            )
        if scan_region is not None:
            if scan_region[1] > scan_rows or scan_region[3] > scan_cols:
                raise HTTPException(400, "scan crop exceeds the source scan shape")
            if scan_region == (0, scan_rows, 0, scan_cols):
                scan_region = None
        return path, BrowsePlan(
            (scan_rows, scan_cols),
            (detector_rows, detector_cols),
            det_bin,
            int(scan_bin),
            scan_region,
        )

    def compute(self, path: Path, operation: Callable[[ResidentAcquisition], object]) -> object:
        """Load the acquisition if needed and keep it resident through one calculation.

        An acquisition evicted between load and calculation raises 409; load
        it again, up to three times, before reporting the conflict.
        """
        if not self.residency.gpus:
            raise HTTPException(503, f"CUDA unavailable: {self.device_error}")
        for attempt in range(3):
            entry = self.residency.entry(path, reserve=True)
            try:
                result = operation(entry)
            except HTTPException as exc:
                if exc.status_code != 409 or attempt == 2:
                    raise
            else:
                self.residency.activate(entry.key)
                return result
            finally:
                self.residency.release(entry)

    # --- Detector products

    def virtual_image(
        self,
        entry: ResidentAcquisition,
        plan: BrowsePlan,
        *,
        mode: str,
        inner: float,
        outer: float,
        center_row: float | None = None,
        center_column: float | None = None,
    ) -> np.ndarray:
        """Return one exact virtual image or centre-of-mass product of a plan.

        ``inner`` and ``outer`` are fractions of the bright-field radius fitted
        on the plan's binned detector; the detector center defaults to the
        fitted bright-field center. Centre-of-mass products are in binned
        detector pixels.
        """
        with self.residency.pinned(entry):
            if mode in CENTER_OF_MASS_MODES:
                com_row, com_column = self._center_of_mass(entry, plan)
                if mode == "CoMy":
                    return com_row
                if mode == "CoMx":
                    return com_column
                if mode in {"CoMmag", "DPC"}:
                    return np.hypot(com_row, com_column).astype(np.float32, copy=False)
                return np.asarray(dpc.integrate(com_row, com_column), dtype=np.float32)
            fit_row, fit_column, radius = self._bf_geometry(entry, plan)
            row = fit_row if center_row is None else float(center_row)
            column = fit_column if center_column is None else float(center_column)
            inner_pixels = max(0.0, float(inner) * radius)
            outer_pixels = max(inner_pixels + 1.0, float(outer) * radius)
            detector_rows, detector_columns = plan.detector_shape
            rows, columns = np.ogrid[:detector_rows, :detector_columns]
            distance_squared = (rows - row) ** 2 + (columns - column) ** 2
            mask = distance_squared <= outer_pixels**2
            if mode != "BF":
                mask &= distance_squared >= inner_pixels**2
            return plan.scan_image(entry.session.masked_sum_exact(plan.source_mask(mask)))

    def custom_detector(
        self,
        entry: ResidentAcquisition,
        plan: BrowsePlan,
        *,
        center_row: float,
        center_column: float,
        inner_radius: float,
        outer_radius: float,
        shape: str = "annulus",
    ) -> np.ndarray:
        """Return the exact count sum inside one custom detector, radii in binned pixels."""
        with self.residency.pinned(entry):
            detector_rows, detector_columns = plan.detector_shape
            rows, columns = np.ogrid[:detector_rows, :detector_columns]
            if shape == "square":
                mask = (
                    (np.abs(rows - center_row) <= outer_radius)
                    & (np.abs(columns - center_column) <= outer_radius)
                )
            else:
                distance_squared = (rows - center_row) ** 2 + (columns - center_column) ** 2
                mask = distance_squared <= outer_radius**2
                if shape == "annulus":
                    mask &= distance_squared >= inner_radius**2
            return plan.scan_image(entry.session.masked_sum_exact(plan.source_mask(mask)))

    def selected_diffraction(
        self,
        entry: ResidentAcquisition,
        plan: BrowsePlan,
        *,
        scan_row: int,
        scan_column: int,
    ) -> np.ndarray:
        """Return the exact binned diffraction pattern at one position of the plan, clamped.

        One native position reads a single frame, which is cheaper than a
        selected-frame reduction; a scan bin adds the patterns it covers.
        """
        with self.residency.pinned(entry):
            scan_rows, scan_columns = plan.scan_shape
            row = max(0, min(scan_rows - 1, int(scan_row)))
            column = max(0, min(scan_columns - 1, int(scan_column)))
            positions = plan.positions(row, column)
            if plan.scan_bin == 1:
                pattern = entry.session.frame(int(positions[0]))
            else:
                pattern = entry.session.reduce_frames_exact(positions)
            return plan.diffraction(pattern)

    def cached_image(self, key: tuple) -> np.ndarray | None:
        """Return a recently computed image so repeated requests skip the device."""
        with self._image_lock:
            image = self._images.get(key)
            if image is not None:
                self._images.move_to_end(key)
            return image

    def store_image(self, key: tuple, image: np.ndarray) -> None:
        """Remember one image, keeping only the most recent few."""
        with self._image_lock:
            self._images[key] = image
            self._images.move_to_end(key)
            while len(self._images) > MAX_IMAGE_ENTRIES:
                self._images.popitem(last=False)

    def close(self) -> None:
        """Stop loading and release every resident acquisition and pooled device block."""
        self.residency.close()

    # --- What a plan derives once per resident

    def _bf_geometry(
        self, entry: ResidentAcquisition, plan: BrowsePlan
    ) -> tuple[float, float, float]:
        """Return the plan's bright-field (row, col, radius) in binned pixels, fitted once.

        Call with the entry pinned. The fit uses the mean pattern of the binned,
        cropped view: the exact count sum divided by the number of binned
        positions in float64, rounded once to float32 like every mean pattern.
        """
        key = (plan, "bf_geometry")
        if key not in entry.derived:
            total = plan.diffraction(
                entry.session.reduce_frames_exact(plan.positions(0, 0, *plan.scan_shape))
            )
            mean_dp = (total / math.prod(plan.scan_shape)).astype(np.float32)
            center, radius = detector.fit_probe(mean_dp)
            entry.derived[key] = (float(center[0]), float(center[1]), float(radius))
        return entry.derived[key]

    def _center_of_mass(
        self, entry: ResidentAcquisition, plan: BrowsePlan
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the plan's (row, col) centre of mass in binned detector pixels, once.

        Call with the entry pinned. Each value is the exact integer moment over
        the exact total count, rounded to float32, and 0 where a pattern has no
        counts, as the dense CUDA kernel computed it. A native-detector,
        unbinned view takes these from the session. A binned view needs moments
        in binned detector coordinates added over scan bins, so the session's
        exact sums weighted by the binned row and column index are scan-binned
        before the division.
        """
        key = (plan, "center_of_mass")
        if key in entry.derived:
            return entry.derived[key]
        if plan.detector_bin == 1 and plan.scan_bin == 1:
            com_row, com_column = entry.session.center_of_mass()
            entry.derived[key] = (plan.crop(com_row), plan.crop(com_column))
            return entry.derived[key]
        binned_rows, binned_columns = np.indices(plan.source_detector_shape) // plan.detector_bin
        total, row_moment, column_moment = (
            plan.scan_image(entry.session.weighted_sum_exact(weights))
            for weights in (np.ones_like(binned_rows), binned_rows, binned_columns)
        )
        entry.derived[key] = (
            _coordinate(row_moment, total),
            _coordinate(column_moment, total),
        )
        return entry.derived[key]


def _coordinate(moment: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Divide an exact detector moment by the exact count total as float32, 0 where empty."""
    result = np.zeros(total.shape, dtype=np.float64)
    np.divide(moment.astype(np.float64), total.astype(np.float64), out=result, where=total > 0)
    return result.astype(np.float32)
