"""Capture native dataset calibration at storage boundaries."""

from copy import deepcopy
from math import isclose

from quantem.core.datastructures import Dataset


def dataset_metadata(data: Dataset) -> dict:
    """Snapshot current native axes and preserve original source provenance.

    For example, saving ``data[4:12, 8:16]`` records the selected shape and
    origin rather than reusing the full acquisition's axis sizes.
    """
    metadata = deepcopy(data.metadata)
    metadata.update(
        name=data.name,
        sampling=data.sampling.tolist(),
        origin=data.origin.tolist(),
        units=list(data.units),
        signal_units=data.signal_units,
        working_shape=list(data.shape),
    )
    if data.ndim != 4:
        return metadata
    metadata.update(
        scan_shape=list(data.shape[:2]), n_frames=data.shape[0] * data.shape[1]
    )
    lengths = {"angstrom": 1, "A": 1, "Å": 1, "nm": 10, "pm": 0.01, "m": 1e10}
    reciprocals = {"1/angstrom": 1, "1/A": 1, "1/Å": 1, "1/nm": 0.1, "1/m": 1e-10}
    axis_sampling = []
    for axis, (value, unit) in enumerate(zip(data.sampling, data.units)):
        factors = lengths if axis < 2 else reciprocals
        if unit in factors:
            axis_sampling.append(
                {
                    "value": abs(float(value)) * factors[unit],
                    "unit": "angstrom" if axis < 2 else "1/angstrom",
                    "provenance": "native_dataset_calibration",
                }
            )
        elif axis >= 2 and unit in {"mrad", "rad"}:
            axis_sampling.append(
                {
                    "value": abs(float(value)) * (1000 if unit == "rad" else 1),
                    "unit": "mrad",
                    "provenance": "native_dataset_calibration",
                }
            )
        else:
            axis_sampling.append(None)
    for first, key in ((0, "scan_sampling_A"), (2, "detector_sampling")):
        pair = axis_sampling[first : first + 2]
        if all(pair) and pair[0]["unit"] == pair[1]["unit"]:
            metadata[key] = [quantity["value"] for quantity in pair]
            if first == 2:
                metadata["detector_sampling_unit"] = pair[0]["unit"]
        else:
            metadata.pop(key, None)
    if "scientific_metadata" in metadata:
        from ._qem_metadata import microscopy_metadata

        scientific = microscopy_metadata(metadata["scientific_metadata"])
        overrides = scientific.get("calibration_overrides", {})
        for first, prefix in (
            (0, "scan_controller/regular_scan/pixel_size_"),
            (2, "imaging_system/reciprocal_pixel_size_"),
        ):
            keys = [prefix + suffix for suffix in ("row", "column")]
            pair = axis_sampling[first : first + 2]
            if any(key in overrides for key in keys) and any(
                sampling is None
                or overrides.get(key, {}).get("unit") != sampling["unit"]
                or not isclose(
                    overrides[key]["value"], sampling["value"], rel_tol=1e-12
                )
                for key, sampling in zip(keys, pair)
            ):
                # A selection or recalibration supersedes the original pair.
                # Retaining it would replace the current axes when reopened.
                for key in keys:
                    overrides.pop(key, None)
        for axis, size, sampling in zip(scientific["axes"], data.shape, axis_sampling):
            axis["size"] = int(size)
            if sampling is None:
                axis.pop("sampling", None)
            elif (
                axis.get("sampling", {}).get("value") != sampling["value"]
                or axis.get("sampling", {}).get("unit") != sampling["unit"]
            ):
                axis["sampling"] = sampling
        microscope = scientific.get("electron_microscope", {})
        for axis, suffix in zip(scientific["axes"], ("row", "column")):
            key = f"scan_controller/regular_scan/pixel_size_{suffix}"
            if "sampling" in axis:
                microscope[key] = dict(axis["sampling"])
            else:
                microscope.pop(key, None)
        metadata["scientific_metadata"] = scientific
    return metadata
