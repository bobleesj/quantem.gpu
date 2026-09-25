"""Scientific metadata shared by QEM readers, independent of payload codecs."""

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

MAGIC = b"QEMDATA1"
CONTAINER = "quantem.qem"
CONTAINER_VERSION = 1
LEGACY_SCHEMA = "quantem.scientific-metadata/1"
SCHEMA = "quantem.scientific-metadata/2"
AXIS_NAMES = ("scan_row", "scan_column", "detector_row", "detector_column")

# Conversion factors are source unit -> public scientific-metadata unit.
_LENGTH = ("angstrom", {"m": 1e10, "nm": 10, "angstrom": 1, "Å": 1})
_DETECTOR = {
    "rad": ("mrad", 1000), "mrad": ("mrad", 1),
    "1/nm": ("1/angstrom", 0.1), "1/Å": ("1/angstrom", 1),
    "1/angstrom": ("1/angstrom", 1),
}
_QUANTITIES = {
    "electron_source/accelerating_voltage": ("kV", {"V": 0.001, "kV": 1}),
    "electron_source/beam_energy": ("keV", {"eV": 0.001, "keV": 1}),
    "illumination_system/semi_convergence_angle": ("mrad", {"rad": 1000, "mrad": 1}),
    "scan_controller/regular_scan/dwell_time": ("us", {"s": 1e6, "ms": 1000, "us": 1}),
    "imaging_system/camera_length": ("mm", {"m": 1000, "cm": 10, "mm": 1}),
    "scan_controller/regular_scan/pixel_size_row": _LENGTH,
    "scan_controller/regular_scan/pixel_size_column": _LENGTH,
}


def _microscopy_quantity(quantity: dict, path: str) -> dict:
    """Convert a known quantity without altering its evidence or source record."""
    if not isinstance(quantity, dict) or not isinstance(quantity.get("unit"), str):
        raise ValueError(f"Invalid QEM quantity at {path}; supply a value and unit.")
    result = dict(quantity)
    unit = quantity.get("unit")
    if path == "detector" or path.startswith("imaging_system/reciprocal_pixel_size_"):
        conversion = _DETECTOR.get(unit)
    else:
        if path != "scan" and path not in _QUANTITIES:
            raise ValueError(f"Unknown QEM calibration quantity {path!r}; update the reader.")
        target, factors = _LENGTH if path == "scan" else _QUANTITIES[path]
        factor = factors.get(unit)
        conversion = None if factor is None else (target, factor)
    value = quantity.get("value")
    if (conversion is None or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value <= 0 or not math.isfinite(value * conversion[1])):
        raise ValueError(f"Invalid QEM quantity {path}: {quantity!r}; check value and unit.")
    result.update(value=value * conversion[1], unit=conversion[0])
    return result


def microscopy_metadata(scientific: dict) -> dict:
    """Convert saved metadata to schema 2 without modifying source fields.

    Parameters
    ----------
    scientific : dict
        Schema-1 or schema-2 scientific metadata from a validated header.

    Returns
    -------
    dict
        Independent metadata with microscopy units and unchanged source fields.

    Raises
    ------
    ValueError
        A known quantity has an unsupported unit or invalid physical value.

    Examples
    --------
    >>> microscopy_metadata({"schema": LEGACY_SCHEMA})["schema"] == SCHEMA
    True
    """
    result = copy.deepcopy(scientific)
    if result.get("schema") not in (LEGACY_SCHEMA, SCHEMA):
        raise ValueError("Unsupported QEM metadata schema; update the reader.")
    for axis in result.get("axes", []):
        if "sampling" in axis:
            kind = "scan" if axis["name"].startswith("scan_") else "detector"
            axis["sampling"] = _microscopy_quantity(axis["sampling"], kind)
    for section in ("electron_microscope", "calibration_overrides"):
        if not isinstance(result.get(section, {}), dict):
            raise ValueError(f"QEM {section} must contain named quantities.")
        for path, quantity in result.get(section, {}).items():
            if path in _QUANTITIES or path.startswith("imaging_system/reciprocal_pixel_size_"):
                result[section][path] = _microscopy_quantity(quantity, path)
    result["schema"] = SCHEMA
    return result


def _legacy_overrides(scientific: dict) -> dict:
    """Adapt public schema-2 units to the existing calculation boundary."""
    overrides = copy.deepcopy(scientific.get("calibration_overrides", {}))
    if scientific.get("schema") == SCHEMA:
        for path, quantity in overrides.items():
            unit = quantity.get("unit")
            target, factor = {
                "angstrom": ("m", 1e-10), "kV": ("V", 1000),
                "us": ("s", 1e-6), "mm": ("m", 0.001),
                "1/angstrom": ("1/Å", 1), "mrad": ("mrad", 1),
            }.get(unit, (unit, 1))
            canonical = _microscopy_quantity(quantity, path)
            if canonical["unit"] != unit:
                raise ValueError(f"Invalid schema-2 calibration override unit at {path}: {unit!r}.")
            quantity.update(value=quantity["value"] * factor, unit=target)
    _validate_overrides(overrides)
    return overrides


def json_metadata(value):
    """Preserve NumPy metadata without arbitrary-object serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"QEM metadata must contain JSON values, not {type(value).__name__}."
    )


def no_duplicate_keys(pairs):
    """Reject repeated manifest fields instead of silently keeping the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate QEM manifest field: {key}.")
        result[key] = value
    return result


def reject_constant(value):
    """Reject NaN and infinity so a manifest always round-trips exactly."""
    raise ValueError(f"QEM metadata cannot contain nonfinite number {value}.")


RETIRED_QUANTITIES = (
    "scan_controller/regular_scan/pixel_size_y",
    "scan_controller/regular_scan/pixel_size_x",
    "imaging_system/reciprocal_pixel_size_y",
    "imaging_system/reciprocal_pixel_size_x",
)


def _processing_records(metadata: dict) -> list[dict]:
    """State every operation between the source counts and the stored counts."""
    records = [dict(operation="lossless_storage", changes_measurements=False)]
    source_dtype, stored_dtype = metadata.get("source_dtype"), metadata.get("dtype")
    if source_dtype and stored_dtype and source_dtype != stored_dtype:
        source_type, stored_type = np.dtype(source_dtype), np.dtype(stored_dtype)
        exact_float = (
            source_type == np.dtype("float64")
            and stored_type == np.dtype("float32")
            and metadata.get("exact_float_narrowing", {}).get("method")
            == "float64-float32-float64-bitwise"
            and metadata.get("file_counts_exact") is True
        )
        if not exact_float and not (
            source_type.kind in "iu" and stored_type.kind in "iu"
            and stored_type.itemsize < source_type.itemsize
            and (metadata.get("file_counts_exact") is True
                 or metadata.get("working_counts_exact") is True)
        ):
            raise ValueError(
                "A dtype change needs explicit processing provenance; only a "
                "reader-verified exact narrowing can be inferred. "
                "Retain the original dtype or provide validated scientific metadata."
            )
        records.append(dict(
            operation="exact_float_narrowing" if exact_float else "exact_integer_narrowing", changes_measurements=False,
            source_dtype=str(source_dtype), stored_dtype=str(stored_dtype),
        ))
    correction = metadata.get("hot_pixel_correction")
    if isinstance(correction, dict) and correction.get("applied"):
        records.append(dict(
            operation="flagged_pixel_replacement", changes_measurements=True,
            method=str(correction.get("method")),
            pixel_count=int(correction.get("pixel_count", 0)),
        ))
    return records


def _nexus_source_metadata(metadata: dict) -> dict:
    """Gather the NXmx master fields that the HDF5 reader retains as flat keys."""
    source = {
        key: value for key, value in metadata.items()
        if isinstance(key, str) and key.startswith("entry/")
    }
    if source:
        description = str(source.get("entry/instrument/detector/description", ""))
        source["sourceFormat"] = (
            "dectris-arina-hdf5" if "ARINA" in description.upper() else "nexus-nxmx-hdf5"
        )
    return source


def acquisition_metadata(shape, metadata: dict) -> dict:
    """Normalize recorded calibration without replacing source fields."""
    if "scientific_metadata" in metadata:
        saved = copy.deepcopy(metadata["scientific_metadata"])
        validate_header(dict(container=CONTAINER, container_version=CONTAINER_VERSION,
                             codec="retained", profile="retained", shape=list(shape),
                             scientific_metadata=saved))
        return microscopy_metadata(saved)
    source = dict(metadata.get("source_metadata") or _nexus_source_metadata(metadata))
    quantities = {}

    def quantity(path, factors, output_unit, source_path=None):
        source_path = source_path or "electron_microscope/" + path
        text = str(source.get(source_path, ""))
        parts = text.split()
        unit = source.get(source_path + "@units")
        if unit is None and len(parts) == 2:
            unit = parts[1]
        try:
            value = float(parts[0]) * factors[unit]
        except (KeyError, IndexError, ValueError, TypeError):
            return
        if math.isfinite(value) and value > 0:
            quantities[path] = dict(
                value=value, unit=output_unit, provenance="source_metadata"
            )

    quantity("electron_source/accelerating_voltage", {"V": 1, "kV": 1000}, "V")
    if "electron_source/accelerating_voltage" not in quantities:
        quantity(
            "electron_source/accelerating_voltage",
            {"eV": 1, "keV": 1000},
            "V",
            "entry/instrument/detector/incident_energy",
        )
    quantity(
        "illumination_system/semi_convergence_angle", {"rad": 1000, "mrad": 1}, "mrad"
    )
    quantity(
        "scan_controller/regular_scan/dwell_time",
        {"s": 1, "ms": 0.001, "us": 1e-6, "µs": 1e-6, "μs": 1e-6},
        "s",
    )
    quantity("imaging_system/camera_length", {"m": 1, "cm": 0.01, "mm": 0.001}, "m")
    # ARINA's electron-energy setting uses an NXmx photon-energy field.
    # Do not reinterpret a generic X-ray NXmx energy as accelerating voltage.
    if "ARINA" in str(source.get("entry/instrument/detector/description", "")).upper():
        for path, field, factors, default_unit, output_unit in (
            ("electron_source/accelerating_voltage", "detectorSpecific/photon_energy",
             {"eV": 1, "keV": 1000}, "eV", "V"),
            ("scan_controller/regular_scan/dwell_time", "count_time",
             {"s": 1, "ms": 0.001, "us": 1e-6}, "s", "s"),
            ("scan_controller/regular_scan/dwell_time", "frame_time",
             {"s": 1, "ms": 0.001, "us": 1e-6}, "s", "s"),
        ):
            key = "entry/instrument/detector/" + field
            value = source.get(key)
            factor = factors.get(source.get(key + "@units", default_unit))
            if (path not in quantities and factor is not None
                    and isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value * factor) and value > 0):
                quantities[path] = dict(
                    value=float(value) * factor, unit=output_unit,
                    provenance="source_metadata", evidence=key,
                )
    # Source files name these by x and y; the saved copy names them by array axis.
    for axis, source_axis in (("row", "y"), ("column", "x")):
        quantity(
            f"imaging_system/reciprocal_pixel_size_{axis}",
            {"rad": 1000, "mrad": 1},
            "mrad",
            f"electron_microscope/imaging_system/reciprocal_pixel_size_{source_axis}",
        )
    voltage = metadata.get("voltage_kV")
    if "electron_source/accelerating_voltage" not in quantities and voltage is not None:
        value = float(voltage) * 1000
        if math.isfinite(value) and value > 0:
            quantities["electron_source/accelerating_voltage"] = dict(
                value=value, unit="V", provenance="source_metadata"
            )
    semiangle = metadata.get("semiangle_mrad")
    if "illumination_system/semi_convergence_angle" not in quantities and semiangle is not None:
        value = float(semiangle)
        if math.isfinite(value) and value > 0:
            quantities["illumination_system/semi_convergence_angle"] = dict(
                value=value, unit="mrad", provenance="source_metadata"
            )
    axes = [dict(name=name, size=int(size)) for name, size in zip(AXIS_NAMES, shape)]
    scan = metadata.get("scan_sampling_A")
    if scan is not None and len(scan) == 2:
        for axis, suffix, value in zip(axes, ("row", "column"), scan):
            value = float(value) * 1e-10
            if math.isfinite(value) and value > 0:
                sampling = dict(value=value, unit="m", provenance="source_metadata")
                axis["sampling"] = sampling
                quantities[f"scan_controller/regular_scan/pixel_size_{suffix}"] = dict(
                    sampling
                )
    detector = metadata.get(
        "detector_sampling", metadata.get("detector_sampling_inv_A")
    )
    unit = metadata.get("detector_sampling_unit", "1/angstrom")
    if detector is not None and len(detector) == 2:
        for axis, value in zip(axes[2:], detector):
            value = float(value)
            if math.isfinite(value) and value > 0:
                axis["sampling"] = dict(
                    value=value, unit=unit, provenance="source_metadata"
                )
    return microscopy_metadata(dict(
        schema=LEGACY_SCHEMA,
        axes=axes,
        electron_microscope=quantities,
        source_metadata=source,
        source_documents=copy.deepcopy(metadata.get("source_documents", [])),
        source_metadata_coverage=metadata.get("source_metadata_coverage", "reader-retained"),
        calibration_overrides=_recorded_overrides(metadata),
        processing=_processing_records(metadata),
        **_recorded_sample(metadata),
        source_format=source.get(
            "sourceFormat", metadata.get("source_kind", "unknown")
        ),
    ))


def _recorded_sample(metadata: dict) -> dict:
    """The declared specimen supplied with a new copy (``sample``: a session's dataset.yaml at conversion), validated;
    nothing when none was supplied (an absent specimen is not written as an empty one)."""
    if not metadata.get("sample"):
        return {}
    sample = copy.deepcopy(metadata["sample"])
    validate_sample(sample)
    return {"sample": sample}


def _recorded_overrides(metadata: dict) -> dict:
    """Explicit calibration supplied with a new copy (for example a session's
    ``dataset.yaml`` at conversion), validated in calculation units."""
    overrides = copy.deepcopy(metadata.get("calibration_overrides") or {})
    _validate_overrides(overrides)
    return overrides


def validate_header(header: dict) -> None:
    """Validate common metadata before interpreting an encoded acquisition."""
    if not isinstance(header, dict) or not isinstance(
        header.get("scientific_metadata"), dict
    ):
        raise ValueError(
            "Missing QEM scientific metadata; re-export the original acquisition."
        )
    scientific = header["scientific_metadata"]
    axes = scientific.get("axes", [])
    if not isinstance(axes, list) or not all(isinstance(axis, dict) for axis in axes):
        raise ValueError("Invalid QEM axes; re-export the original acquisition.")
    if (
        header.get("container") != CONTAINER
        or header.get("container_version") != CONTAINER_VERSION
        or header.get("codec") != header.get("profile")
        or scientific.get("schema") not in (LEGACY_SCHEMA, SCHEMA)
        or [axis.get("size") for axis in axes] != header.get("shape")
        or [axis.get("name") for axis in axes] != list(AXIS_NAMES)
    ):
        raise ValueError(
            "Unsupported or inconsistent QEM metadata; update the reader or re-export the original."
        )
    if scientific["schema"] == SCHEMA:
        canonical = microscopy_metadata(scientific)
        if canonical != scientific:
            raise ValueError("Schema-2 QEM quantities require canonical microscopy units.")
    _validate_scientific(scientific)
    _legacy_overrides(scientific)


def _validate_source_documents(documents: list[dict]) -> None:
    """Authenticate bounded original attachments without interpreting unknown fields."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"QEM metadata JSON contains non-standard constant {value}.")

    if not isinstance(documents, list) or len(documents) > 16:
        raise ValueError("QEM permits at most 16 XML/JSON metadata attachments.")
    total = 0
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("QEM metadata attachments must be named documents.")
        name, content = document.get("filename"), document.get("content")
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                or not isinstance(content, str)):
            raise ValueError("QEM metadata attachments require a filename and UTF-8 content.")
        payload = content.encode("utf-8")
        total += len(payload)
        if total > 4 << 20 or hashlib.sha256(payload).hexdigest() != document.get("sha256"):
            raise ValueError("QEM metadata attachment is oversized or its checksum does not match.")
        if document.get("mediaType") == "application/xml":
            if "<!DOCTYPE" in content.upper() or "<!ENTITY" in content.upper():
                raise ValueError("QEM metadata XML cannot contain document types or entities.")
            try:
                ET.fromstring(payload)
            except ET.ParseError as error:
                raise ValueError("QEM metadata attachment contains malformed XML.") from error
        elif document.get("mediaType") == "application/json":
            if not isinstance(json.loads(content, parse_constant=reject_constant), dict):
                raise ValueError("QEM metadata JSON must contain named fields.")
        else:
            raise ValueError("QEM metadata attachments support XML and JSON only.")


def _validate_scientific(scientific: dict) -> None:
    """Check provenance and one unambiguous physical calibration at the boundary."""
    normalized = microscopy_metadata(scientific)
    _validate_source_documents(scientific.get("source_documents", []))
    quantities = normalized.get("electron_microscope", {})
    if not isinstance(scientific.get("source_metadata"), dict):
        raise ValueError("QEM source_metadata must be an object, including when empty.")
    if scientific.get("source_metadata_coverage") not in (
        "reader-retained", "exhaustive", "unknown"
    ):
        raise ValueError("Declare QEM source_metadata_coverage explicitly.")
    for section in ("electron_microscope", "calibration_overrides"):
        retired = sorted(set(scientific.get(section) or {}) & set(RETIRED_QUANTITIES))
        if retired:
            raise ValueError(
                f"QEM {section} uses the x/y names of specification 0.0.1 ({retired[0]}); "
                "quantities are named by row and column. Re-export the original acquisition."
            )
    processing = scientific.get("processing")
    if not isinstance(processing, list) or not processing or not all(
        isinstance(record, dict) and isinstance(record.get("operation"), str)
        and record["operation"] and type(record.get("changes_measurements")) is bool
        for record in processing
    ):
        raise ValueError(
            "QEM processing must list every operation with its name and whether it changes measurements."
        )
    paths = [
        "scan_controller/regular_scan/pixel_size_row",
        "scan_controller/regular_scan/pixel_size_column",
        "imaging_system/reciprocal_pixel_size_row",
        "imaging_system/reciprocal_pixel_size_column",
    ]
    for axis, path in zip(normalized["axes"], paths):
        sampling = axis.get("sampling")
        if sampling is not None:
            _require_provenance(sampling)
        duplicate = quantities.get(path)
        if scientific["schema"] == SCHEMA and duplicate is not None:
            if (sampling is None or sampling["unit"] != duplicate["unit"]
                    or not math.isclose(sampling["value"], duplicate["value"], rel_tol=1e-14)):
                raise ValueError(f"Conflicting QEM axis and microscope calibration at {path}.")
    if "sample" in scientific:
        validate_sample(scientific["sample"])
    for path, quantity in quantities.items():
        if (not isinstance(quantity, dict)
                or not isinstance(quantity.get("unit"), str) or not quantity["unit"]
                or isinstance(quantity.get("value"), bool)
                or not isinstance(quantity.get("value"), (int, float))
                or not math.isfinite(quantity["value"]) or quantity["value"] <= 0):
            raise ValueError(f"Invalid QEM microscope quantity at {path}.")
        _require_provenance(quantity)


SAMPLE_GEOMETRIES = ("cross-section", "plan-view")
SAMPLE_ROLES = ("film", "substrate", "support", "particle")
THICKNESS_METHODS = ("diffraction_ridge", "pacbed_fit", "ssb_depth", "ptychography_multislice", "cross_section", "nominal")


def validate_sample(sample: dict) -> None:
    """Check the optional ``sample`` group (specification 0.0.3): the specimen as a person declared it, never derived.

    ``components`` maps free labels (e.g. "BTO") to a chemical formula, a zone axis [u, v, w] in the CIF's cell, an optional
    role and CIF document reference, and a list of thickness estimates (method from a fixed list, value in angstrom, the
    scan region it was measured on). The group carries one provenance and evidence, like a calibration override: who
    declared it and in which file."""
    if not isinstance(sample, dict):
        raise ValueError("QEM sample must be an object.")
    _require_provenance(sample)
    for key in ("id", "name", "description", "orientation_relationship", "evidence"):
        if key in sample and not isinstance(sample[key], str):
            raise ValueError(f"QEM sample {key} must be text.")
    if "geometry" in sample and sample["geometry"] not in SAMPLE_GEOMETRIES:
        raise ValueError(f"QEM sample geometry is one of {SAMPLE_GEOMETRIES}.")
    if "growth_direction" in sample:
        _require_indices(sample["growth_direction"], "growth_direction")
    components = sample.get("components", {})
    if not isinstance(components, dict) or not all(isinstance(label, str) and label for label in components):
        raise ValueError("QEM sample components map nonempty labels to components.")
    for label, component in components.items():
        _validate_component(label, component)
    in_view = sample.get("components_in_view")
    if in_view is not None and (not isinstance(in_view, list) or not all(isinstance(label, str) and label in components for label in in_view)):
        raise ValueError("QEM sample components_in_view lists labels of sample components.")
    for label, component in components.items():
        for estimate in component.get("thickness_estimates", []):
            if isinstance(estimate.get("region"), str) and estimate["region"] not in components:
                raise ValueError(f"QEM sample component {label} thickness estimate: region {estimate['region']!r} is not a component.")


def _validate_component(label: str, component: dict) -> None:
    """One sample component: formula, zone, role, CIF reference and thickness estimates."""
    if not isinstance(component, dict):
        raise ValueError(f"QEM sample component {label} must be an object.")
    if "chemical_formula" in component and not isinstance(component["chemical_formula"], str):
        raise ValueError(f"QEM sample component {label}: chemical_formula must be text.")
    if "role" in component and component["role"] not in SAMPLE_ROLES:
        raise ValueError(f"QEM sample component {label}: role is one of {SAMPLE_ROLES}.")
    if "zone_axis" in component:
        _require_indices(component["zone_axis"], f"{label} zone_axis")
    cif = component.get("cif")
    if cif is not None and (not isinstance(cif, dict) or not isinstance(cif.get("document"), str)
                            or not isinstance(cif.get("sha256"), str)):
        raise ValueError(f"QEM sample component {label}: cif names a source document and its sha256.")
    estimates = component.get("thickness_estimates", [])
    if not isinstance(estimates, list):
        raise ValueError(f"QEM sample component {label}: thickness_estimates is a list.")
    for estimate in estimates:
        _validate_thickness_estimate(label, estimate)
    if sum(bool(estimate.get("preferred")) for estimate in estimates) > 1:
        raise ValueError(f"QEM sample component {label}: at most one preferred thickness estimate.")


def _validate_thickness_estimate(label: str, estimate: dict) -> None:
    """One thickness estimate: a method from the fixed list, a positive value in angstrom and where it was measured."""
    where = f"QEM sample component {label} thickness estimate"
    if not isinstance(estimate, dict) or estimate.get("method") not in THICKNESS_METHODS:
        raise ValueError(f"{where}: method is one of {THICKNESS_METHODS}.")
    if estimate.get("unit") != "angstrom" or not _positive(estimate.get("value")):
        raise ValueError(f"{where}: a positive value in angstrom.")
    if "uncertainty" in estimate and not _positive(estimate["uncertainty"]):
        raise ValueError(f"{where}: uncertainty is positive, in angstrom.")
    bounds = estimate.get("range")
    if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or not all(map(_positive, bounds))
                               or bounds[0] > bounds[1]):
        raise ValueError(f"{where}: range is [low, high] in angstrom.")
    region = estimate.get("region")
    if region is not None and not (isinstance(region, str) or _scan_region(region)):
        raise ValueError(f"{where}: region is a component label, {{rows, cols}} or {{point}} in scan positions.")
    for key in ("reference", "date"):
        if key in estimate and not isinstance(estimate[key], str):
            raise ValueError(f"{where}: {key} must be text.")
    if "preferred" in estimate and type(estimate["preferred"]) is not bool:
        raise ValueError(f"{where}: preferred is true or false.")


def _scan_region(region) -> bool:
    """A rectangle {rows: [start, end], cols: [start, end]} or a point {point: [row, col]}, in scan positions."""
    if not isinstance(region, dict):
        return False
    if set(region) == {"point"}:
        return _integers(region["point"], 2, minimum=0)
    return (set(region) == {"rows", "cols"} and all(_integers(region[k], 2, minimum=0) and region[k][0] < region[k][1]
                                                    for k in ("rows", "cols")))


def _require_indices(value, name: str) -> None:
    if not (_integers(value, 3) or _integers(value, 4)) or not any(value):
        raise ValueError(f"QEM sample {name} is [u, v, w] or hexagonal [u, v, t, w] integers, not all zero.")


def _integers(value, length: int, minimum: int | None = None) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(type(v) is int and (minimum is None or v >= minimum) for v in value))


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _require_provenance(quantity: dict) -> None:
    """Require an explicit origin for a calibrated quantity."""
    if not isinstance(quantity.get("provenance"), str) or not quantity["provenance"].strip():
        raise ValueError("Calibrated QEM quantities require nonempty provenance.")


def recorded_metadata(scientific: dict) -> dict:
    """Derive calculation conveniences exclusively from public recorded quantities.

    Parameters
    ----------
    scientific : dict
        Validated scientific metadata, with explicit quantities and units.

    Returns
    -------
    dict
        Recorded calibration in the existing Python calculation API units.

    Examples
    --------
    >>> record = acquisition_metadata((1, 1, 2, 2), {"scan_sampling_A": [0.4, 0.6]})
    >>> recorded_metadata(record)["scan_sampling_A"]
    [0.4, 0.6]
    """
    normalized = microscopy_metadata(scientific)
    result = {"source_metadata": copy.deepcopy(normalized.get("source_metadata", {}))}
    axes = normalized.get("axes", [])
    for first, field in ((0, "scan_sampling_A"), (2, "detector_sampling")):
        pair = [axis.get("sampling") for axis in axes[first:first + 2]]
        if len(pair) == 2 and all(pair):
            if pair[0]["unit"] != pair[1]["unit"]:
                raise ValueError("QEM row and column sampling require the same unit.")
            result[field] = [quantity["value"] for quantity in pair]
            if first == 2:
                result["detector_sampling_unit"] = pair[0]["unit"]
    for path, field in (
        ("electron_source/accelerating_voltage", "voltage_kV"),
        ("illumination_system/semi_convergence_angle", "semiangle_mrad"),
        ("scan_controller/regular_scan/dwell_time", "dwell_time_us"),
        ("imaging_system/camera_length", "camera_length_mm"),
    ):
        quantity = normalized.get("electron_microscope", {}).get(path)
        if quantity is not None:
            result[field] = quantity["value"]
    return result


def _validate_overrides(overrides: dict) -> None:
    units = {
        "electron_source/accelerating_voltage": {"V"},
        "illumination_system/semi_convergence_angle": {"mrad"},
        "scan_controller/regular_scan/dwell_time": {"s"},
        "imaging_system/camera_length": {"m"},
    }
    pairs = [
        ("scan_controller/regular_scan/pixel_size_", {"m"}),
        ("imaging_system/reciprocal_pixel_size_", {"mrad", "1/nm", "1/Å"}),
    ]
    for prefix, allowed in pairs:
        units.update({prefix + axis: allowed for axis in ("row", "column")})
    if not isinstance(overrides, dict):
        raise ValueError("QEM calibration overrides must be named quantities.")
    for path, quantity in overrides.items():
        if not isinstance(quantity, dict):
            raise ValueError(f"Invalid QEM calibration quantity at {path}.")
        value = quantity.get("value")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                or quantity.get("unit") not in units.get(path, set())
                or quantity.get("provenance") != "user_override"
                or not isinstance(quantity.get("evidence"), str)
                or not quantity["evidence"]):
            raise ValueError(f"Invalid QEM calibration override at {path}; check value, units and evidence.")
        if path.startswith("scan_controller/regular_scan/pixel_size_") and not 1e-14 <= value <= 1e-6:
            raise ValueError("Scan sampling must be between 0.0001 and 10000 angstrom per pixel.")
    for prefix, _ in pairs:
        row, column = overrides.get(prefix + "row"), overrides.get(prefix + "column")
        if ((row is None) != (column is None)
                or row is not None and row["unit"] != column["unit"]):
            raise ValueError("QEM calibration requires both row and column in the same units.")


def effective_metadata(metadata: dict, scientific: dict) -> dict:
    """Restore explicit user calibration without overwriting its recorded source.

    Examples
    --------
    >>> effective_metadata({}, {"calibration_overrides": {}})
    {}
    """
    overrides = _legacy_overrides(scientific)
    result = dict(metadata)
    if scientific.get("schema") == SCHEMA:
        for field in ("scan_sampling_A", "detector_sampling", "detector_sampling_unit",
                      "detector_sampling_inv_A", "voltage_kV", "semiangle_mrad",
                      "dwell_time_us", "camera_length_mm"):
            result.pop(field, None)
        result.update(recorded_metadata(scientific))
    for prefix, field, factor in (
        ("scan_controller/regular_scan/pixel_size_", "scan_sampling_A", 1e10),
        ("imaging_system/reciprocal_pixel_size_", "detector_sampling", 1),
    ):
        if prefix + "row" in overrides:
            result[field] = [overrides[prefix + axis]["value"] * factor for axis in ("row", "column")]
            if field == "detector_sampling":
                result["detector_sampling_unit"] = overrides[prefix + "row"]["unit"]
                result.pop("detector_sampling_inv_A", None)
    voltage = overrides.get("electron_source/accelerating_voltage")
    if voltage is not None:
        result["voltage_kV"] = voltage["value"] / 1000
    for path, field, factor in (
        ("illumination_system/semi_convergence_angle", "semiangle_mrad", 1),
        ("scan_controller/regular_scan/dwell_time", "dwell_time_us", 1e6),
        ("imaging_system/camera_length", "camera_length_mm", 1000),
    ):
        if path in overrides:
            result[field] = overrides[path]["value"] * factor
    return result
