"""Header-only acquisition readiness and source stability signatures."""

import os
import re
from typing import Any

import h5py
import numpy as np

from .models import MasterReadiness


def _absolute_source_path(path: str | os.PathLike[str]) -> str:
    """Absolute source spelling without resolving a watched symlink."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _file_source_signature(path: str) -> dict[str, Any]:
    """JSON-serializable identity and stability fields for one source file."""
    absolute = _absolute_source_path(path)
    try:
        stat = os.stat(absolute)
    except FileNotFoundError:
        return {"path": absolute, "missing": True}
    except OSError as exc:
        return {"path": absolute, "unreadable": True, "error": str(exc)}
    signature: dict[str, Any] = {
        "path": absolute,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    if os.path.islink(absolute):
        try:
            link_stat = os.lstat(absolute)
            signature["symlink_target"] = os.readlink(absolute)
            signature["symlink_mtime_ns"] = int(link_stat.st_mtime_ns)
            signature["symlink_ctime_ns"] = int(link_stat.st_ctime_ns)
        except OSError:
            signature["symlink_unreadable"] = True
    return signature


def _master_source_signature(
    master_path: str,
    source_paths: set[str],
    datasets: list[dict[str, Any]],
    *,
    expected_frames: int | None,
    expected_basis: str | None,
) -> dict[str, Any]:
    """Build a deterministic master/chunk fingerprint for poll comparison."""
    return {
        "master": master_path,
        "files": [
            _file_source_signature(path)
            for path in sorted({_absolute_source_path(path) for path in source_paths})
        ],
        "datasets": [dict(dataset) for dataset in datasets],
        "expectation": {
            "frames": expected_frames,
            "basis": expected_basis,
        },
    }


def _normalise_readiness_scan_shape(
    scan_shape: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Validate an explicit readiness frame-count contract."""
    if scan_shape is None:
        return None
    try:
        values = tuple(scan_shape)
    except TypeError as exc:
        raise ValueError(
            "scan_shape must be two positive integers (scan_row, scan_col), "
            "for example scan_shape=(512, 512)."
        ) from exc
    if len(values) != 2:
        raise ValueError(
            "scan_shape must contain exactly two positive integers "
            f"(scan_row, scan_col); got {scan_shape!r}."
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(  # noqa: TRY004 - preserve the public validation contract
                "scan_shape values must be positive integers, not booleans; "
                f"got {scan_shape!r}."
            )
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "scan_shape values must be positive integers; "
                f"got {scan_shape!r}."
            ) from exc
        if integer < 1 or integer != value:
            raise ValueError(
                "scan_shape values must be positive integers; "
                f"got {scan_shape!r}."
            )
        normalized.append(integer)
    return normalized[0], normalized[1]


def _scalar_int(handle: h5py.File, path: str) -> int | None:
    """Read one optional scalar integer without following detector data."""
    dataset = handle.get(path)
    if dataset is None:
        return None
    try:
        value = np.asarray(dataset[()])
        if value.size != 1:
            return None
        return int(value.reshape(()).item())
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _attribute_scan_shape(attributes: Any) -> tuple[int, int] | None:
    """Return a valid positive ``scan_shape`` HDF5 attribute, when present."""
    if "scan_shape" not in attributes:
        return None
    try:
        values = tuple(int(value) for value in attributes["scan_shape"])
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 2 or any(value < 1 for value in values):
        return None
    return values[0], values[1]


def inspect_master_readiness(
    filepath: str | os.PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
) -> MasterReadiness:
    """Inspect whether a 4D-STEM master is complete enough to browse.

    The inspection reads HDF5 headers only. It supports a self-contained
    ``entry/data/data`` dataset and the usual ``data_NNNNNN`` entries, including
    external links to sibling files. For the selected source it totals stored
    frames, checks detector shape and dtype consistency, and compares the total
    with ``scan_shape`` or the master's ``ntrigger``/``nimages`` metadata.

    The returned ``source_signature`` captures the master, every selected source
    file, and every selected dataset header. Compare it across separate polls to
    establish a caller-defined stability probation; this function deliberately
    does not sleep or perform a second poll itself.

    Parameters
    ----------
    filepath
        Master HDF5 path.
    scan_shape
        Optional explicit ``(scan_row, scan_col)`` contract. When supplied, it
        takes precedence over the master's expected-frame metadata.

    Returns
    -------
    MasterReadiness
        Structured readiness state, corrective action, and source signature.

    Raises
    ------
    ValueError
        If ``scan_shape`` is not exactly two positive integers.
    """
    explicit_scan_shape = _normalise_readiness_scan_shape(scan_shape)
    master_path = _absolute_source_path(filepath)
    source_paths = {master_path}
    datasets: list[dict[str, Any]] = []
    initial_files = {master_path: _file_source_signature(master_path)}
    expected_frames = (
        int(explicit_scan_shape[0] * explicit_scan_shape[1])
        if explicit_scan_shape is not None
        else None
    )
    expected_basis = (
        f"explicit scan_shape={explicit_scan_shape}"
        if explicit_scan_shape is not None
        else None
    )
    source_kind = "unavailable"
    actual_frames: int | None = None
    detector_shape: tuple[int, int] | None = None
    common_dtype: str | None = None

    def result(ready: bool, reason: str, action: str) -> MasterReadiness:
        return MasterReadiness(
            ready=bool(ready),
            reason=str(reason),
            action=str(action),
            source_kind=source_kind,
            actual_frames=actual_frames,
            expected_frames=expected_frames,
            detector_shape=detector_shape,
            dtype=common_dtype,
            source_signature=_master_source_signature(
                master_path,
                source_paths,
                datasets,
                expected_frames=expected_frames,
                expected_basis=expected_basis,
            ),
        )

    master_stat = initial_files[master_path]
    if master_stat.get("missing", False):
        return result(
            False,
            f"master file is missing: {master_path}",
            "Wait for the master file to be atomically renamed into place, "
            "then poll again.",
        )
    if master_stat.get("unreadable", False):
        return result(
            False,
            f"master file cannot be inspected: {master_path} "
            f"({master_stat.get('error', 'unknown filesystem error')})",
            "Fix file permissions or storage availability, then poll again.",
        )
    if int(master_stat.get("size", 0)) <= 0:
        return result(
            False,
            f"master file is empty: {master_path}",
            "Wait for the master HDF5 header to finish writing, then poll again.",
        )

    try:
        with h5py.File(master_path, "r") as master:
            data_group = master.get("entry/data")
            if data_group is None:
                return result(
                    False,
                    "master is missing the entry/data group",
                    "Wait for acquisition to finish, or recopy a complete master file.",
                )

            chunk_names = sorted(
                name
                for name in data_group
                if re.fullmatch(r"data_\d{6}", name)
            )
            source_names = chunk_names or (["data"] if "data" in data_group else [])
            if not source_names:
                return result(
                    False,
                    "master has no entry/data/data dataset or data_NNNNNN sources",
                    "Wait for detector data links to finish writing, or recopy "
                    "the complete acquisition group.",
                )

            links = [data_group.get(name, getlink=True) for name in source_names]
            source_kind = (
                "external"
                if any(isinstance(link, h5py.ExternalLink) for link in links)
                else "inline"
            )
            # Discover every dependency before an incomplete chunk can return
            # early, so arrival watchers see the full expected file count.
            for link in links:
                if isinstance(link, h5py.ExternalLink):
                    link_filename = os.fsdecode(os.fspath(link.filename))
                    source_paths.add(_absolute_source_path(
                        os.path.join(os.path.dirname(master_path), link_filename)
                    ))
            ntrigger = _scalar_int(
                master,
                "entry/instrument/detector/detectorSpecific/ntrigger",
            )
            nimages = _scalar_int(
                master,
                "entry/instrument/detector/detectorSpecific/nimages",
            )
            if explicit_scan_shape is None and ntrigger is not None:
                if ntrigger < 1:
                    return result(
                        False,
                        f"master ntrigger metadata is not positive: {ntrigger}",
                        "Wait for acquisition metadata to finish writing, or repair "
                        "ntrigger before loading.",
                    )
                images_per_trigger = int(1 if nimages is None else nimages)
                if images_per_trigger < 1:
                    return result(
                        False,
                        "master nimages metadata is not positive: "
                        f"{images_per_trigger}",
                        "Wait for acquisition metadata to finish writing, or repair "
                        "nimages before loading.",
                    )
                expected_frames = int(ntrigger * images_per_trigger)
                expected_basis = (
                    f"master metadata ntrigger={ntrigger}, nimages={images_per_trigger}"
                )

            observed_frames = 0
            detector_shapes: set[tuple[int, int]] = set()
            dtypes: set[str] = set()
            metadata_scan_shapes: set[tuple[int, int]] = set()
            for attributes in (master.attrs, data_group.attrs):
                metadata_shape = _attribute_scan_shape(attributes)
                if metadata_shape is not None:
                    metadata_scan_shapes.add(metadata_shape)

            for name, link in zip(source_names, links, strict=True):
                if isinstance(link, h5py.ExternalLink):
                    link_filename = os.fsdecode(os.fspath(link.filename))
                    source_path = (
                        _absolute_source_path(link_filename)
                        if os.path.isabs(link_filename)
                        else _absolute_source_path(
                            os.path.join(os.path.dirname(master_path), link_filename)
                        )
                    )
                    dataset_path = str(link.path)
                    source_paths.add(source_path)
                    initial_files[source_path] = _file_source_signature(source_path)
                    record: dict[str, Any] = {
                        "name": str(name),
                        "kind": "external",
                        "file": source_path,
                        "dataset": dataset_path,
                    }
                    datasets.append(record)
                    source_stat = initial_files[source_path]
                    if source_stat.get("missing", False):
                        return result(
                            False,
                            f"linked detector file is missing: {source_path}",
                            "Finish copying or writing the linked detector file "
                            "next to the master, then poll again.",
                        )
                    if source_stat.get("unreadable", False):
                        return result(
                            False,
                            f"linked detector file cannot be inspected: "
                            f"{source_path} "
                            f"({source_stat.get('error', 'unknown filesystem error')})",
                            "Fix file permissions or storage availability, then "
                            "poll again.",
                        )
                    if int(source_stat.get("size", 0)) <= 0:
                        return result(
                            False,
                            f"linked detector file is empty: {source_path}",
                            "Wait for the linked detector HDF5 file to finish "
                            "writing, then poll again.",
                        )
                    try:
                        source_handle = h5py.File(source_path, "r")
                    except OSError as exc:
                        return result(
                            False,
                            "linked detector file is not readable HDF5: "
                            f"{source_path} ({exc})",
                            "Wait for the detector writer to close or flush the "
                            "file, then poll again; recopy it if the error persists.",
                        )
                    with source_handle:
                        dataset = source_handle.get(dataset_path)
                        if dataset is None:
                            return result(
                                False,
                                "linked detector dataset is missing: "
                                f"{source_path}:{dataset_path}",
                                "Repair the external link or recopy the acquisition "
                                "so it targets entry/data/data.",
                            )
                        shape = tuple(int(value) for value in dataset.shape)
                        dtype_str = np.dtype(dataset.dtype).str
                        metadata_shape = _attribute_scan_shape(dataset.attrs)
                else:
                    source_path = master_path
                    dataset_path = f"{data_group.name}/{name}"
                    record = {
                        "name": str(name),
                        "kind": "inline",
                        "file": source_path,
                        "dataset": dataset_path,
                    }
                    datasets.append(record)
                    try:
                        dataset = data_group[name]
                        shape = tuple(int(value) for value in dataset.shape)
                        dtype_str = np.dtype(dataset.dtype).str
                        metadata_shape = _attribute_scan_shape(dataset.attrs)
                    except (KeyError, OSError, TypeError, ValueError) as exc:
                        return result(
                            False,
                            "inline detector dataset is not readable: "
                            f"{dataset_path} ({exc})",
                            "Wait for the master dataset header to finish writing, "
                            "then poll again; recopy it if the error persists.",
                        )

                if len(shape) < 3:
                    record.update({"shape": list(shape), "dtype": dtype_str})
                    return result(
                        False,
                        f"detector dataset {dataset_path} has shape {shape}; "
                        "expected at least (frame, det_row, det_col)",
                        "Repair or reacquire the dataset with explicit frame and "
                        "two detector dimensions.",
                    )
                frames = int(np.prod(shape[:-2], dtype=np.int64))
                current_detector_shape = (int(shape[-2]), int(shape[-1]))
                record.update(
                    {
                        "shape": list(shape),
                        "dtype": dtype_str,
                        "frames": frames,
                        "detector_shape": list(current_detector_shape),
                    }
                )
                if metadata_shape is not None:
                    metadata_scan_shapes.add(metadata_shape)
                    record["scan_shape"] = list(metadata_shape)
                observed_frames += frames
                detector_shapes.add(current_detector_shape)
                dtypes.add(dtype_str)

            actual_frames = int(observed_frames)
            if len(detector_shapes) != 1:
                observed = ", ".join(str(shape) for shape in sorted(detector_shapes))
                return result(
                    False,
                    f"detector sources have inconsistent detector shapes: {observed}",
                    "Use a narrower master pattern or repair the acquisition so "
                    "every detector chunk has the same (row, col) shape.",
                )
            detector_shape = next(iter(detector_shapes))
            if len(dtypes) != 1:
                observed = ", ".join(sorted(dtypes))
                return result(
                    False,
                    f"detector sources have inconsistent dtypes: {observed}",
                    "Repair or reacquire the acquisition so every detector chunk "
                    "uses the same stored dtype.",
                )
            common_dtype = next(iter(dtypes))
            if len(metadata_scan_shapes) > 1:
                observed = ", ".join(
                    str(shape) for shape in sorted(metadata_scan_shapes)
                )
                return result(
                    False,
                    "detector sources have inconsistent scan_shape metadata: "
                    f"{observed}",
                    "Repair the conflicting scan_shape attributes, or pass the "
                    "correct explicit scan_shape after confirming the acquisition "
                    "layout.",
                )
            if (
                expected_frames is None
                and len(metadata_scan_shapes) == 1
            ):
                metadata_scan_shape = next(iter(metadata_scan_shapes))
                expected_frames = int(metadata_scan_shape[0] * metadata_scan_shape[1])
                expected_basis = f"HDF5 scan_shape={metadata_scan_shape}"
            if actual_frames < 1:
                return result(
                    False,
                    "detector sources contain zero stored frames",
                    "Wait for detector frames to be written, then poll again.",
                )
            if expected_frames is not None and actual_frames != expected_frames:
                if actual_frames < expected_frames:
                    action = (
                        "Wait for the remaining detector frames or chunks to finish writing, "
                        "then poll again."
                    )
                else:
                    action = (
                        "Pass the correct explicit scan_shape or repair the master frame-count "
                        "metadata before loading."
                    )
                return result(
                    False,
                    f"stored frame count is {actual_frames}; expected "
                    f"{expected_frames} from {expected_basis}",
                    action,
                )
    except OSError as exc:
        return result(
            False,
            f"master file is not readable HDF5: {master_path} ({exc})",
            "Wait for the master writer to close or flush the file, then poll "
            "again; recopy it if the error persists.",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return result(
            False,
            f"master HDF5 headers are incomplete or invalid: {master_path} ({exc})",
            "Wait for acquisition to finish, then poll again; recopy or repair "
            "the group if the error persists.",
        )

    final_signature = _master_source_signature(
        master_path,
        source_paths,
        datasets,
        expected_frames=expected_frames,
        expected_basis=expected_basis,
    )
    final_files = {item["path"]: item for item in final_signature["files"]}
    changed = [
        path
        for path, before in initial_files.items()
        if final_files.get(path, {"path": path, "missing": True}) != before
    ]
    if changed:
        names = ", ".join(os.path.basename(path) for path in changed)
        return MasterReadiness(
            ready=False,
            reason=f"source files changed during readiness inspection: {names}",
            action=(
                "Wait for acquisition or copy writes to finish, then compare a "
                "fresh readiness signature on the next poll."
            ),
            source_kind=source_kind,
            actual_frames=actual_frames,
            expected_frames=expected_frames,
            detector_shape=detector_shape,
            dtype=common_dtype,
            source_signature=final_signature,
        )
    return MasterReadiness(
        ready=True,
        reason=(
            "master and detector sources are complete, readable, and internally "
            "consistent"
        ),
        action="Ready to open with Show4DSTEM.",
        source_kind=source_kind,
        actual_frames=actual_frames,
        expected_frames=expected_frames,
        detector_shape=detector_shape,
        dtype=common_dtype,
        source_signature=final_signature,
    )
