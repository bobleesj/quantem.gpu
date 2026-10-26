"""Read a 4D-STEM dataset that QuantEM saved as a zipped Zarr v3 store.

QuantEM's ``Dataset4dstem`` serializer writes a Zarr v3 group (``zarr.json``)
whose ``_array`` member holds the measurement and whose ``_sampling``,
``_origin`` and ``_units`` members hold the axis calibration. This reader
opens the ZIP in place, decodes each stored chunk straight into one output
array on several threads, and maps the calibration to the same metadata keys
as the other formats. It never extracts the archive, changes the dtype, bins,
crops, or computes statistics over the data.

Blosc chunks need ``numcodecs`` (``pip install quantem.gpu[zarr]``); chunks
stored with the ``bytes`` codec alone need nothing more.
"""

import json
import math
import os
import struct
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from quantem.gpu.formats.emd import DETECTOR_UNITS, SCAN_UNITS, axis_calibration

DATASET_CLASS = "Dataset4dstem"
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")


def is_zarr_zip(path) -> bool:
    """Return whether ``path`` is a ZIP holding a QuantEM ``Dataset4dstem`` Zarr v3 group.

    Only the ZIP directory and the root ``zarr.json`` are read.
    """
    if not isinstance(path, (str, os.PathLike)) or not Path(path).is_file():
        return False
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            root = json.loads(archive.read("zarr.json"))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False
    serialized = root.get("attributes", {}).get("_autoserialize", {})
    return root.get("zarr_format") == 3 and serialized.get("class_name") == DATASET_CLASS


def load_array_source(path, scan_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Decode a saved QuantEM 4D-STEM Zarr archive into one host array and its metadata.

    Parameters
    ----------
    path
        The ``.zip`` archive. It is read in place.
    scan_shape
        Optional check against the stored ``(row, col)`` scan shape.

    Returns
    -------
    tuple of numpy.ndarray and dict
        The stored array, bit for bit in its stored dtype, and the metadata
        that ``io.load`` attaches to the loaded acquisition.

    Examples
    --------
    >>> data, metadata = load_array_source("dataset.zip")
    >>> metadata["scan_sampling_A"]
    """
    path = Path(path)
    with _Archive(path) as archive:
        root = archive.json("zarr.json")
        if root.get("zarr_format") != 3 or root.get("attributes", {}).get(
            "_autoserialize", {}
        ).get("class_name") != DATASET_CLASS:
            raise ValueError(f"{path} is not a saved QuantEM 4D-STEM Zarr archive.")
        data = archive.array("_array")
        if data.ndim != 4 or min(data.shape) <= 0:
            raise ValueError(
                "The saved array must have four axes: scan row, scan column, "
                f"detector row, detector column; got shape {data.shape}."
            )
        if scan_shape is not None and tuple(scan_shape) != data.shape[:2]:
            raise ValueError(f"scan_shape {tuple(scan_shape)} disagrees with the saved array {data.shape[:2]}.")
        sampling = _calibration(archive, "_sampling", data.ndim)
        origin = _calibration(archive, "_origin", data.ndim)
        units = _list_group(archive, "_units", data.ndim)
        attributes = root.get("attributes", {})
        recorded = {
            key: value
            for key, value in archive.json("_metadata/zarr.json", {}).get("attributes", {}).items()
            if key != "_container_type"
        }
    metadata = {
        "backend": "cpu",
        "representation": "dense",
        "source_kind": "zarr",
        "source_format": "QuantEM Zarr archive",
        "source_path": str(path),
        "source_metadata": {
            "class": DATASET_CLASS,
            "name": attributes.get("_name"),
            "signal_units": attributes.get("_signal_units"),
            "sampling": sampling,
            "origin": origin,
            "units": units,
            "metadata": recorded,
        },
        "source_metadata_coverage": "reader-retained",
    }
    if sampling and units:
        metadata.update(axis_calibration(list(zip(sampling, units))))
        # Axis values in the units the dataset reports (angstrom, mrad, 1/angstrom).
        factors = [_unit(axis, unit) for axis, unit in enumerate(units)]
        if all(factors):
            metadata["sampling"] = [value * factor for value, (factor, _) in zip(sampling, factors)]
            metadata["units"] = [unit for _, unit in factors]
            if origin:
                metadata["origin"] = [value * factor for value, (factor, _) in zip(origin, factors)]
    return data, metadata


def _unit(axis: int, unit: str) -> tuple[float, str] | None:
    if axis < 2:
        return (SCAN_UNITS[unit], "angstrom") if unit in SCAN_UNITS else None
    return DETECTOR_UNITS.get(unit)


def _calibration(archive: "_Archive", name: str, ndim: int) -> list[float] | None:
    if not archive.has(f"{name}/zarr.json"):
        return None
    values = archive.array(name).reshape(-1)
    if values.size != ndim:
        return None
    return [float(value) for value in values]


def _list_group(archive: "_Archive", name: str, ndim: int) -> list[str] | None:
    attributes = archive.json(f"{name}/zarr.json", {}).get("attributes", {})
    if attributes.get("_container_type") != "list":
        return None
    values = [attributes.get(str(index)) for index in range(ndim)]
    return values if all(isinstance(value, str) for value in values) else None


class _Archive:
    """Positional reads of ZIP members; stored members are read without the ZIP lock."""

    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path)
        self.members = {info.filename: info for info in self.zip.infolist()}
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self.lock = threading.Lock()
        self.offsets: dict[str, int] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        self.zip.close()

    def has(self, name: str) -> bool:
        return name in self.members

    def json(self, name: str, default=None):
        if name not in self.members:
            if default is None:
                raise ValueError(f"{self.path} has no {name}.")
            return default
        return json.loads(self.read(name))

    def read(self, name: str) -> bytes:
        info = self.members[name]
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            with self.lock:
                return self.zip.read(info)
        offset = self.offsets.get(name)
        if offset is None:
            header = os.pread(self.fd, _LOCAL_HEADER.size, info.header_offset)
            fields = _LOCAL_HEADER.unpack(header)
            if fields[0] != b"PK\x03\x04":
                raise ValueError(f"{self.path}: damaged ZIP entry {name}.")
            offset = info.header_offset + _LOCAL_HEADER.size + fields[9] + fields[10]
            self.offsets[name] = offset
        payload = os.pread(self.fd, info.file_size, offset)
        if len(payload) != info.file_size:
            raise ValueError(f"{self.path}: truncated ZIP entry {name}.")
        return payload

    def array(self, name: str) -> np.ndarray:
        """Decode every chunk of the Zarr v3 array ``name`` into one array."""
        meta = self.json(f"{name}/zarr.json")
        if meta.get("node_type") != "array" or meta.get("zarr_format") != 3:
            raise ValueError(f"{self.path}: {name} is not a Zarr v3 array.")
        if meta.get("storage_transformers"):
            raise NotImplementedError(f"{name}: Zarr storage transformers are not supported.")
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["data_type"])
        grid = meta["chunk_grid"]
        if grid.get("name") != "regular":
            raise NotImplementedError(f"{name}: only a regular Zarr chunk grid is supported.")
        chunks = tuple(grid["configuration"]["chunk_shape"])
        encoding = meta.get("chunk_key_encoding", {"name": "default"})
        separator = encoding.get("configuration", {}).get("separator", "/" if encoding["name"] == "default" else ".")
        prefix = f"{name}/c{separator}" if encoding["name"] == "default" else f"{name}/"
        stored_dtype, blosc = _codecs(name, meta.get("codecs", []), dtype)
        if blosc:
            try:
                from numcodecs import blosc as blosc_codec
            except ImportError as error:
                raise ImportError(
                    "Reading Blosc-compressed Zarr chunks needs numcodecs; "
                    "install quantem.gpu[zarr]."
                ) from error
        output = np.empty(shape, dtype)
        chunk_bytes = math.prod(chunks) * dtype.itemsize
        fill = meta.get("fill_value", 0)
        grid_counts = [-(-size // chunk) for size, chunk in zip(shape, chunks)]

        def decode(index):
            key = prefix + separator.join(map(str, index)) if index else prefix.rstrip(separator)
            region = tuple(
                slice(i * chunk, min((i + 1) * chunk, size))
                for i, chunk, size in zip(index, chunks, shape)
            )
            if key not in self.members:
                output[region] = fill
                return
            payload = self.read(key)
            if blosc:
                raw = np.empty(chunk_bytes, np.uint8)
                blosc_codec.decompress(payload, raw)
            else:
                raw = np.frombuffer(payload, np.uint8)
            if raw.size != chunk_bytes:
                raise ValueError(f"{self.path}: chunk {key} holds {raw.size} bytes, expected {chunk_bytes}.")
            block = raw.view(stored_dtype).reshape(chunks)
            output[region] = block[tuple(slice(0, item.stop - item.start) for item in region)]

        indices = list(product(*(range(count) for count in grid_counts)))
        if len(indices) <= 1:
            for index in indices:
                decode(index)
        else:
            cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
            workers = min(len(indices), cores or 1, 32)
            with ThreadPoolExecutor(workers) as pool:
                for _ in pool.map(decode, indices):
                    pass
        return output


def _codecs(name: str, codecs: list[dict], dtype: np.dtype) -> tuple[np.dtype, bool]:
    """Accept ``bytes`` optionally followed by ``blosc``; return the stored dtype and whether Blosc is used."""
    names = [codec.get("name") for codec in codecs]
    if names not in (["bytes"], ["bytes", "blosc"]):
        raise NotImplementedError(
            f"{name}: Zarr codecs {names} are not supported; this reader decodes "
            "'bytes' optionally followed by 'blosc'."
        )
    endian = codecs[0].get("configuration", {}).get("endian", "little")
    stored = dtype.newbyteorder("<" if endian == "little" else ">") if dtype.itemsize > 1 else dtype
    return stored, len(names) == 2
