"""A 4D-STEM dataset QuantEM saved as a zipped Zarr v3 store loads exactly, in place."""

import json
import zipfile

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats import zarr_zip

SHAPE = (5, 3, 4, 6)
CHUNKS = (2, 2, 3, 4)


def _array_json(shape, chunks, dtype, codecs, fill_value=0.0):
    return {
        "shape": list(shape),
        "data_type": dtype,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _write(path, data, *, blosc=False, skip=(), compression=zipfile.ZIP_STORED,
           units=("A", "A", "mrad", "mrad"), codecs=None):
    """Write the layout of QuantEM's Dataset4dstem serializer, chunk by chunk."""
    codecs = codecs or [{"name": "bytes", "configuration": {"endian": "little"}}]
    if blosc:
        codecs = codecs + [{"name": "blosc", "configuration": {
            "typesize": data.dtype.itemsize, "cname": "zstd", "clevel": 4,
            "shuffle": "bitshuffle", "blocksize": 0}}]

    def encode(block):
        raw = np.ascontiguousarray(block).astype(block.dtype.newbyteorder("<")).tobytes()
        if blosc:
            from numcodecs import Blosc
            return Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE).encode(raw)
        return raw

    def group(attributes):
        return json.dumps({"attributes": attributes, "zarr_format": 3, "node_type": "group"})

    with zipfile.ZipFile(path, "w", compression) as archive:
        archive.writestr("zarr.json", group({
            "_autoserialize": {"version": 1, "class_module": "quantem.core.datastructures.dataset4dstem",
                               "class_name": "Dataset4dstem"},
            "_name": "synthetic", "_signal_units": "arb. units"}))
        archive.writestr("_array/zarr.json", json.dumps(_array_json(data.shape, CHUNKS, data.dtype.name, codecs)))
        grid = [-(-size // chunk) for size, chunk in zip(data.shape, CHUNKS)]
        for index in np.ndindex(*grid):
            if index in skip:
                continue
            block = np.zeros(CHUNKS, data.dtype)  # Zarr stores edge chunks at full size
            region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, CHUNKS, data.shape))
            block[tuple(slice(0, r.stop - r.start) for r in region)] = data[region]
            archive.writestr("_array/c/" + "/".join(map(str, index)), encode(block))
        for name, values in (("_sampling", [0.5, 0.5, 2.0, 2.0]), ("_origin", [1.0, 2.0, 0.0, 0.0])):
            archive.writestr(f"{name}/zarr.json", json.dumps(_array_json((4,), (4,), "float64", codecs)))
            archive.writestr(f"{name}/c/0", encode(np.asarray(values, np.float64)))
        archive.writestr("_units/zarr.json", group({"_container_type": "list", **dict(enumerate(units))}))
        archive.writestr("_metadata/zarr.json", group({"_container_type": "dict", "ellipticity": None}))


@pytest.fixture
def data():
    return np.random.default_rng(7).standard_normal(SHAPE).astype(np.float32)


def test_reader_decodes_every_chunk_bit_for_bit_with_calibration(tmp_path, data):
    path = tmp_path / "dataset.zip"
    _write(path, data)
    assert zarr_zip.is_zarr_zip(path)
    loaded, metadata = zarr_zip.load_array_source(path)
    assert loaded.dtype == np.float32 and loaded.shape == SHAPE
    assert loaded.tobytes() == data.tobytes()
    assert metadata["scan_sampling_A"] == [0.5, 0.5]
    assert metadata["detector_sampling_mrad"] == [2.0, 2.0]
    assert metadata["units"] == ["angstrom", "angstrom", "mrad", "mrad"]
    assert metadata["origin"] == [1.0, 2.0, 0.0, 0.0]
    assert metadata["source_metadata"]["units"] == ["A", "A", "mrad", "mrad"]


def test_blosc_chunks_decode_exactly(tmp_path, data):
    pytest.importorskip("numcodecs")
    path = tmp_path / "dataset.zip"
    _write(path, data, blosc=True)
    loaded, _ = zarr_zip.load_array_source(path)
    assert loaded.tobytes() == data.tobytes()


def test_missing_chunk_reads_fill_value_and_deflated_members_read(tmp_path, data):
    path = tmp_path / "dataset.zip"
    _write(path, data, skip={(0, 0, 0, 0)}, compression=zipfile.ZIP_DEFLATED)
    loaded, _ = zarr_zip.load_array_source(path)
    expected = data.copy()
    expected[:2, :2, :3, :4] = 0
    assert loaded.tobytes() == expected.tobytes()


def test_big_endian_bytes_keep_values(tmp_path, data):
    path = tmp_path / "dataset.zip"
    _write(path, data, codecs=[{"name": "bytes", "configuration": {"endian": "big"}}])
    # The writer above always stores little-endian bytes; rewrite them big-endian.
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            if "/c/" in name:
                dtype = "<f4" if name.startswith("_array") else "<f8"
                payload = np.frombuffer(payload, dtype).astype(dtype.replace("<", ">")).tobytes()
            archive.writestr(name, payload)
    loaded, metadata = zarr_zip.load_array_source(path)
    assert np.array_equal(loaded, data) and loaded.dtype == np.float32
    assert metadata["scan_sampling_A"] == [0.5, 0.5]


def test_io_load_returns_the_dataset_with_axis_calibration(tmp_path, data):
    path = tmp_path / "dataset.zip"
    _write(path, data)
    loaded = io.load(path, device="cpu", verbose=False)
    assert np.asarray(loaded.data).tobytes() == data.tobytes()
    assert loaded.sampling == (0.5, 0.5, 2.0, 2.0)
    assert loaded.units == ("angstrom", "angstrom", "mrad", "mrad")
    assert loaded.origin == (1.0, 2.0, 0.0, 0.0)
    assert loaded.metadata["source_kind"] == "zarr"


def test_other_zips_are_not_datasets_and_unknown_codecs_are_refused(tmp_path, data):
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("readme.txt", "not a dataset")
    assert not zarr_zip.is_zarr_zip(plain)
    assert not zarr_zip.is_zarr_zip(tmp_path / "missing.zip")
    path = tmp_path / "dataset.zip"
    _write(path, data, codecs=[{"name": "bytes", "configuration": {"endian": "little"}},
                               {"name": "gzip", "configuration": {"level": 5}}])
    with pytest.raises(NotImplementedError, match="codecs"):
        zarr_zip.load_array_source(path)
    _write(path, data)
    with pytest.raises(ValueError, match="scan_shape"):
        zarr_zip.load_array_source(path, scan_shape=(4, 4))
