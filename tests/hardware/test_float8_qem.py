"""Original FP8 acquisitions retain their bytes and GPU calculation semantics."""

import os

import h5py
import numpy as np
import pytest
import torch

from quantem.gpu import SSB, detector, io
from quantem.gpu.formats.float8 import FLOAT8_DTYPES
from quantem.gpu.formats.qem.reference import load_array
from quantem.gpu.formats.qem.validation import validate_qem
from quantem.gpu.io.convert import convert


@pytest.fixture
def backend():
    name = os.environ.get("QEM_TEST_BACKEND")
    if name not in ("cuda", "mps"):
        pytest.skip("Set QEM_TEST_BACKEND to the physical CUDA or MPS device.")
    return name


def host_array(value):
    return value.get() if hasattr(value, "get") else value.cpu().numpy()


def write_float8_h5(path, values, mask=None):
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("entry/data/data", data=values.view(np.uint8))
        data.attrs["quantem_dtype"] = values.dtype.name
        if mask is not None:
            handle.create_dataset(
                "entry/instrument/detector/detectorSpecific/pixel_mask", data=mask
            )


def test_e8m0_extreme_products_preserve_values(backend):
    # Tiny independent scalar oracle covers the subnormal and overflow edges.
    codes = np.asarray([0, 1, 0, 254, 0, 2, 254, 254], np.uint8).reshape(2, 1, 2, 2)
    values = codes.view(FLOAT8_DTYPES["float8_e8m0fnu"])
    expected = values.astype(np.float64).mean(axis=(0, 1)).astype(np.float32)
    with io.load(values, backend=backend, verbose=False) as data:
        np.testing.assert_array_equal(detector.prepare(data).mean_dp(), expected)

    codes = np.asarray(
        [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 2], [0, 127, 126, 125],
         [252] * 4, [254] * 4, [0, 0, 0, 255]], np.uint8,
    )
    values = codes.reshape(7, 1, 2, 2).view(FLOAT8_DTYPES["float8_e8m0fnu"])
    with io.load(values, backend=backend, verbose=False) as data:
        session = detector.prepare(data)
        for selected in (0, 1, 4):
            mask = (np.arange(4) < selected).reshape(2, 2)
            with np.errstate(over="ignore"):
                expected = values.astype(np.float64)[..., mask].sum(axis=-1).astype(np.float32)
            actual = session.masked_sum(mask)
            finite = ~np.isnan(expected)
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_array_equal(
                actual[finite].view(np.uint32), expected[finite].view(np.uint32)
            )


@pytest.mark.parametrize("kind", FLOAT8_DTYPES)
def test_float8_linked_collection_conversion(tmp_path, backend, kind):
    bits = (
        np.broadcast_to(np.arange(256, dtype=np.uint8), (4096, 256))
        .copy()
        .reshape(4096, 16, 16)
    )
    original = tmp_path / "scan_master.h5"
    chunk = tmp_path / "scan_data_000001.h5"
    with h5py.File(chunk, "w") as handle:
        data = handle.create_dataset("entry/data/data", data=bits)
        data.attrs["quantem_dtype"] = kind
    with h5py.File(original, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(
            chunk.name, "/entry/data/data"
        )
    result = convert(original, tmp_path / "copy.qem", backend=backend, write=True)
    assert result.verified, result
    assert result.verification["compared_values"] == bits.size
    assert original.exists() and chunk.exists()
    with io.load(tmp_path / "copy.qem", backend=backend, verbose=False) as data:
        assert data.dtype == FLOAT8_DTYPES[kind]
        np.testing.assert_array_equal(
            data.read().view(torch.uint8).cpu().reshape(bits.shape), bits
        )


@pytest.mark.parametrize("kind", FLOAT8_DTYPES)
@pytest.mark.parametrize(
    "input_kind",
    (
        "numpy",
        "torch",
        "numpy-strided",
        "torch-strided",
        "h5",
        "h5-opaque",
        "h5-3d",
        "h5-master",
    ),
)
def test_float8_original_bits_survive_save_move_and_reopen(
    tmp_path, backend, kind, input_kind
):
    # Every bit pattern crosses multiple codec chunks; no NaN canonicalization.
    bits = (
        np.broadcast_to(np.arange(256, dtype=np.uint8), (33, 17, 256))
        .copy()
        .reshape(33, 17, 16, 16)
    )
    if input_kind.endswith("strided"):
        bits = bits.swapaxes(-1, -2)
    values = bits.view(FLOAT8_DTYPES[kind])
    options = {}
    if input_kind.startswith("h5"):
        original = tmp_path / "source.h5"
        with h5py.File(original, "w") as handle:
            stored = bits.view("V1") if input_kind == "h5-opaque" else bits
            if input_kind in ("h5-3d", "h5-master"):
                stored = stored.reshape(-1, 16, 16)
                options = {"scan_shape": (33, 17)}
                if input_kind == "h5-3d":
                    options["dataset_path"] = "data"
            data = handle.create_dataset(
                "entry/data/data" if input_kind == "h5-master" else "data", data=stored
            )
            data.attrs["quantem_dtype"] = kind
        assert io.inspect(original, **options).dtype == kind
    elif input_kind.startswith("torch"):
        original = torch.from_numpy(bits).to(backend).view(getattr(torch, kind))
    else:
        original = values
    saved = tmp_path / "copy.qem"
    with io.load(original, backend=backend, verbose=False, **options) as loaded:
        assert loaded.dtype == values.dtype
        assert loaded.metadata["file_counts_exact"] is True
        assert loaded.representation.value == "encoded"
        np.testing.assert_array_equal(loaded.read().view(torch.uint8).cpu(), bits)
        calculated = host_array(loaded.data._tensor(0, 1)).reshape(16, 16)
        # The independent scalar oracle is only these 256 possible values.
        expected = values[0, 0].astype(np.float32)
        np.testing.assert_array_equal(calculated, expected)
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(
            calculated.view(np.uint32)[finite], expected.view(np.uint32)[finite]
        )
        if input_kind == "numpy":
            np.testing.assert_allclose(
                detector.prepare(loaded).mean_dp(),
                expected,
                rtol=1e-6,
                atol=0,
            )
            np.testing.assert_allclose(
                detector.prepare(loaded).reduce_frames([0, 510, 560], mode="max"),
                expected,
                rtol=1e-6,
                atol=0,
            )
        loaded.calibration.voltage_kV = 300
        loaded.calibration.scan_sampling_A = (0.4, 0.5)
        io.save(saved, loaded, metadata={"source_note": "Original FP8 bytes"})
        assert loaded.data.peak_decode_bytes <= 32 << 20
    assert validate_qem(saved)["dtype"] == kind
    moved = tmp_path / "moved.qem"
    saved.rename(moved)
    if input_kind.startswith("h5"):
        original.unlink()
    with io.load(moved, backend=backend, verbose=False) as reopened:
        assert reopened.dtype == values.dtype
        assert reopened.sampling[:2] == (0.4, 0.5)
        assert reopened.metadata["source_note"] == "Original FP8 bytes"
        result = reopened.read(
            scan_region=(29, 33, 13, 17),
            detector_region=(1, 15, 2, 14),
            output="native",
        )
        try:
            assert result.dtype == values.dtype
            assert result.get().tobytes() == values[29:33, 13:17, 1:15, 2:14].tobytes()
        finally:
            result.release()
        np.testing.assert_array_equal(reopened.read().view(torch.uint8).cpu(), bits)
    reference, _ = load_array(moved)
    assert reference.dtype == values.dtype
    assert reference.tobytes() == values.tobytes()


@pytest.mark.parametrize("kind", FLOAT8_DTYPES)
def test_float8_detector_products_match_float32(tmp_path, backend, kind):
    rng = np.random.default_rng(822)
    codes = (
        rng.integers(120, 133, (33, 17, 16, 16), dtype=np.uint8)
        if kind == "float8_e8m0fnu"
        else rng.integers(32, 80, (33, 17, 16, 16), dtype=np.uint8)
    )
    values = codes.view(FLOAT8_DTYPES[kind])
    mask = np.zeros((16, 16), np.uint32)
    mask[0, 0] = 1
    codes[..., 0, 0] = 255
    original = tmp_path / "fp8.h5"
    baseline = tmp_path / "float32.h5"
    write_float8_h5(original, values, mask)
    with h5py.File(baseline, "w") as handle:
        handle.create_dataset("entry/data/data", data=values.astype(np.float32))
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask", data=mask
        )
    with (
        io.load(original, backend=backend, verbose=False) as data,
        io.load(baseline, backend=backend, verbose=False) as reference,
    ):
        assert not data.data.valid_pixels[0, 0]
        assert not reference.data.valid_pixels[0, 0]
        candidate, expected = detector.prepare(data), detector.prepare(reference)
        rows, cols = np.indices((16, 16))
        for radius in (0, 3.5, 20):
            selection = (rows - 6.25) ** 2 + (cols - 8.75) ** 2 < radius**2
            np.testing.assert_array_equal(
                candidate.masked_sum(selection), expected.masked_sum(selection)
            )
        np.testing.assert_allclose(
            candidate.mean_dp(), expected.mean_dp(), rtol=1e-6, atol=0
        )
        np.testing.assert_array_equal(candidate.frame(0), expected.frame(0))
        for reduction in ("mean", "sum", "max"):
            np.testing.assert_allclose(
                candidate.reduce_frames([0, 510, 560], mode=reduction),
                expected.reduce_frames([0, 510, 560], mode=reduction),
                rtol=1e-6,
                atol=0,
            )


@pytest.mark.parametrize("kind", FLOAT8_DTYPES)
def test_float8_ssb_matches_float32_and_survives_reopen(tmp_path, backend, kind):
    codes = (
        np.random.default_rng(28).integers(120, 133, (16, 16, 16, 16), dtype=np.uint8)
        if kind == "float8_e8m0fnu"
        else np.random.default_rng(28).integers(
            32, 80, (16, 16, 16, 16), dtype=np.uint8
        )
    )
    values = codes.view(FLOAT8_DTYPES[kind])
    baseline = tmp_path / "float32.npy"
    np.save(baseline, values.astype(np.float32))
    settings = {
        "backend": backend,
        "voltage_kV": 300,
        "semiangle_mrad": 20,
        "scan_sampling_A": (0.5, 0.5),
        "detector_sampling_mrad": (4, 4),
        "bright_field_center_px": (7.5, 7.5),
        "bright_field_radius_px": 5,
    }

    def phases(data):
        with SSB(data, **settings) as session:
            result = []
            for tilt in (None, (1.0, -2.0)):
                session.sample_tilt_mrad = tilt
                session.depth_spread_nm = 5 if tilt else None
                phase = session.reconstruct(
                    aberrations={"C10_nm": 2}, upsampling_factor=2, compute_loss=False
                ).phase
                result.append(phase.get() if backend == "cuda" else np.asarray(phase))
            return result

    with io.load(baseline, backend=backend, verbose=False) as reference:
        expected = phases(reference)
    saved = tmp_path / "float8.qem"
    with io.load(values, backend=backend, verbose=False) as data:
        before = phases(data)
        io.save(saved, data)
    with io.load(saved, backend=backend, verbose=False) as data:
        after = phases(data)
    for original, reopened, reference in zip(before, after, expected, strict=True):
        np.testing.assert_array_equal(original, reopened)
        np.testing.assert_array_equal(original, reference)
