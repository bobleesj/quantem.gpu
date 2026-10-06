import json
from importlib.resources import files

import numpy as np
import pytest

from quantem.gpu import geometry
from quantem.gpu.io.dataset import Dataset4dstemGPU


def _scan_rotation_gold() -> tuple[np.ndarray, list[dict]]:
    fixture = json.loads(
        files("quantem.gpu")
        .joinpath("parity/scan_rotation_v1.json")
        .read_text(encoding="utf-8")
    )
    source = fixture["source"]
    data = np.asarray(source["values"], dtype=np.uint16).reshape(source["shape"])
    return data, fixture["cases"]


def _gold_expected(case: dict) -> np.ndarray:
    return np.asarray(case["expected_values"], dtype=np.uint16).reshape(
        case["output_shape"]
    )


def _indexed_4dstem(
    scan_shape: tuple[int, int] = (3, 5),
    detector_shape: tuple[int, int] = (2, 3),
) -> np.ndarray:
    shape = (*scan_shape, *detector_shape)
    return np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)


def _bilinear_reference(
    data: np.ndarray,
    angle_degrees: float,
    output_shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros((*output_shape, *data.shape[2:]), dtype=np.float32)
    angle_radians = np.deg2rad(angle_degrees)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    for output_row in range(output_shape[0]):
        for output_column in range(output_shape[1]):
            row = output_row - (output_shape[0] - 1) / 2
            column = output_column - (output_shape[1] - 1) / 2
            source_column = cosine * column - sine * row + (data.shape[1] - 1) / 2
            source_row = sine * column + cosine * row + (data.shape[0] - 1) / 2
            row0 = int(np.floor(source_row))
            column0 = int(np.floor(source_column))
            row_fraction = source_row - row0
            column_fraction = source_column - column0
            for row_offset, column_offset, weight in (
                (0, 0, (1 - row_fraction) * (1 - column_fraction)),
                (0, 1, (1 - row_fraction) * column_fraction),
                (1, 0, row_fraction * (1 - column_fraction)),
                (1, 1, row_fraction * column_fraction),
            ):
                source_r = row0 + row_offset
                source_c = column0 + column_offset
                if 0 <= source_r < data.shape[0] and 0 <= source_c < data.shape[1]:
                    result[output_row, output_column] += (
                        data[source_r, source_c].astype(np.float32) * weight
                    )
    return result


def test_rotate_scan_orients_a_loaded_90_degree_acquisition() -> None:
    raw, cases = _scan_rotation_gold()
    case = cases[0]
    loaded = Dataset4dstemGPU(
        raw,
        {
            "scan_shape": raw.shape[:2],
            "working_shape": raw.shape,
            "scan_sampling": (0.25, 0.5),
            "source": "acquisition_020",
        },
    )

    oriented = geometry.rotate_scan(loaded, angle_degrees=case["angle_degrees"])

    np.testing.assert_array_equal(oriented.data, _gold_expected(case))
    assert oriented.shape == oriented.data.shape
    assert loaded.shape == raw.shape
    assert oriented.data.flags.c_contiguous
    assert oriented.data.dtype == np.uint16
    assert oriented.metadata["source"] == "acquisition_020"
    assert oriented.metadata["scan_shape"] == tuple(case["output_shape"][:2])
    assert oriented.metadata["scan_sampling"] == (0.5, 0.25)
    assert oriented.metadata["scan_rotation_history"][-1] == {
        "angle_degrees": -90.0,
        "interpolation": "exact",
        "output_shape": "full",
        "source_scan_shape": (3, 5),
        "result_scan_shape": (5, 3),
    }


def test_rotate_scan_arbitrary_angle_matches_bilinear_reference() -> None:
    raw = _indexed_4dstem(scan_shape=(4, 5), detector_shape=(2, 2))
    angle_degrees = 31.0

    rotated, valid = geometry.rotate_scan(
        raw,
        angle_degrees=angle_degrees,
        output_shape="same",
        return_valid_mask=True,
    )

    expected = _bilinear_reference(raw, angle_degrees, raw.shape[:2])
    np.testing.assert_allclose(rotated, expected, rtol=0, atol=1e-5)
    assert rotated.dtype == np.float32
    assert valid.shape == raw.shape[:2]
    assert not valid[0, 0]
    assert valid[raw.shape[0] // 2, raw.shape[1] // 2]


def test_rotate_scan_preserves_torch_residency_and_counts() -> None:
    torch = pytest.importorskip("torch")
    raw_np, cases = _scan_rotation_gold()
    case = cases[1]
    raw = torch.as_tensor(raw_np)

    rotated = geometry.rotate_scan(raw, angle_degrees=case["angle_degrees"])

    torch.testing.assert_close(rotated, torch.as_tensor(_gold_expected(case)))
    assert rotated.device == raw.device
    assert rotated.dtype == torch.uint16
    assert rotated.is_contiguous()


def test_rotate_scan_mps_matches_shared_gold() -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("A Torch MPS device is required for scan-rotation parity")
    raw_np, cases = _scan_rotation_gold()
    raw = torch.as_tensor(raw_np, device="mps")

    for case in cases:
        rotated = geometry.rotate_scan(raw, angle_degrees=case["angle_degrees"])
        expected = torch.as_tensor(_gold_expected(case), device="mps")
        torch.testing.assert_close(rotated, expected)
        assert rotated.device.type == "mps"
        assert rotated.dtype == torch.uint16


def test_rotate_scan_torch_cuda_matches_shared_gold() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A Torch CUDA device is required for scan-rotation parity")
    raw_np, cases = _scan_rotation_gold()
    raw = torch.as_tensor(raw_np, device="cuda")

    for case in cases:
        rotated = geometry.rotate_scan(raw, angle_degrees=case["angle_degrees"])
        expected = torch.as_tensor(_gold_expected(case), device="cuda")
        torch.testing.assert_close(rotated, expected)
        assert rotated.device.type == "cuda"
        assert rotated.dtype == torch.uint16


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cupy") is None,
    reason="CuPy is required for CUDA scan-rotation parity",
)
def test_rotate_scan_cuda_matches_reference() -> None:
    import cupy as cp

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("A CUDA device is required for CUDA scan-rotation parity")
    raw_np, cases = _scan_rotation_gold()
    raw = cp.asarray(raw_np)

    for case in cases:
        exact = geometry.rotate_scan(raw, angle_degrees=case["angle_degrees"])
        np.testing.assert_array_equal(cp.asnumpy(exact), _gold_expected(case))

    arbitrary_np = _indexed_4dstem(scan_shape=(6, 7), detector_shape=(3, 4))
    arbitrary_source = cp.asarray(arbitrary_np)
    arbitrary = geometry.rotate_scan(
        arbitrary_source,
        angle_degrees=17.0,
        output_shape="same",
    )

    np.testing.assert_allclose(
        cp.asnumpy(arbitrary),
        _bilinear_reference(arbitrary_np, 17.0, arbitrary_np.shape[:2]),
        rtol=2e-7,
        atol=2e-6,
    )


def test_rotated_qem_preserves_shape_and_calibration(tmp_path):
    from copy import deepcopy
    from quantem.gpu import io

    values = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
    source_path, target_path = tmp_path / "source.qem", tmp_path / "rotated.qem"
    io.save(source_path, values, backend="cpu", metadata={"scan_sampling_A": [0.4, 0.6]})
    with io.load(source_path, backend="cpu", representation="dense", verbose=False) as source:
        original = deepcopy(source.metadata)
        rotated = geometry.rotate_scan(source, 90)
        io.save(target_path, rotated, backend="cpu")
        assert source.metadata["scientific_metadata"] == original["scientific_metadata"]
        np.testing.assert_array_equal(source.metadata["scan_sampling_A"], original["scan_sampling_A"])
    with io.load(target_path, backend="cpu", representation="dense", verbose=False) as loaded:
        np.testing.assert_array_equal(loaded.data, np.rot90(values, axes=(0, 1)))
        assert loaded.metadata["scan_sampling_A"] == pytest.approx([0.6, 0.4])
        assert [axis["size"] for axis in loaded.metadata["scientific_metadata"]["axes"]] == list(loaded.shape)


def test_rotate_scan_rotates_an_encoded_acquisition_exactly(tmp_path) -> None:
    """io.load output rotates into a new encoded acquisition with the dense reference's counts."""
    from quantem.gpu import detector, io

    _cuda()
    counts = np.random.default_rng(5).poisson(30, (13, 21, 16, 16)).astype(np.uint16)
    cases = [(angle, shape, "auto", 0) for angle in (90, -90, 180) for shape in ("full", "same")]
    cases += [(30.0, "full", "nearest", 7), (-45.0, "same", "nearest", 0)]
    with io.load(
        _write_master(tmp_path, counts), backend="cuda", scan_shape=(13, 21), verbose=False
    ) as loaded:
        for angle, output_shape, interpolation, fill_value in cases:
            options = {
                "output_shape": output_shape, "interpolation": interpolation,
                "fill_value": fill_value, "return_valid_mask": True,
            }
            rotated, valid = geometry.rotate_scan(loaded, angle, **options)
            expected, expected_valid = geometry.rotate_scan(counts, angle, **options)
            assert rotated.representation.value == "encoded"
            np.testing.assert_array_equal(rotated.read().cpu().numpy(), expected)
            np.testing.assert_array_equal(valid, expected_valid)
            rotated.close()

        # Detector products and saving treat the rotation like any loaded acquisition.
        oriented = np.rot90(counts, -1, axes=(0, 1))
        mask = np.zeros((16, 16), bool)
        mask[4:9, 5:12] = True
        rotated = geometry.rotate_scan(loaded, -90)
        np.testing.assert_array_equal(detector.mean(rotated), detector.mean(loaded))
        np.testing.assert_array_equal(
            detector.prepare(rotated).masked_sum_exact(mask),
            oriented[..., mask].sum(axis=-1, dtype=np.uint64),
        )
        io.save(tmp_path / "rotated.qem", rotated)
        rotated.close()
        with io.load(tmp_path / "rotated.qem", backend="cuda", verbose=False) as reopened:
            np.testing.assert_array_equal(reopened.read().cpu().numpy(), oriented)
            assert reopened.metadata["scan_rotation_history"][-1]["angle_degrees"] == -90.0
        np.testing.assert_array_equal(loaded.read().cpu().numpy(), counts)


def test_rotate_scan_never_interpolates_encoded_counts(tmp_path) -> None:
    from quantem.gpu import io

    _cuda()
    counts = np.ones((4, 6, 8, 8), np.uint16)
    with io.load(
        _write_master(tmp_path, counts), backend="cuda", scan_shape=(4, 6), verbose=False
    ) as loaded:
        for interpolation in ("auto", "bilinear"):
            with pytest.raises(NotImplementedError, match="float32"):
                geometry.rotate_scan(loaded, 30, interpolation=interpolation)
        for fill_value in (0.5, -1, 70000):
            with pytest.raises(ValueError, match="uint16 count"):
                geometry.rotate_scan(loaded, 30, interpolation="nearest", fill_value=fill_value)


def test_rotate_scan_reports_that_apple_gpus_cannot_rotate_encoded_data(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("An Apple GPU (MPS) is required.")
    from quantem.gpu import io

    counts = np.ones((4, 6, 8, 8), np.uint16)
    with (
        io.load(
            _write_master(tmp_path, counts), backend="mps", scan_shape=(4, 6), verbose=False
        ) as loaded,
        pytest.raises(NotImplementedError, match=r"Apple GPU \(MPS\)"),
    ):
        geometry.rotate_scan(loaded, 90)


def _cuda():
    """Skip where no CUDA device can hold an encoded acquisition."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("A CUDA device is required for encoded scan rotation.")


def _write_master(folder, counts: np.ndarray):
    """Write counts as an Arina-style master: bitshuffle/LZ4 frames in scan order."""
    import h5py
    import hdf5plugin

    path = folder / "rotation_master.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=counts.reshape(-1, *counts.shape[2:]),
            chunks=(1, *counts.shape[2:]),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = (
            counts.shape[0] * counts.shape[1]
        )
    return path
