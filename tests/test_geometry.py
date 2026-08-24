import json
from importlib.resources import files

import numpy as np
import pytest

from quantem.gpu import geometry
from quantem.gpu.io.load import LoadResult


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
    loaded = LoadResult(
        raw,
        {
            "scan_shape": raw.shape[:2],
            "scan_sampling": (0.25, 0.5),
            "source": "acquisition_020",
        },
    )

    oriented = geometry.rotate_scan(loaded, angle_degrees=case["angle_degrees"])

    np.testing.assert_array_equal(oriented.data, _gold_expected(case))
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
