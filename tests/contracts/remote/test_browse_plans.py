"""Every browse plan of an encoded resident equals the old dense route, bit for bit.

The browse service used to decode a dense CUDA volume per plan: detector bins
summed into uint32 while loading, the scan crop selected positions, scan bins
summed with zero-padded partial edge bins into uint32,
and the products came from a detector session over that dense volume. These
tests rebuild exactly that dense oracle from the known counts of a small
synthetic Arina acquisition and compare it with what the service now computes
from the one encoded resident ``io.load`` returns.
"""

import os
import weakref
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(
    os.environ.get("QEM_TEST_BACKEND") != "cuda",
    reason="Set QEM_TEST_BACKEND=cuda to serve an encoded resident on a physical GPU.",
)

from fastapi.testclient import TestClient

from quantem.gpu import detector, dpc
from quantem.gpu.io.save import save_compressed_arina_h5
from quantem.gpu.remote import create_app

SESSION = "detector/session"
SCAN_SHAPE = (7, 6)
DETECTOR_SHAPE = (16, 16)
CROP = (1, 6, 0, 5)
MODES = [
    ("BF", 0.0, 1.0),
    ("ABF", 0.5, 1.0),
    ("ADF", 1.0, 2.0),
    ("HAADF", 1.5, 3.0),
    ("DF", 1.0, 4.0),
    ("CoMx", 0.0, 1.0),
    ("CoMy", 0.0, 1.0),
    ("CoMmag", 0.0, 1.0),
    ("DPC", 0.0, 1.0),
    ("iCoM", 0.0, 1.0),
]
SHAPES = [("circle", 0.0, 2.5), ("square", 0.0, 1.5), ("annulus", 1.0, 3.0)]


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory):
    """Write a bitshuffle-LZ4 Arina master whose counts the tests know exactly."""
    root = tmp_path_factory.mktemp("browse-plans")
    rows, columns = np.ogrid[: DETECTOR_SHAPE[0], : DETECTOR_SHAPE[1]]
    disk = (rows - 7.3) ** 2 + (columns - 8.1) ** 2 <= 4.2**2
    counts = np.random.default_rng(7).poisson(
        np.where(disk, 200.0, 3.0), size=(*SCAN_SHAPE, *DETECTOR_SHAPE)
    ).astype(np.uint16)
    master = root / SESSION / "sample_master.h5"
    save_compressed_arina_h5(master, counts, dtype="u16", compression_backend="hdf5")
    app = create_app(root)
    with TestClient(app) as client:
        yield app.state.browse_service, client, master, counts


def _dense_products(counts, *, det_bin, scan_bin, crop):
    """Compute every product the way the dense CUDA browse route did."""
    cp = pytest.importorskip("cupy")

    row_start, row_stop, column_start, column_stop = crop or (0, SCAN_SHAPE[0], 0, SCAN_SHAPE[1])
    dense = cp.asarray(counts[row_start:row_stop, column_start:column_stop])
    if det_bin > 1:
        rows, columns = dense.shape[:2]
        dense = dense.reshape(
            rows, columns, DETECTOR_SHAPE[0] // det_bin, det_bin, DETECTOR_SHAPE[1] // det_bin, det_bin
        ).sum(axis=(3, 5), dtype=cp.uint32)
    if scan_bin > 1:
        rows, columns = dense.shape[:2]
        padded_rows = -(-rows // scan_bin) * scan_bin
        padded_columns = -(-columns // scan_bin) * scan_bin
        padded = cp.zeros((padded_rows, padded_columns, *dense.shape[2:]), dtype=dense.dtype)
        padded[:rows, :columns] = dense
        dense = padded.reshape(
            padded_rows // scan_bin, scan_bin, padded_columns // scan_bin, scan_bin, *dense.shape[2:]
        ).sum(axis=(1, 3), dtype=cp.uint32)
    session = detector.prepare(dense)
    center, radius = detector.fit_probe(np.asarray(session.mean_dp(), dtype=np.float32))
    detector_rows, detector_columns = dense.shape[2:]
    grid_rows, grid_columns = np.ogrid[:detector_rows, :detector_columns]
    distance_squared = (grid_rows - center[0]) ** 2 + (grid_columns - center[1]) ** 2
    com_row, com_column = session.center_of_mass()
    products = {}
    for mode, inner, outer in MODES:
        if mode == "CoMy":
            products[mode] = com_row
        elif mode == "CoMx":
            products[mode] = com_column
        elif mode in {"CoMmag", "DPC"}:
            products[mode] = np.hypot(com_row, com_column).astype(np.float32)
        elif mode == "iCoM":
            products[mode] = np.asarray(dpc.integrate(com_row, com_column), dtype=np.float32)
        else:
            inner_pixels = max(0.0, inner * radius)
            outer_pixels = max(inner_pixels + 1.0, outer * radius)
            mask = distance_squared <= outer_pixels**2
            if mode != "BF":
                mask &= distance_squared >= inner_pixels**2
            products[mode] = session.masked_sum_exact(mask)
    for shape, inner, outer in SHAPES:
        if shape == "square":
            mask = (np.abs(grid_rows - 3.0) <= outer) & (np.abs(grid_columns - 4.0) <= outer)
        else:
            distance = (grid_rows - 3.0) ** 2 + (grid_columns - 4.0) ** 2
            mask = distance <= outer**2
            if shape == "annulus":
                mask &= distance >= inner**2
        products[shape] = session.masked_sum_exact(mask)
    scan_rows, scan_columns = dense.shape[:2]
    # The route clamps a position outside the view to its nearest edge.
    products["cbed"] = {
        (row, column): session.frame(
            min(row, scan_rows - 1) * scan_columns + min(column, scan_columns - 1)
        )
        for row, column in [(1, 2), (0, 0), (scan_rows - 1, scan_columns - 1), (99, 99)]
    }
    return products


def _image(response):
    assert response.status_code == 200, response.text
    dtype = response.headers["x-dtype"]
    shape = int(response.headers["x-height"]), int(response.headers["x-width"])
    return np.frombuffer(response.content, dtype=dtype).reshape(shape)


@pytest.mark.parametrize("crop", [None, CROP], ids=["full", "crop"])
@pytest.mark.parametrize("scan_bin", [1, 2])
@pytest.mark.parametrize("det_bin", [1, 2])
def test_binned_and_cropped_products_equal_dense_route(acquisition, det_bin, scan_bin, crop):
    _service, client, master, counts = acquisition
    expected = _dense_products(counts, det_bin=det_bin, scan_bin=scan_bin, crop=crop)
    plan = {"session": SESSION, "file": master.name, "det_bin": det_bin, "scan_bin": scan_bin}
    if crop is not None:
        plan.update(dict(zip(("row_start", "row_stop", "column_start", "column_stop"), crop)))

    for mode, inner, outer in MODES:
        response = client.get(
            "/api/browse/realspace",
            params={**plan, "mode": mode, "inner": inner, "outer": outer, "ensure_resident": True},
        )
        image = _image(response)
        if mode.startswith(("Co", "DPC", "iCoM")):
            assert response.headers["x-dtype"] == "<f4"
            np.testing.assert_array_equal(image, expected[mode], err_msg=mode)
        else:
            assert response.headers["x-dtype"] == "<u4"
            np.testing.assert_array_equal(image, expected[mode].astype(np.uint32), err_msg=mode)
    for shape, inner, outer in SHAPES:
        image = _image(
            client.get(
                "/api/browse/realspace-shape",
                params={**plan, "shape": shape, "cx": 4.0, "cy": 3.0, "inner": inner, "outer": outer},
            )
        )
        np.testing.assert_array_equal(image, expected[shape].astype(np.uint32), err_msg=shape)
    for (row, column), pattern in expected["cbed"].items():
        image = _image(client.get("/api/browse/cbed", params={**plan, "sx": row, "sy": column}))
        np.testing.assert_array_equal(image, pattern.astype(np.uint32), err_msg=f"cbed {row} {column}")


def test_every_plan_reads_one_resident_and_bad_plans_never_evict_it(acquisition, monkeypatch):
    service, client, master, _counts = acquisition
    common = {"session": SESSION, "file": master.name, "mode": "BF", "ensure_resident": True}
    assert client.get("/api/browse/realspace", params=common).status_code == 200
    resident = service.residency._entries[str(master)]
    monkeypatch.setattr(
        service.residency, "_load", lambda *_args: pytest.fail("a plan change reloaded the acquisition")
    )

    for plan in ({"det_bin": 2}, {"scan_bin": 4}, dict(zip(("row_start", "row_stop", "column_start", "column_stop"), CROP))):
        assert client.get("/api/browse/realspace", params={**common, **plan}).status_code == 200
    rejected = [
        client.get("/api/browse/realspace", params={**common, "det_bin": 3}),
        client.get("/api/browse/realspace", params={**common, "scan_bin": 3}),
        client.get(
            "/api/browse/realspace",
            params={**common, "row_start": 0, "row_stop": 8, "column_start": 0, "column_stop": 6},
        ),
        client.get("/api/browse/realspace", params={**common, "mode": "unknown"}),
    ]

    assert [response.status_code for response in rejected] == [400, 400, 400, 400]
    assert "does not divide the 16 x 16 detector" in rejected[0].text
    assert service.residency._entries[str(master)] is resident
    assert not resident.closed


def test_admission_sizes_the_encoded_resident_not_a_dense_decode(acquisition):
    service, _client, master, counts = acquisition
    stored = sum(path.stat().st_size for path in master.parent.iterdir())

    resident_bytes, peak_bytes = service.residency._expected_bytes(master)

    assert resident_bytes == min(stored, counts.nbytes)
    pixels = DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]
    assert peak_bytes == resident_bytes + (128 << 20) + counts.size // pixels * pixels * 64


def test_eviction_releases_the_resident_even_while_referenced(tmp_path):
    counts = np.random.default_rng(3).poisson(20.0, size=(8, 8, 16, 16)).astype(np.uint16)
    master = tmp_path / SESSION / "held_master.h5"
    save_compressed_arina_h5(master, counts, dtype="u16", compression_backend="hdf5")
    app = create_app(tmp_path)
    service = app.state.browse_service
    client = TestClient(app)
    assert client.get("/api/browse/realspace", params={"session": SESSION, "file": master.name}).status_code == 200
    resident = service.residency._entries[str(master)]
    held = resident.loaded
    backend = weakref.ref(resident.session._backend)

    with service.residency._lock:
        freed = service.residency._evict_one(resident.gpu)
    service.close()

    assert freed == held.metadata["physical_resident_bytes"]
    assert held.data.is_released
    assert backend() is None
    assert str(master) not in service.residency._entries
    assert Path(master).is_file()
