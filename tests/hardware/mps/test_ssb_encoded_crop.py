"""An encoded bright-field crop prepares the same SSB evidence as the decoded crop it replaces, bit for bit."""

import math
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Requires physical Apple GPU")
pytest.importorskip("Metal")
torch = pytest.importorskip("torch")
pytest.importorskip("mlx.core")

SCAN = (128, 128)
DETECTOR = (24, 26)
CENTER = (11.3, 12.6)
RADIUS = 6.2


def _counts():
    rng = np.random.default_rng(7)
    rows, cols = np.mgrid[:DETECTOR[0], :DETECTOR[1]]
    disk = np.hypot(rows - CENTER[0], cols - CENTER[1]) <= RADIUS + 0.5
    counts = rng.integers(0, 6, (*SCAN, *DETECTOR)).astype(np.uint16)
    counts += (disk * rng.integers(200, 900, (*SCAN, *DETECTOR))).astype(np.uint16)
    # one hot pixel inside the disk and one outside: stored, flagged invalid
    counts[..., 10, 12] = 60000
    counts[..., 2, 3] = 50000
    valid = np.ones(DETECTOR, bool)
    valid[10, 12] = valid[2, 3] = False
    return counts, valid


def _resident(counts, valid):
    from quantem.gpu.device.metal_runtime import shared_array
    from quantem.gpu.io.dataset import Dataset4dstemGPU
    from quantem.gpu.resident.mps.counts import MPSStreamedCounts
    from quantem.gpu.resident.mps.precision import upload

    source = MPSStreamedCounts(counts.shape, counts.dtype, valid)
    flat = counts.reshape(-1, *counts.shape[2:])
    for first in range(0, flat.shape[0], 512):
        staged = upload(np.ascontiguousarray(flat[first:first + 512]))
        source.append(shared_array(staged._mtl, staged.dtype, staged.shape))
        staged.release()
    return Dataset4dstemGPU(source, {"working_shape": counts.shape, "working_dtype": "uint16",
                                     "representation": "encoded"})


def _backend(data, center, flipped):
    from quantem.gpu.direct_ptycho.mps.backend import MpsDirectPtychoBackend

    return MpsDirectPtychoBackend(
        data, voltage_kV=300.0, semiangle_mrad=24.0, scan_sampling=(0.5, 0.5),
        detector_sampling_mrad=(4.0, 4.0), bright_field_intensity_threshold=0.0,
        bright_field_center_px=center[::-1] if flipped else center, bright_field_radius_px=RADIUS,
        scan_detector_rotation_deg=17.0,
    )


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("scan_region", [None, (8, 72, 4, 100)])
@pytest.mark.parametrize("chunk_bf", [37, 64])
@pytest.mark.parametrize("block_rows", [6, 18])
def test_encoded_crop_prepares_the_decoded_crop_bit_for_bit(flipped, scan_region, chunk_bf, block_rows, monkeypatch):
    """G(q, k), the selection, the DC value and a preview equal the decoded crop's: odd chunks pair FFT rows across
    scan rows, a rectangular region is padded with its mean pattern, a flipped session swaps the detector axes."""
    from quantem.gpu.direct_ptycho import mps
    from quantem.gpu.direct_ptycho.brightfield import bright_field_crop_region, crop_bright_field
    from quantem.gpu.direct_ptycho.mps.frames import EncodedBrightFieldCrop

    from quantem.gpu.direct_ptycho.mps.frames import EncodedBrightFieldFrames

    monkeypatch.setattr(mps.backend, "default_object_setup_chunk_bf", lambda: chunk_bf)
    # Vary scan-row blocks independently of BF chunks: reducing temporary
    # memory must preserve every Fourier value, including padded regions.
    monkeypatch.setattr(EncodedBrightFieldFrames, "block_bytes", block_rows * 128 * 26 * 26 * 2)
    monkeypatch.setattr(mps.prepared, "ENCODED_BF_PER_FFT", 16)
    counts, valid = _counts()
    loaded = _resident(counts, valid)
    try:
        dense, center, _, _, origin, _ = crop_bright_field(
            loaded, "mps", 0.0, RADIUS, bright_field_center_px=CENTER, scan_region=scan_region)
        _, _, _, region, _ = bright_field_crop_region(loaded, 0.0, RADIUS, bright_field_center_px=CENTER)
        assert (region[0], region[2]) == origin
        encoded = EncodedBrightFieldCrop(loaded, region, scan_region=scan_region)
        if flipped:
            dense, encoded = dense.swapaxes(-1, -2), encoded.swapaxes(-1, -2)
        assert tuple(encoded.shape) == tuple(dense.shape) and str(encoded.dtype) == str(dense.dtype)
        in_crop = (center[0] - origin[0], center[1] - origin[1])
        expected, observed = _backend(dense, in_crop, flipped), _backend(encoded, in_crop, flipped)
        for backend in (expected, observed):
            backend.cache_rotation(math.radians(17.0))
        np.testing.assert_array_equal(observed._selection.rows, expected._selection.rows)
        np.testing.assert_array_equal(observed._selection.cols, expected._selection.cols)
        assert observed._dc_value_override == expected._dc_value_override
        assert observed._prepared.dc_value == expected._prepared.dc_value
        assert observed.scan_shape == expected.scan_shape
        g_expected, g_observed = np.asarray(expected._prepared.g_qk), np.asarray(observed._prepared.g_qk)
        assert g_observed.shape == g_expected.shape
        assert np.array_equal(g_observed.view(np.uint32), g_expected.view(np.uint32))
        aberrations = {"C10": -40.0, "C12": 12.0, "phi12": 0.4}
        phase_expected, loss_expected = expected.phase_variance(aberrations)
        phase_observed, loss_observed = observed.phase_variance(aberrations)
        assert np.array_equal(phase_observed, phase_expected) and loss_observed == loss_expected
        columns = np.zeros((5, int(np.prod(observed.scan_shape))), np.float32)
        reference = np.zeros_like(columns)
        rows, cols = expected._selection.rows[3:8], expected._selection.cols[3:8]
        observed._frames.columns_float32_into(rows, cols, out=columns)
        expected._frames.columns_float32_into(rows, cols, out=reference)
        np.testing.assert_array_equal(columns, reference)
        for backend in (expected, observed):
            backend.close()
    finally:
        loaded.close()


def test_resident_session_keeps_the_crop_encoded_and_its_identity():
    """SSB(data) on an encoded MPS acquisition holds no decoded crop and records the decoded crop's identity sums."""
    from quantem.gpu import SSB
    from quantem.gpu.direct_ptycho.brightfield import crop_bright_field
    from quantem.gpu.direct_ptycho.mps.frames import EncodedBrightFieldCrop

    counts, valid = _counts()
    loaded = _resident(counts, valid)
    try:
        settings = dict(backend="mps", voltage_kV=300, semiangle_mrad=24, scan_sampling_A=0.5,
                        detector_sampling_mrad=4.0, scan_detector_rotation_deg=17.0,
                        aberrations={"C10_nm": -4.0, "C12_nm": 1.2, "phi12_deg": 23.0})
        ssb = SSB(loaded, **settings)
        assert isinstance(ssb._data, EncodedBrightFieldCrop)
        dense, *_ = crop_bright_field(loaded, "mps", 0.0, ssb.bright_field_radius_px,
                                      bright_field_center_px=ssb.bright_field_center_px)
        frames = dense.reshape(dense.shape[0] * dense.shape[1], -1)
        # Some Torch MPS builds lack the fused uint16-to-int64 reduction.
        # Cast on the GPU first; integer counts and their sums remain exact.
        per_frame = frames.to(torch.int64).sum(dim=1)
        positions = torch.arange(1, per_frame.numel() + 1, dtype=torch.int64, device=per_frame.device)
        assert ssb.resident_identity["bright_field_sums"] == [
            int(per_frame.sum()) % 2**64, int((per_frame * positions).sum()) % 2**64]
        assert ssb.resident_identity["shape"] == list(counts.shape)
        assert tuple(ssb._data.shape) == tuple(dense.shape)
        phase, loss = ssb.preview(settings["aberrations"], compute_loss=True)
        assert np.isfinite(phase).all() and np.isfinite(loss)
        with ssb.select_region((8, 72, 4, 100)) as region:
            assert isinstance(region._data, EncodedBrightFieldCrop)
            assert region.scan_shape == (128, 128)
            region_phase, _ = region.preview(settings["aberrations"], compute_loss=False)
            assert np.isfinite(region_phase).all()
        ssb.close()
    finally:
        loaded.close()


def test_tiled_512_loss_route_matches_the_whole_chunk(monkeypatch):
    """LOSS_ROW_IFFT_TILE_BF sends a 512 x 512 loss chunk through the tiled route (the default), None keeps the whole
    chunk: the same loss and phase bits."""
    from quantem.gpu import SSB
    from quantem.gpu.direct_ptycho.mps import reconstruct

    rng = np.random.default_rng(3)
    rows, cols = np.mgrid[:16, :16]
    disk = np.hypot(rows - 7.6, cols - 8.2) <= 5.4
    counts = (rng.integers(0, 4, (512, 512, 16, 16)) + disk * rng.integers(100, 400, (512, 512, 16, 16))).astype(np.uint16)
    loaded = _resident(counts, np.ones((16, 16), bool))
    try:
        ssb = SSB(loaded, backend="mps", voltage_kV=300, semiangle_mrad=24, scan_sampling_A=0.5,
                  detector_sampling_mrad=4.0, scan_detector_rotation_deg=17.0)
        aberrations = {"C10_nm": -4.0, "C12_nm": 1.2, "phi12_deg": 23.0}
        tilted = dict(sample_tilt_mrad=(0.4, -1.1), depth_spread_nm=6.0)
        monkeypatch.setattr(reconstruct, "LOSS_ROW_IFFT_TILE_BF", None)
        whole = [ssb.preview(aberrations, compute_loss=True), ssb.preview(aberrations, compute_loss=True, **tilted)]
        monkeypatch.setattr(reconstruct, "LOSS_ROW_IFFT_TILE_BF", 16)
        tiles = [ssb.preview(aberrations, compute_loss=True), ssb.preview(aberrations, compute_loss=True, **tilted)]
        for (phase_whole, loss_whole), (phase_tiles, loss_tiles) in zip(whole, tiles):
            assert np.array_equal(phase_tiles, phase_whole) and loss_tiles == loss_whole
        ssb.close()
    finally:
        loaded.close()


def test_region_session_frees_and_rebuilds_the_full_field_bit_for_bit(monkeypatch):
    """A region session frees the full field's G(q, k); the next full-field preview rebuilds the same bits, also after
    a rotation change retargeted the geometry."""
    from quantem.gpu import SSB
    from quantem.gpu.direct_ptycho.mps.hardware import require_mlx

    mx = require_mlx()
    info = mx.device_info()
    monkeypatch.setattr(mx, "device_info", lambda: {
        **info, "max_recommended_working_set_size": 0,
    })

    counts, valid = _counts()
    loaded = _resident(counts, valid)
    try:
        ssb = SSB(loaded, backend="mps", voltage_kV=300, semiangle_mrad=24, scan_sampling_A=0.5,
                  detector_sampling_mrad=4.0, scan_detector_rotation_deg=17.0)
        aberrations = {"C10_nm": -4.0, "C12_nm": 1.2, "phi12_deg": 23.0}
        ssb.preview(aberrations, compute_loss=False)
        ssb.set_rotation(197.0)
        before = ssb.preview(aberrations, compute_loss=True)
        g_before = np.asarray(ssb._compute._prepared.g_qk).copy()
        with ssb.select_region((8, 72, 4, 100)) as region:
            assert ssb._compute._prepared is None
            region.preview(aberrations, compute_loss=False)
        after = ssb.preview(aberrations, compute_loss=True)
        assert np.array_equal(np.asarray(ssb._compute._prepared.g_qk), g_before)
        assert np.array_equal(after[0], before[0]) and after[1] == before[1]
        ssb.close()
    finally:
        loaded.close()
