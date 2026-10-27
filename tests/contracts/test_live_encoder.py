"""A live virtual detector must agree before and after ANS block completion."""

import threading

import numpy as np
import pytest

from quantem.gpu.streaming import LiveEncoder


def test_partial_scan_and_detector_changes_preserve_every_arrived_count():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA hardware required")
    raw = (cp.arange(96 * 96 * 16 * 16, dtype=cp.uint32) % 65536).astype(cp.uint16).reshape(-1, 16, 16)
    live = LiveEncoder((96, 96), (16, 16))
    session = live.detector_session()
    try:
        for count in (128, 4096, 4352, 9216):
            live.add(live.received, raw[live.received:count].copy())
            for offset in (0, 1):
                mask = (np.indices((16, 16)).sum(0) + offset) % 3 == 0
                image = session.masked_sum(mask).reshape(-1)
                expected = raw[:count, cp.asarray(mask)].sum(1, dtype=cp.uint64).astype(cp.float32).get()
                np.testing.assert_array_equal(image[:count], expected)
                assert np.isnan(image[count:]).all()
                np.testing.assert_array_equal(session.masked_sum(mask).reshape(-1), image)
                frozen = image.copy()
                session.masked_sum(~mask)
                np.testing.assert_array_equal(image, frozen)
            np.testing.assert_array_equal(session.frame(count - 1), raw[count - 1].get())
            expected_mean = (raw[:count].sum(0, dtype=cp.uint64).astype(cp.float64) / count).astype(cp.float32).get()
            np.testing.assert_array_equal(session.mean_dp(), expected_mean)
            if count < live.total:
                assert np.isnan(session.frame(count)).all()
        dataset = live.finish()
        np.testing.assert_array_equal(dataset.data.decode_scan_range_device(0, live.total).get(), raw.get())
    finally:
        session.close()
        live.release()


def test_borrowed_tail_survives_append_and_compression_rollover():
    """A held preview prefix stays exact while the receiver advances to a new block."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA hardware required")
    raw = (cp.arange(4352 * 8 * 8, dtype=cp.uint32) % 65536).astype(cp.uint16).reshape(-1, 8, 8)
    live = LiveEncoder((16, 512), (8, 8))
    session = live.detector_session()
    mask = np.indices((8, 8)).sum(0) % 3 == 0
    try:
        live.add(0, raw[:128].copy())
        first_image = session.masked_sum(mask)
        frozen_image, borrowed = first_image.copy(), session.tail
        for count in (256, 4352):
            live.add(live.received, raw[live.received:count].copy())
            assert bool(cp.array_equal(borrowed, raw[:128]))
            image = session.masked_sum(mask).reshape(-1)
            expected = raw[:count, cp.asarray(mask)].sum(1, dtype=cp.uint64).astype(cp.float32).get()
            np.testing.assert_array_equal(image[:count], expected)
            assert np.isnan(image[count:]).all()
            np.testing.assert_array_equal(session.frame(count - 1), raw[count - 1].get())
            np.testing.assert_array_equal(first_image, frozen_image)
    finally:
        session.close()
        live.release()


def test_unchanged_aperture_only_decodes_new_counts_across_ans_boundary(monkeypatch):
    from quantem.gpu.resident.cuda.counts import StreamedCounts

    cp = pytest.importorskip("cupy")
    raw = cp.full((4608, 32, 32), 65535, cp.uint16)
    raw[:, ::3, ::3] = 17
    flags = np.zeros((32, 32), np.uint32)
    flags[0, 0] = 1
    live = LiveEncoder((9, 512), (32, 32), pixel_mask=flags, hot_pixel_correction="none")
    session = live.detector_session()
    decode, ranges = StreamedCounts.decode_scan_range_device, []

    def tracked_decode(self, first, stop, **kwargs):
        ranges.append((first, stop))
        return decode(self, first, stop, **kwargs)

    monkeypatch.setattr(StreamedCounts, "decode_scan_range_device", tracked_decode)
    mask = np.ones((32, 32), bool)
    first_image = None
    try:
        for count in (128, 4096, 4352, 4608):
            live.add(live.received, raw[live.received:count].copy())
            image = session.masked_sum(mask).reshape(-1)
            expected = raw[:count, cp.asarray(flags == 0)].sum(1, dtype=cp.uint64).astype(cp.float32).get()
            np.testing.assert_array_equal(image[:count], expected)
            assert np.isnan(image[count:]).all()
            if first_image is None:
                first_image = image
                frozen = image.copy()
            np.testing.assert_array_equal(first_image, frozen)
        assert ranges[0][0] == 128
        assert ranges[-1][1] == 4608
        assert all(start >= 128 for start, _ in ranges)
        mask[:, :16] = False
        expected = raw[:, cp.asarray(mask & (flags == 0))].sum(1, dtype=cp.uint64).astype(cp.float32).get()
        np.testing.assert_array_equal(session.masked_sum(mask).reshape(-1), expected)
    finally:
        session.close()
        live.release()


def test_selected_frame_does_not_wait_for_a_virtual_image(monkeypatch):
    """A DP drag can read both compressed and arriving frames during a BF update."""
    cp = pytest.importorskip("cupy")
    raw = cp.arange(4352 * 8 * 8, dtype=cp.uint32).astype(cp.uint16).reshape(-1, 8, 8)
    live = LiveEncoder((16, 512), (8, 8))
    live.add(0, raw.copy())
    session = live.detector_session()
    entered, resume = threading.Event(), threading.Event()
    snapshot, errors = session._snapshot, []

    def paused_snapshot():
        snapshot()
        entered.set()
        assert resume.wait(5)

    def virtual_image():
        try:
            session.masked_sum(np.ones((8, 8), bool))
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(session, "_snapshot", paused_snapshot)
    worker = threading.Thread(target=virtual_image, daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        for index in (100, 4200):
            np.testing.assert_array_equal(session.frame(index), raw[index].get())
        assert not resume.is_set() and worker.is_alive()
    finally:
        resume.set()
        worker.join(timeout=5)
        session.close()
        live.release()
    assert not worker.is_alive() and not errors
