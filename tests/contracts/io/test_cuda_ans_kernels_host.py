"""Exercise actual integer kernel code serially, without claiming GPU evidence."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.parity.ans_counts_fixture import make_fixture


@pytest.fixture(scope="module")
def serial_kernels(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the serial kernel oracle.")
    root = Path(__file__).resolve().parents[2]
    destination = tmp_path_factory.mktemp("ans-host-kernels") / "kernels.so"
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-shared",
            "-fPIC",
            str(root / "parity/ans_kernel_host.cpp"),
            "-I",
            str(root.parent / "src/quantem/gpu/io/backends/cuda"),
            "-o",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(destination))

    def launch(name, count, *arguments):
        converted = [
            ctypes.c_void_p(argument.ctypes.data)
            if isinstance(argument, np.ndarray)
            else argument
            for argument in arguments
        ]
        for index in range(count):
            library.set_thread(ctypes.c_uint32(index))
            getattr(library, name)(*converted)

    return launch


def _inputs(encoded):
    return tuple(
        encoded[name]
        for name in (
            "payload",
            "offsets",
            "model_ids",
            "context_offsets",
            "symbols",
            "cumulative",
            "frequencies",
            "literal",
        )
    )


def test_actual_kernel_recurrence_and_products_match_independent_counts(serial_kernels):
    counts, encoded = make_fixture()
    inputs = _inputs(encoded)
    errors = np.zeros(1, np.uint32)
    scan_count, detector_count, block_frames, scale, streams = 15, 6, 4, 4, 24
    arguments = (
        ctypes.c_uint64(scan_count),
        ctypes.c_uint32(detector_count),
        ctypes.c_uint32(block_frames),
        ctypes.c_uint32(scale),
        ctypes.c_uint64(streams),
    )
    serial_kernels(
        "ans_validate", streams, *inputs, errors, *arguments, ctypes.c_uint32(65535)
    )
    assert errors[0] == 0
    requested = np.asarray([14, 0, 9, 14], np.uint64)
    output = np.empty((4, 2, 3), np.uint16)
    serial_kernels(
        "ans_diffraction",
        output.size,
        *inputs,
        requested,
        output,
        errors,
        ctypes.c_uint64(4),
        ctypes.c_uint32(6),
        ctypes.c_uint32(4),
        ctypes.c_uint32(4),
    )
    np.testing.assert_array_equal(output, counts.reshape(15, 2, 3)[requested])
    mask = np.asarray([[0, 1, 0], [1, 0, 1]], np.uint8)
    total = np.zeros((3, 5), np.uint64)
    serial_kernels(
        "ans_detector_sum", streams, *inputs, mask, total, errors, *arguments
    )
    np.testing.assert_array_equal(
        total, (counts * mask).sum(axis=(2, 3), dtype=np.uint64)
    )

    widths, lengths = np.empty(streams, np.uint8), np.empty(streams, np.uint64)
    serial_kernels(
        "ans_measure_packed", streams, *inputs, widths, lengths, errors, *arguments
    )
    offsets = np.concatenate(
        (np.zeros(1, np.uint64), np.cumsum(lengths, dtype=np.uint64))
    )
    words = np.empty(int(offsets[-1]), np.uint32)
    serial_kernels(
        "ans_write_packed", streams, *inputs, widths, offsets, words, errors, *arguments
    )
    assert errors[0] == 0
    assert 0 in widths and 16 in widths
    packed = (words, offsets, widths)
    serial_kernels(
        "packed_diffraction",
        output.size,
        *packed,
        requested,
        output,
        ctypes.c_uint64(4),
        ctypes.c_uint32(6),
        ctypes.c_uint32(4),
    )
    np.testing.assert_array_equal(output, counts.reshape(15, 2, 3)[requested])
    total.fill(0)
    serial_kernels(
        "packed_detector_sum",
        streams,
        *packed,
        mask,
        total,
        ctypes.c_uint64(15),
        ctypes.c_uint32(6),
        ctypes.c_uint32(4),
        ctypes.c_uint64(24),
    )
    np.testing.assert_array_equal(
        total, (counts * mask).sum(axis=(2, 3), dtype=np.uint64)
    )


@pytest.mark.parametrize("width", [0, 1, 7, 16])
def test_actual_transcode_handles_word_boundaries_and_short_streams(
    serial_kernels, width
):
    counts = np.full(15, (1 << width) - 1, np.uint16)
    counts[0] = 0
    inputs = (
        counts.view(np.uint8),
        np.asarray([0, counts.nbytes], np.uint64),
        np.asarray([0], np.uint32),
        np.asarray([0, 0], np.uint32),
        np.empty(0, np.uint16),
        np.empty(0, np.uint16),
        np.empty(0, np.uint16),
        np.ones(1, np.uint8),
    )
    arguments = (
        ctypes.c_uint64(15),
        ctypes.c_uint32(1),
        ctypes.c_uint32(15),
        ctypes.c_uint32(4),
        ctypes.c_uint64(1),
    )
    errors = np.zeros(1, np.uint32)
    widths, lengths = np.empty(1, np.uint8), np.empty(1, np.uint64)
    serial_kernels(
        "ans_measure_packed", 1, *inputs, widths, lengths, errors, *arguments
    )
    assert widths[0] == width
    assert lengths[0] == (15 * width + 31) // 32
    offsets = np.asarray([0, lengths[0]], np.uint64)
    words = np.empty(int(lengths[0]), np.uint32)
    serial_kernels(
        "ans_write_packed", 1, *inputs, widths, offsets, words, errors, *arguments
    )
    output = np.empty_like(counts)
    serial_kernels(
        "packed_decode_block",
        15,
        words,
        offsets,
        widths,
        output,
        ctypes.c_uint64(0),
        ctypes.c_uint32(15),
        ctypes.c_uint32(1),
    )
    np.testing.assert_array_equal(output, counts)
    assert errors[0] == 0


def test_actual_admission_rejects_bad_state_and_wrong_native_dtype(serial_kernels):
    _, encoded = make_fixture()
    errors = np.zeros(1, np.uint32)
    arguments = (
        ctypes.c_uint64(15),
        ctypes.c_uint32(6),
        ctypes.c_uint32(4),
        ctypes.c_uint32(4),
        ctypes.c_uint64(24),
    )
    serial_kernels(
        "ans_validate", 24, *_inputs(encoded), errors, *arguments, ctypes.c_uint32(255)
    )
    assert errors[0] & 2
    encoded["payload"][:4] = 0
    errors.fill(0)
    serial_kernels(
        "ans_validate",
        24,
        *_inputs(encoded),
        errors,
        *arguments,
        ctypes.c_uint32(65535),
    )
    assert errors[0] & 1
