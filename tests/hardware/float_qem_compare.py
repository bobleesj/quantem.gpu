"""Full-resolution product/timing checks; inputs and reports stay caller-owned."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from quantem.gpu import io, detector

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("reference")
parser.add_argument("output")
parser.add_argument("--backend", required=True)
args = parser.parse_args()


def host(value):
    if hasattr(value, "to_numpy"):
        try:
            return value.to_numpy()
        finally:
            value.release()
    return value.get() if hasattr(value, "get") else value.detach().cpu().numpy()


def difference(actual, expected):
    absolute = np.abs(actual.astype(np.float64) - expected)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    return {
        "max_absolute": float(absolute.max()),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "max_relative_to_peak": float(absolute.max() / scale),
    }


started = time.perf_counter()
with io.load(args.input, backend=args.backend) as loaded:
    source = loaded.data
    source.synchronize()
    report = {
        "backend": args.backend,
        "load_seconds": time.perf_counter() - started,
        "metadata": loaded.metadata["load_timings"],
        "resident_bytes": source.nbytes,
        "shape": source.shape,
        "products": {},
        "timings": {},
        "dp_arms": [],
    }
    session = detector.prepare(loaded)
    expected_points = np.fromfile(args.reference, np.float32).reshape(-1, 128, 128)
    for n, index in enumerate([0, 8191, 16383]):
        actual = host(source.extract_diffraction_device(*divmod(index, 128)))
        np.testing.assert_array_equal(
            actual.view(np.uint32), expected_points[n].view(np.uint32)
        )
    candidate_class = type(source._lanes)
    if args.backend == "cuda":
        from quantem.gpu._compact.streamed import StreamedCounts as Reference
    else:
        from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts as Reference
    frozen = {}
    for arm in ["reference", "candidate", "reference"]:
        source._lanes.__class__ = Reference if arm == "reference" else candidate_class
        samples = []
        for index in [0, 511, 512, 8191, 16383] * 3:
            source.synchronize()
            start = time.perf_counter()
            output = source.extract_diffraction_device(*divmod(index, 128))
            source.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            sha = hashlib.sha256(host(output).tobytes()).hexdigest()
            if index in frozen:
                assert sha == frozen[index]
            else:
                frozen[index] = sha
        report["dp_arms"].append({"arm": arm, "milliseconds": samples})
    source._lanes.__class__ = candidate_class
    row, col = np.indices((128, 128))
    radius = (row - 64) ** 2 + (col - 64) ** 2
    masks = [
        radius <= 256,
        (radius >= 64) & (radius <= 256),
        (radius >= 1024) & (radius <= 3969),
        np.ones((128, 128), bool),
    ]
    expected = np.fromfile(args.reference + ".products", np.float32).reshape(
        4, 128, 128
    )
    report["detector_arms"] = []
    for arm in ["reference", "candidate", "reference"]:
        times = []
        for mask in masks * 3:
            source.synchronize()
            start = time.perf_counter()
            result = (
                source.products_device(mask)[0]
                if arm == "reference"
                else source.detector_sum_device(mask)
            )
            host(result)
            source.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        report["detector_arms"].append({"arm": arm, "milliseconds": times})
    for n, mask in enumerate(masks):
        start = time.perf_counter()
        actual = session.masked_sum(mask)
        report["timings"]["detector_" + str(n)] = time.perf_counter() - start
        report["products"]["detector_" + str(n)] = difference(actual, expected[n])
        report["products"]["detector_" + str(n)]["bit_exact"] = bool(
            np.array_equal(actual.view(np.uint32), expected[n].view(np.uint32))
        )
        np.testing.assert_allclose(actual, expected[n], rtol=2e-5, atol=2e-3)
    start = time.perf_counter()
    mean = session.mean_dp()
    report["timings"]["mean_dp"] = time.perf_counter() - start
    expected = np.fromfile(args.reference + ".mean", np.float32).reshape(128, 128)
    report["products"]["mean_dp"] = difference(mean, expected)
    np.testing.assert_allclose(mean, expected, rtol=2e-5, atol=2e-4)
    start = time.perf_counter()
    r, c = session.center_of_mass()
    report["timings"]["dpc"] = time.perf_counter() - start
    for name, actual in [("com-row", r), ("com-column", c)]:
        expected = np.fromfile(args.reference + "." + name, np.float32).reshape(
            128, 128
        )
        expected -= np.nanmean(expected)
        report["products"][name] = difference(actual, expected)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-4)
    report["peak_decode_bytes"] = source.peak_decode_bytes
    logical = hashlib.sha256()
    start = time.perf_counter()
    for first in range(0, 128 * 128, 256):
        logical.update(
            host(source.decode_scan_range_device(first, first + 256)).tobytes()
        )
    assert logical.hexdigest() == source.header["logical_sha256"]
    report["logical_sha256"] = logical.hexdigest()
    report["full_bit_verification_seconds"] = time.perf_counter() - start
    try:
        source.decode_scan_range_device(0, 513)
    except ValueError:
        report["oversized_decode_rejected"] = True
    else:
        raise AssertionError("Oversized decode was admitted")
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
