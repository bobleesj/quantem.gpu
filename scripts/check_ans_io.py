"""Run strict acquisition tests and retain a small, evidence-derived table.

Examples
--------
python scripts/check_ans_io.py --backend mps --output /tmp/ans-mps.json
python scripts/check_ans_io.py --combine /tmp/ans-*.json --output /tmp/ans-table.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARRAY = "tests/hardware/test_array_ans_workflows.py::"
FLOAT = "tests/hardware/test_float_qem_resident.py::"
STRICT = "tests/hardware/test_ans_io_acceptance.py::"
NATIVE = "tests/hardware/metal/test_qem_acceptance.py::"
GROUPS = {
    "Resident selections and previews": {
        "python": ["tests/hardware/test_resident_selection.py"],
    },
    "Masked uint32 detector counts": {
        "python": ["tests/hardware/test_masked_count_narrowing.py"],
    },
    "NumPy uint8/uint16": {
        "python": [
            ARRAY + f"test_integer_original_native_outputs_and_qem[{dtype}-numpy]"
            for dtype in ("uint8", "uint16")
        ],
        "metal": [NATIVE + "test_count_numpy_export_reopen"],
    },
    "NumPy int32 exact narrowing": {
        "python": [ARRAY + "test_simulated_integer_counts_to_qem"],
    },
    "Exact promoted float64 measurements": {
        "python": [ARRAY + "test_exact_promoted_float_measurements"],
    },
    "Float32 detector geometries": {
        "python": [
            ARRAY + "test_float_detector_geometries",
            ARRAY + "test_float_ingestion_preserves_ieee_bits",
        ],
        "metal": [NATIVE + "test_float_numpy_native_qem_geometry"],
    },
    "Original HDF5 count ANS round trip": {
        "python": [STRICT + "test_original_hdf5_default_ans_roundtrip"],
    },
    "NumPy float32": {
        "python": [ARRAY + "test_float_original_to_qem_and_native_patterns[numpy]"],
        "metal": [NATIVE + "test_float_numpy_native_qem_geometry"],
    },
    "EMPAD-G1 processed float32": {
        "python": [
            ARRAY + "test_float_original_to_qem_and_native_patterns[empad-raw]",
            ARRAY + "test_float_original_to_qem_and_native_patterns[empad-xml]",
            ARRAY + "test_empad_multiple_detector_regions",
        ],
    },
    "EMPAD2 processed float32": {
        "python": [
            ARRAY + "test_float_original_to_qem_and_native_patterns[empad2-xml]"
        ],
    },
    "HDF5 float32 arrays": {
        "python": [
            ARRAY + "test_float_original_to_qem_and_native_patterns[" + name + "]"
            for name in ("hdf5-4d", "hdf5-3d", "hdf5-gzip")
        ],
    },
    "DM3 float32": {
        "python": [ARRAY + "test_float_original_to_qem_and_native_patterns[dm3]"],
    },
    "DM4 float32": {
        "python": [ARRAY + "test_float_original_to_qem_and_native_patterns[dm4]"],
    },
    "EMD calibrated arrays": {
        "python": ["tests/hardware/test_emd_ans_workflows.py"],
    },
    "QEM float32 exact bits": {
        "python": [
            FLOAT + "test_float_qem_preserves_bits_and_metadata_without_dense_residency"
        ],
        "metal": [NATIVE + "test_float_qem_bits"],
    },
    "Float32 GPU products and lifetime": {
        "python": [
            FLOAT + name
            for name in (
                "test_float_products_and_selected_mean_stay_on_device",
                "test_float_background_is_applied_once_and_survives_export",
                "test_float_reductions_keep_cancellation_and_nonfinite_semantics",
                "test_float_com_does_not_overflow_or_hide_invalid_frames",
                "test_corrupt_upload_releases_partial_resident",
            )
        ],
    },
    "ANS-only bounded ingestion": {
        "python": [
            STRICT + "test_default_array_ingestion_is_bounded_ans_only",
            FLOAT + "test_decode_limit_is_checked_before_allocating",
        ],
    },
    "Frozen MPS product hashes": {
        "mps": [ARRAY + "test_existing_float_products_preserve_gpu_baseline"],
    },
    "Zenodo 7464234 full collection": {
        "python": ["tests/hardware/test_zenodo_ans_roundtrip.py"],
    },
    "Zenodo 15084123 full collection": {
        "python": ["tests/hardware/test_tcmep_ans_roundtrip.py"],
    },
}
BACKENDS = ("cuda", "mps", "metal")


def _fingerprint() -> str:
    digest = hashlib.sha256()
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
    suffixes = {".py", ".cu", ".cuh", ".msl", ".swift", ".metal", ".h", ".c", ".cpp", ".json"}
    for folder in roots:
        for path in sorted(folder.rglob("*")):
            if path.suffix not in suffixes or not path.is_file():
                continue
            if any(part in {".build", "__pycache__", "Vendor"} for part in path.parts):
                continue
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _device(backend: str) -> dict:
    if backend == "cuda":
        import cupy as cp
        import torch

        if not cp.cuda.runtime.getDeviceCount() or not torch.cuda.is_available():
            raise RuntimeError("Physical CUDA device and CUDA-enabled Torch required.")
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return dict(
            name=properties["name"].decode(),
            runtime=cp.cuda.runtime.runtimeGetVersion(),
            torch=torch.__version__,
        )
    import Metal
    import torch

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None or (backend == "mps" and not torch.backends.mps.is_available()):
        raise RuntimeError("Physical Apple GPU required; no CPU fallback permitted.")
    return dict(
        name=str(device.name()), runtime=platform.mac_ver()[0], torch=torch.__version__
    )


def _selectors(backend: str) -> dict[str, list[str]]:
    return {
        name: group.get(backend, group.get("python", []) if backend != "metal" else [])
        for name, group in GROUPS.items()
    }


class _Results:
    def __init__(self):
        self.tests = {}
        self.collection_errors = []
        self.deselected = []

    def pytest_runtest_logreport(self, report):
        item = self.tests.setdefault(
            report.nodeid,
            dict(nodeid=report.nodeid, status="not-run", seconds=0.0, properties={}),
        )
        item["seconds"] += report.duration
        item["properties"].update(dict(report.user_properties))
        if report.failed:
            item.update(status="failed", detail=str(report.longrepr))
        elif (report.skipped or hasattr(report, "wasxfail")) and item[
            "status"
        ] != "failed":
            item.update(status="blocked", detail=str(report.longrepr))
        elif report.when == "call" and item["status"] == "not-run":
            item["status"] = "passed"

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)


def _cell(tests: list[dict]) -> str:
    if not tests:
        return "not-run"
    states = {test["status"] for test in tests}
    if "failed" in states:
        return "failed"
    return "passed" if states == {"passed"} else "blocked"


def _run(backend: str, output: Path, zenodo_root: Path | None) -> dict:
    report = dict(
        schema_version=1,
        scope="ANS acquisition API acceptance; not native UI, universal format support, or FPS qualification",
        backend=backend,
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_fingerprint=_fingerprint(),
        cells={},
        tests=[],
    )
    selected = _selectors(backend)
    try:
        report["device"] = _device(backend)
    except Exception as error:
        report.update(status="blocked", detail=str(error))
        report["cells"] = {
            name: "blocked" if nodes else "not-covered"
            for name, nodes in selected.items()
        }
        return report
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import quantem

    quantem.__path__ = [str(ROOT / "src/quantem"), *quantem.__path__]
    import quantem.gpu
    import pytest

    if not Path(quantem.gpu.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(
            "Tests imported a different checkout; stop rather than test stale installed code."
        )
    os.environ["QEM_TEST_BACKEND"] = backend
    os.environ["QGPU_ORIGINAL_READ_AHEAD"] = "0"
    if zenodo_root is not None:
        os.environ["QEM_ZENODO_ROOT"] = str(zenodo_root.resolve())
    results = _Results()
    nodes = list(dict.fromkeys(node for values in selected.values() for node in values))
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix="ans-acceptance-") as temporary:
        with output.with_suffix(".log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                code = pytest.main(
                    [*nodes, "-q", "-ra", "--basetemp=" + temporary + "/pytest"],
                    plugins=[results],
                )
    report["tests"] = list(results.tests.values())
    report["collection_errors"] = results.collection_errors
    report["deselected"] = results.deselected
    for name, prefixes in selected.items():
        matching = [
            test
            for test in report["tests"]
            if any(
                test["nodeid"] == prefix
                or test["nodeid"].startswith(prefix + "[")
                or test["nodeid"].startswith(prefix + "::")
                for prefix in prefixes
            )
        ]
        report["cells"][name] = _cell(matching) if prefixes else "not-covered"
    # A skipped/xfail/empty test is never a pass, even if pytest exits zero.
    active = [report["cells"][name] for name, nodes in selected.items() if nodes]
    report["source_unchanged"] = report["source_fingerprint"] == _fingerprint()
    report["status"] = (
        "passed"
        if code == 0
        and not results.deselected
        and active
        and set(active) == {"passed"}
        and report["source_unchanged"]
        else "failed"
    )
    report["pytest_exit_code"] = int(code)
    report["temporary_exports_deleted"] = True
    return report


def _table(reports: list[dict]) -> str:
    by_backend = {report["backend"]: report for report in reports}
    lines = [
        "# ANS acquisition acceptance",
        "",
        "Results cover the named API tests only. `not-covered` is not support; skipped or missing prerequisites block acceptance. Native UI, raw EMPAD2/G3 words, Velox events and performance signoff are outside this gate.",
        "",
        "| Workflow | CUDA | Python MPS/Metal | Native Swift/Metal |",
        "|---|---|---|---|",
    ]
    for name in GROUPS:
        states = [
            by_backend.get(backend, {}).get("cells", {}).get(name, "not-run")
            for backend in BACKENDS
        ]
        lines.append("| " + " | ".join([name, *states]) + " |")
    lines += ["", "| Runtime | Device tested | Date tested |", "|---|---|---|"]
    for report in reports:
        lines.append(
            f"| {report['backend']} | {report.get('device', {}).get('name', 'unavailable')} | {report['date'][:10]} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run one physical backend or combine matching-source reports.

    Examples
    --------
    >>> # python scripts/check_ans_io.py --backend mps --output /tmp/mps.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--backend", choices=BACKENDS)
    selection.add_argument("--combine", nargs="+", type=Path)
    parser.add_argument("--zenodo-root", type=Path)
    parser.add_argument("--tcmep-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tcmep_root is not None:
        os.environ["QEM_TCMEP_ROOT"] = str(args.tcmep_root.resolve())
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or any(
        output.with_suffix(suffix).exists() for suffix in (".json", ".md", ".log")
    ):
        parser.error(
            "Choose a new .json report path outside the repository; previous reports are never replaced."
        )
    if output.suffix != ".json":
        parser.error("Report path must end in .json.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.combine:
        reports = [json.loads(path.read_text()) for path in args.combine]
        if len({report["backend"] for report in reports}) != len(reports):
            parser.error("Supply one report per backend.")
        if len({report["source_fingerprint"] for report in reports}) != 1:
            parser.error(
                "Source fingerprints differ; rerun the same candidate on every backend."
            )
        content = dict(schema_version=1, reports=reports)
    else:
        content = _run(args.backend, output, args.zenodo_root)
        reports = [content]
    output.write_text(json.dumps(content, indent=2) + "\n")
    output.with_suffix(".md").write_text(_table(reports))
    print(_table(reports))
    print(f"Detailed results: {output}")
    return int(any(report["status"] != "passed" for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
