from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _benchmark_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "quantem_gpu_test_benchmark_screening",
        scripts / "benchmark_screening.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_result(*, exact_dtype=np.uint64):
    scan = np.arange(6, dtype=np.float32).reshape(2, 3)
    exact = np.arange(6, dtype=exact_dtype).reshape(2, 3)
    return SimpleNamespace(
        mean_dp=np.arange(4, dtype=np.float32).reshape(2, 2),
        total_intensity=exact,
        bright_field=scan,
        annular_bright_field=exact,
        annular_dark_field=exact,
        dark_field=scan,
        dpc_phase=scan,
        com_row=scan,
        com_col=scan,
    )


def test_screening_benchmark_hashes_full_suite_with_dtype_and_shape(
    monkeypatch,
) -> None:
    benchmark = _benchmark_module(monkeypatch)

    arrays = benchmark._product_arrays(_full_result())
    hashes = benchmark._product_hashes(arrays)
    specs = benchmark._product_specs(arrays)

    assert tuple(hashes) == benchmark.FULL_PRODUCT_FIELDS
    assert set(hashes) == set(benchmark.FULL_PRODUCT_FIELDS)
    assert len(set(hashes.values())) > 1
    for name in benchmark.EXACT_INTEGER_PRODUCT_FIELDS:
        assert specs[name] == {
            "shape": [2, 3],
            "dtype": "uint64",
            "nbytes": 48,
        }


def test_screening_benchmark_fails_closed_on_missing_or_inexact_product(
    monkeypatch,
) -> None:
    benchmark = _benchmark_module(monkeypatch)
    missing = _full_result()
    missing.annular_dark_field = None

    with pytest.raises(RuntimeError, match="missing annular_dark_field"):
        benchmark._product_arrays(missing)
    with pytest.raises(RuntimeError, match="must preserve exact uint64"):
        benchmark._product_arrays(_full_result(exact_dtype=np.float32))


def test_screening_benchmark_keeps_historical_suite_compatible(monkeypatch) -> None:
    benchmark = _benchmark_module(monkeypatch)
    result = _full_result()
    del result.total_intensity
    del result.annular_bright_field
    del result.annular_dark_field

    arrays = benchmark._product_arrays(result, benchmark.PRODUCT_FIELDS)

    assert tuple(arrays) == benchmark.PRODUCT_FIELDS


def test_screening_benchmark_reference_requires_full_suite(
    monkeypatch,
    tmp_path,
) -> None:
    benchmark = _benchmark_module(monkeypatch)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {name: "0" * 64 for name in benchmark.FULL_PRODUCT_FIELDS[:-1]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="com_col"):
        benchmark._reference_hashes(
            reference_path,
            benchmark.FULL_PRODUCT_FIELDS,
        )
