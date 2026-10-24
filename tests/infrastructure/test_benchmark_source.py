"""Timing evidence must reject package imports from a stale checkout."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_source


def test_runtime_source_records_the_actual_imported_module():
    import quantem.gpu

    evidence = benchmark_source.runtime_source()
    module = evidence["modules"]["quantem.gpu"]
    path = Path(quantem.gpu.__file__).resolve()
    assert evidence["matched_checkout"]
    assert module["path"] == path.relative_to(benchmark_source.ROOT).as_posix()
    assert module["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_source_rejects_a_stale_submodule(tmp_path, monkeypatch):
    stale = tmp_path / "stale.py"
    stale.write_text("# Different checkout\n")
    monkeypatch.setitem(
        sys.modules, "quantem.gpu.stale_benchmark_fixture",
        SimpleNamespace(__file__=str(stale)),
    )
    with pytest.raises(RuntimeError, match="different checkout.*Set PYTHONPATH"):
        benchmark_source.runtime_source()
