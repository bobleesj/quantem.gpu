"""The organized suite retains a lossless map of earlier test entry points."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def test_all_migrated_test_paths_exist():
    mapping = json.loads((ROOT / "tests/path_migrations.json").read_text())
    assert len(set(mapping.values())) == len(mapping)
    for old, new in mapping.items():
        assert old.startswith("tests/") and new.startswith("tests/")
        assert (ROOT / new).is_file(), (old, new)


def test_runner_translates_node_ids_without_changing_pytest_options(monkeypatch):
    spec = spec_from_file_location("qgpu_test_runner", ROOT / "scripts/run_tests.py")
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["tests/io/test_load.py::test_name", "-q"]) == 7
    assert calls[0][0][-2:] == [
        "tests/contracts/io/test_load.py::test_name", "-q"
    ]
    assert calls[0][1]["cwd"] == ROOT
