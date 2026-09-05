"""Qualified acquisition-to-deployment workflows without source mutation."""

from __future__ import annotations

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.cli import main
from quantem.gpu.remote import load_compact_browse_sources, prepare_browse_source

_fixture_spec = spec_from_file_location(
    "compact_fixture", Path(__file__).parents[2] / "contracts/io/test_compact_h5.py"
)
_fixture = module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)


def _sources(tmp_path):
    values = np.arange(64 * 4, dtype=np.uint16).reshape(64, 4)
    master = tmp_path / "sample_master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset("entry/data/data", data=values.reshape(64, 2, 2))
        specific = handle.create_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=64)
        specific.create_dataset("nimages", data=1)
        specific.create_dataset("x_pixels_in_detector", data=2)
        specific.create_dataset("y_pixels_in_detector", data=2)
    source = tmp_path / "packed.h5"
    _fixture._write_v3_fixture(source, values, detector_shape=(2, 2), scan_shape=(8, 8))
    return master, source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_prepare_qualified_source_and_reopen_the_sealed_registry(tmp_path):
    master, source, expected = _sources(tmp_path)
    master_before = master.read_bytes()
    destination = tmp_path / "deployment"

    registry = prepare_browse_source(
        master, source, destination, expected_source_sha256=expected
    )
    bindings = load_compact_browse_sources(registry, tmp_path)
    binding = bindings[master]
    row = json.loads(registry.read_text())["sources"][0]
    integrity = io.SourceIntegrity.from_file(
        destination / "source.integrity.json",
        expected_sha256=row["expected_chunk_integrity_manifest_sha256"],
    )

    assert binding.path == source
    assert binding.source_integrity == integrity
    assert integrity.whole_file_sha256 == expected
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    assert master.read_bytes() == master_before
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_browse_source(
            master, source, destination, expected_source_sha256=expected
        )
    assert load_compact_browse_sources(registry, tmp_path) == bindings
    alias = tmp_path / "data-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    assert load_compact_browse_sources(registry, alias) == bindings


def test_damaged_source_or_manifest_cannot_be_admitted(tmp_path):
    master, source, expected = _sources(tmp_path)
    destination = tmp_path / "deployment"
    with pytest.raises(ValueError, match="Source SHA-256"):
        prepare_browse_source(
            master, source, destination, expected_source_sha256="a" * 64
        )
    assert not destination.exists()
    registry = prepare_browse_source(
        master, source, destination, expected_source_sha256=expected
    )
    manifest = destination / "source.integrity.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_compact_browse_sources(registry, tmp_path)


def test_cli_preparation_uses_the_same_source_preserving_workflow(tmp_path, capsys):
    master, source, expected = _sources(tmp_path)
    destination = tmp_path / "deployment"
    assert (
        main(
            [
                "prepare-browse",
                str(master),
                str(source),
                str(destination),
                "--expected-source-sha256",
                expected,
            ]
        )
        == 0
    )
    registry = destination / "sources.json"
    assert capsys.readouterr().out.strip() == str(registry)
    assert (
        load_compact_browse_sources(registry, tmp_path)[master].source_integrity
        is not None
    )
