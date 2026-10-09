"""Guard what this public repository may track.

Raw experiment and benchmark logs live in the private quantem.gpu-experiments
archive; dated summaries go under docs/. Datasets stay out except the small
synthetic and native test fixtures, and no tracked text may name a private
computer, person, partner, dataset or home path. See AGENTS.md.
"""

import json
import subprocess
from pathlib import Path

from tests.infrastructure.private_names import mentions_private_name

MAX_BYTES = 5 * 1024 * 1024
VENDORED_HDF5 = "native/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a"
DATA_SUFFIXES = (
    ".jsonl",
    ".log",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".h5",
    ".hdf5",
    ".emd",
    ".dm3",
    ".dm4",
    ".mrc",
    ".ser",
    ".npy",
    ".npz",
    ".qem",
)
FIXTURE_DIRECTORIES = ("tests/data/",)
# h5py's global lock object shares its name with a private host; only these exact lines may use it.
H5PY_LOCK = "ph" + "il"  # built from parts so this guard does not trip on itself
H5PY_LOCK_LINES = {f"from h5py._objects import {H5PY_LOCK}", f"with {H5PY_LOCK}:"}


def _tracked_files() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True
    ).stdout.decode()
    return [path for path in listing.split("\0") if path and Path(path).is_file()]


def _is_fixture(path: str) -> bool:
    native_fixture = path.startswith("native/swift/Tests/") and "/Fixtures/" in path
    return path.startswith(FIXTURE_DIRECTORIES) or native_fixture


def _public_text(path: str) -> str | None:
    """Tracked text without binary files, notebook images or the h5py lock lines.

    Base64 image outputs are random letters that would match short names by
    chance, so notebooks are checked on their sources and text outputs only.
    """
    data = Path(path).read_bytes()
    if b"\0" in data[:8192]:
        return None
    text = data.decode("utf-8", errors="ignore")
    if path.endswith(".ipynb"):
        notebook = json.loads(text)
        parts = []
        for cell in notebook["cells"]:
            parts.append("".join(cell["source"]))
            for output in cell.get("outputs", []):
                parts.append("".join(output.get("text", "")))
                for mime, value in output.get("data", {}).items():
                    if not mime.startswith("image/"):
                        parts.append("".join(value) if isinstance(value, list) else str(value))
        text = "\n".join(parts)
    return "\n".join(line for line in text.splitlines() if line.strip() not in H5PY_LOCK_LINES)


def test_experiment_evidence_is_not_tracked() -> None:
    assert [path for path in _tracked_files() if path.startswith("experiments/")] == []


def test_no_large_files_except_the_vendored_hdf5_library() -> None:
    large = [
        path
        for path in _tracked_files()
        if path != VENDORED_HDF5 and Path(path).stat().st_size > MAX_BYTES
    ]

    assert large == []


def test_data_and_log_files_stay_in_fixture_directories() -> None:
    misplaced = [
        path
        for path in _tracked_files()
        if path.lower().endswith(DATA_SUFFIXES)
        and not _is_fixture(path)
        and not path.startswith("native/swift/Vendor/CHDF5.xcframework/")
    ]

    assert misplaced == []


def test_no_macos_metadata_files() -> None:
    junk = [
        path
        for path in _tracked_files()
        if Path(path).name.startswith("._") or Path(path).name == ".DS_Store"
    ]

    assert junk == []


def test_tracked_text_names_no_private_computer_person_or_path() -> None:
    leaked = []
    for path in _tracked_files():
        text = _public_text(path)
        if text is None or not mentions_private_name(text):
            continue
        lines = [number for number, line in enumerate(text.splitlines(), 1) if mentions_private_name(line)]
        leaked.append(f"{path}:{lines or 'adjacent lines'}")

    assert leaked == []
