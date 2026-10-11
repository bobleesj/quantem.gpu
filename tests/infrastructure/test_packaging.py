import re
import tomllib
from pathlib import Path

EXTRAS = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def _distributions(extra: str) -> set[str]:
    return {re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].lower() for requirement in EXTRAS[extra]}


def test_lazily_imported_packages_are_declared_in_an_extra() -> None:
    # SSBResult.show, SSB.show_trials and the rotation histograms import these when called.
    assert {"matplotlib", "quantem"} <= _distributions("plot")
    # movie.cuda writes MP4 through NVENC.
    assert "pynvvideocodec" in _distributions("movie")
    # The NVENC movie tests decode the written files with cv2.
    assert "opencv-python-headless" in _distributions("dev")
