"""Freeze the shader extraction independently of Windows hardware execution."""

import hashlib
from pathlib import Path
import re


def test_direct3d_fft_matches_frozen_windows_shaders() -> None:
    source = Path("src/quantem/gpu/display/backends/direct3d/ImageFft.cs").read_text()
    # Captured from Windows cd780db before extraction, not from this backend.
    expected = {
        "InitializeShader": "2d98b40b09d3f4624920a7b34108581a6c988c439e5eaf4e3a430e2228baa4e4",
        "BitReverseColumnsShader": "1d3a8438dded090f1a753d62d3cb8b57fdf975d601a85980d0b6cb3e54f65ce2",
        "ButterflyShader": "153e659195912b601baf0ddab60b4ebb9a19a702ccb0b491575b8ea8f15126d1",
        "MagnitudeShader": "2c0499d4812611d9483dec4e6d900028564467124f01603ab324000961c3d04a",
    }
    shaders = re.findall(
        r'private const string (\w+) = """(.*?)""";', source, re.S
    )
    assert {name: hashlib.sha256(body.encode()).hexdigest() for name, body in shaders} == expected
