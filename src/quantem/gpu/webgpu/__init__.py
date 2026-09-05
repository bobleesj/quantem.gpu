"""Export the complete browser source graph for consumer builds.

This is a build-time resource boundary, not a Python GPU execution backend.
"""

from importlib.resources import files
import json
from pathlib import Path


def source_names() -> tuple[str, ...]:
    """Return packaged browser resource paths relative to ``quantem.gpu``.

    Returns
    -------
    tuple[str, ...]
        Canonical sources, compatibility exports, and required JSON resources.

    Examples
    --------
    >>> "webgpu/index.ts" in source_names()
    True
    """
    return tuple(json.loads(files(__package__).joinpath("sources.json").read_text()))


def source_text(name: str) -> str:
    """Read one declared browser build resource without initializing a GPU.

    Parameters
    ----------
    name : str
        A package-relative resource returned by :func:`source_names`.

    Returns
    -------
    str
        UTF-8 TypeScript or JSON source.

    Raises
    ------
    ValueError
        If the name is not a declared resource.

    Examples
    --------
    >>> "export" in source_text("webgpu/index.ts")
    True
    """
    if name not in source_names():
        raise ValueError(f"Unknown browser resource {name!r}; choose from source_names().")
    return files("quantem.gpu").joinpath(name).read_text(encoding="utf-8")


def export_sources(directory: str | Path) -> Path:
    """Write the complete graph to an empty consumer-owned build directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Empty or nonexistent destination. Existing source trees are never erased.

    Returns
    -------
    pathlib.Path
        Destination containing the same relative paths as :func:`source_names`.

    Raises
    ------
    FileExistsError
        If the destination contains files from another build.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as temporary:
    ...     exported = export_sources(temporary)
    ...     assert (exported / "webgpu/index.ts").is_file()
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"Browser export directory {str(destination)!r} is not empty; "
            "choose a fresh generated directory."
        )
    for name in source_names():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_text(name), encoding="utf-8")
    return destination


__all__ = ["export_sources", "source_names", "source_text"]
