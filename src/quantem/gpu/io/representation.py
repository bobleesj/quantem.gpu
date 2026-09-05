"""Backend-neutral 4D-STEM in-memory representation contract."""

from __future__ import annotations

from enum import Enum
from os import PathLike
from pathlib import Path

__all__ = ["DataRepresentation"]

_LOSSLESS_PACK_CONTAINER_MAGIC = b"QGPUH5\0\x01"


class DataRepresentation(str, Enum):
    """Public representation of the complete logical 4D-STEM array.

    ``LOSSLESS_PACKED`` preserves every declared working value while retaining
    a compact representation for accelerator kernels. ``DENSE`` stores one
    unpacked value for every logical array element. Neither value changes scan
    coverage, detector coverage, binning, calibration, or scientific dtype.

    Examples
    --------
    >>> DataRepresentation.parse("lossless_packed")
    <DataRepresentation.LOSSLESS_PACKED: 'lossless_packed'>
    >>> DataRepresentation.parse(DataRepresentation.DENSE)
    <DataRepresentation.DENSE: 'dense'>
    """

    LOSSLESS_PACKED = "lossless_packed"
    DENSE = "dense"

    @classmethod
    def parse(cls, value: DataRepresentation | str) -> DataRepresentation:
        """Return the canonical representation selected by a public caller.

        Parameters
        ----------
        value
            Canonical representation enum or string.

        Returns
        -------
        DataRepresentation
            Parsed representation.

        Raises
        ------
        ValueError
            If ``value`` does not name a supported representation.

        Examples
        --------
        >>> DataRepresentation.parse("dense") is DataRepresentation.DENSE
        True
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(repr(item.value) for item in cls)
            raise ValueError(
                f"representation must be one of {choices}; got {value!r}."
            ) from error

    @classmethod
    def detect_source(
        cls, source: str | PathLike[str]
    ) -> DataRepresentation:
        """Identify the representation encoded by one source container.

        Ordinary HDF5 is dense-compatible source evidence. A QuantEM lossless
        pack container carries a fixed user-block magic and is loaded directly
        as ``LOSSLESS_PACKED``. This inspection reads only eight bytes.

        Parameters
        ----------
        source
            Source path to inspect.

        Returns
        -------
        DataRepresentation
            Source representation available without transcoding.

        Examples
        --------
        >>> DataRepresentation.detect_source("ordinary-master.h5")
        <DataRepresentation.DENSE: 'dense'>
        """
        path = Path(source)
        try:
            with path.open("rb") as stream:
                magic = stream.read(len(_LOSSLESS_PACK_CONTAINER_MAGIC))
        except OSError:
            return cls.DENSE
        return (
            cls.LOSSLESS_PACKED
            if magic == _LOSSLESS_PACK_CONTAINER_MAGIC
            else cls.DENSE
        )
