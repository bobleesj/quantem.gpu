"""Backend-neutral 4D-STEM in-memory representation contract."""

from __future__ import annotations

from enum import Enum
from os import PathLike
from pathlib import Path

__all__ = ["DataRepresentation"]

_LOSSLESS_PACK_CONTAINER_MAGIC = b"QGPUH5\0\x01"
_PAIRED_RESIDENT_MAGIC = b"QGPUPAIR"


class DataRepresentation(str, Enum):
    """Public representation of the complete logical 4D-STEM array.

    ``PACKED`` preserves every declared working value while retaining
    compact count storage for accelerator kernels. ``DENSE`` stores one
    unpacked value for every logical array element. No representation changes
    scan coverage, detector coverage, binning, calibration, or scientific dtype.

    ``ANS`` names an exact entropy-coded resident source. Disk encoding is
    independent: an ANS file can be transcoded into packed storage. The
    authenticated storage schema selects the precise decoder within a
    representation. ``PAIRED`` names the opt-in CUDA paired-count tANS resident
    layout with its polar interaction index; original HDF5 selects it only
    explicitly, while a saved paired resident form is detected from its magic.
    These names do not guarantee that every conversion or backend is
    implemented.

    Examples
    --------
    >>> DataRepresentation.parse("packed")
    <DataRepresentation.PACKED: 'packed'>
    >>> DataRepresentation.parse(DataRepresentation.DENSE)
    <DataRepresentation.DENSE: 'dense'>
    """

    DENSE = "dense"
    PACKED = "packed"
    ANS = "ans"
    PAIRED = "paired"

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
        as ``PACKED``. Standalone QuantEM/ANS files select ``ANS``
        from their magic regardless of extension, and saved paired resident
        forms select ``PAIRED`` the same way. This inspection reads only
        eight bytes; it does not validate the complete file.

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
        if magic == b"QGANS\0\1\0":
            return cls.ANS
        if magic == _PAIRED_RESIDENT_MAGIC:
            return cls.PAIRED
        return (
            cls.PACKED
            if magic == _LOSSLESS_PACK_CONTAINER_MAGIC
            else cls.DENSE
        )
