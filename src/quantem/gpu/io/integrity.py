"""Explicit source seals for authenticated prepared-data loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_CHUNK_BYTES = 16 << 20
_SCHEMA = "quantem.gpu.compact-chunk-integrity/v1"


def _require_sha256(value: str, name: str) -> None:
    """Validate an external identity at the trust boundary."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{name} must be a lowercase SHA-256 digest; got {value!r}. "
            "Use the digest recorded when the source was qualified."
        )


@dataclass(frozen=True)
class SourceIntegrity:
    """Immutable source identity and independently authenticated byte ranges.

    A manifest is trusted only through a separately supplied SHA-256 seal.
    Adjacent files are never discovered or trusted automatically. Chunk hashes
    permit concurrent verification without changing the scientific source.

    Parameters
    ----------
    whole_file_sha256
        SHA-256 of the complete qualified source container.
    file_bytes
        Exact byte length of that container.
    chunk_bytes
        Byte length of consecutive ranges, except the final remainder.
    chunk_sha256
        Ordered SHA-256 digests covering the complete container.

    Examples
    --------
    >>> seal = SourceIntegrity.from_file(
    ...     "source.integrity.json", expected_sha256=qualified_manifest_sha256
    ... )
    >>> loaded = io.load("source.h5", source_integrity=seal)
    """

    whole_file_sha256: str
    file_bytes: int
    chunk_bytes: int
    chunk_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.whole_file_sha256, "whole_file_sha256")
        if type(self.file_bytes) is not int or self.file_bytes <= 0:
            raise ValueError("file_bytes must be the positive source file size.")
        if type(self.chunk_bytes) is not int or self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive byte count.")
        object.__setattr__(self, "chunk_sha256", tuple(self.chunk_sha256))
        expected_count = (self.file_bytes + self.chunk_bytes - 1) // self.chunk_bytes
        if len(self.chunk_sha256) != expected_count:
            raise ValueError(
                f"Integrity manifest has {len(self.chunk_sha256)} ranges; "
                f"expected {expected_count} to cover {self.file_bytes} bytes."
            )
        for digest in self.chunk_sha256:
            _require_sha256(digest, "chunk_sha256")

    @classmethod
    def from_file(cls, path: str | Path, *, expected_sha256: str) -> SourceIntegrity:
        """Read an integrity manifest only after verifying its external seal.

        Parameters
        ----------
        path
            Manifest generated during source preparation.
        expected_sha256
            Independently recorded SHA-256 of the manifest, not the dataset.

        Returns
        -------
        SourceIntegrity
            Validated immutable identity passed to ``io.load``.

        Raises
        ------
        ValueError
            If the manifest differs from its seal or cannot cover the source.

        Examples
        --------
        >>> seal = SourceIntegrity.from_file(
        ...     "source.integrity.json", expected_sha256=manifest_sha256
        ... )
        """
        _require_sha256(expected_sha256, "expected_sha256")
        payload = Path(path).read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected_sha256:
            raise ValueError(
                f"Integrity manifest SHA-256 is {observed}, expected "
                f"{expected_sha256}. Restore the qualified manifest."
            )
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("schema") != _SCHEMA:
            raise ValueError(
                "Unsupported source integrity schema; prepare a v1 manifest."
            )
        try:
            return cls(
                whole_file_sha256=value["whole_file_sha256"],
                file_bytes=value["file_bytes"],
                chunk_bytes=value["chunk_bytes"],
                chunk_sha256=tuple(value["chunk_sha256"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Incomplete source integrity manifest; prepare it again."
            ) from error

    def validate_source(self, path: str | Path) -> None:
        """Check source size before allocation, without authenticating content.

        Parameters
        ----------
        path
            Qualified packed source to compare with the recorded byte length.

        Raises
        ------
        ValueError
            If the file size differs. A matching size alone does not establish
            integrity; ``io.load`` must still verify all source bytes.

        Examples
        --------
        >>> seal.validate_source("source.h5")
        """
        observed = Path(path).stat().st_size
        if observed != self.file_bytes:
            raise ValueError(
                f"Source has {observed} bytes, expected {self.file_bytes}. "
                "Restore the qualified source before loading."
            )


def _seal_source(path: Path, expected_sha256: str) -> SourceIntegrity:
    """Verify and index a complete source in one bounded-memory read."""
    _require_sha256(expected_sha256, "expected_source_sha256")
    whole = hashlib.sha256()
    chunks: list[str] = []
    file_bytes = 0
    with path.open("rb") as stream:
        while block := stream.read(_CHUNK_BYTES):
            file_bytes += len(block)
            whole.update(block)
            chunks.append(hashlib.sha256(block).hexdigest())
    observed = whole.hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"Source SHA-256 is {observed}, expected {expected_sha256}. "
            "Do not admit this file; restore or independently qualify the source."
        )
    return SourceIntegrity(observed, file_bytes, _CHUNK_BYTES, tuple(chunks))


def _manifest_bytes(seal: SourceIntegrity) -> bytes:
    """Serialize the versioned preparation artifact deterministically."""
    return (
        json.dumps(
            {
                "schema": _SCHEMA,
                "whole_file_sha256": seal.whole_file_sha256,
                "file_bytes": seal.file_bytes,
                "chunk_bytes": seal.chunk_bytes,
                "chunk_sha256": seal.chunk_sha256,
            },
            indent=2,
        )
        + "\n"
    ).encode()
