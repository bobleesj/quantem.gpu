"""Read portable saved phases without loading or reconstructing a 4D volume."""

import base64
import json
import threading
from collections import OrderedDict
from pathlib import Path

import h5py

from ..io.ssb_result import acquisition_identity, read_result


class SavedSSBResults:
    """Serve acquisition-verified phases, caching only verified file identities.

    Parameters
    ----------
    root : Path
        Data directory containing acquisitions and sibling screening results.

    Examples
    --------
    >>> saved = SavedSSBResults(Path("/data"))
    >>> results = saved.read(Path("/data/sample_master.h5"))
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._identities = OrderedDict()
        self._lock = threading.Lock()

    def _identity(self, master: Path) -> str:
        with h5py.File(master, "r") as handle:
            group = handle["entry/data"]
            paths = [master]
            for name in sorted(group):
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink):
                    raise ValueError(
                        "Saved SSB discovery requires an ARINA acquisition."
                    )
                member = (master.parent / link.filename).resolve()
                member.relative_to(self.root)
                paths.append(member)

        def signature():
            return tuple(
                (
                    str(path),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
                for path in paths
                for stat in [path.stat()]
            )

        key = signature()
        with self._lock:
            if key in self._identities:
                self._identities.move_to_end(key)
                return self._identities[key]
            identity = acquisition_identity(master)
            if signature() != key:
                raise ValueError(
                    "Acquisition changed during verification. Retry after acquisition completes."
                )
            self._identities[key] = identity
            while len(self._identities) > 64:
                self._identities.popitem(last=False)
            return identity

    def read(self, master: Path) -> dict[str, object]:
        """Return matching calibrated phases for a completed acquisition.

        Parameters
        ----------
        master : Path
            ARINA HDF5 master with external data members inside ``root``.

        Returns
        -------
        dict[str, object]
            Versioned response with source identity and validated phase records.
            Phase bytes and the original run metadata are base64 encoded.

        Raises
        ------
        ValueError
            If source or result paths escape the data root, source bytes change
            during verification, or a result exceeds the supported transfer size.

        Examples
        --------
        >>> results = SavedSSBResults(Path('/data')).read(Path('/data/sample_master.h5'))
        """
        master = master.resolve()
        master.relative_to(self.root)
        folder = master.parent / "live" / "screen"
        folder.resolve().relative_to(self.root)
        candidates = []
        if folder.is_dir():
            for child in sorted(folder.iterdir()):
                manifest = child / "ssb.json"
                if child.is_symlink() or not child.is_dir() or not manifest.is_file():
                    continue
                manifest.resolve().relative_to(self.root)
                if manifest.stat().st_size > 32 << 20:
                    raise ValueError("Saved SSB manifest exceeds 32 MB.")
                candidates.append((manifest, json.loads(manifest.read_text())))
        if not candidates:
            return {"schemaVersion": 1, "sourceIdentity": None, "results": []}
        identity = self._identity(master)
        results = []
        total_bytes = 0
        for manifest, hint in candidates:
            if hint.get("sourceIdentity") != identity:
                continue
            record, phase = read_result(manifest, identity)
            # Transport the validated artifact directly; neither G_k nor raw data.
            record = dict(record)
            record.pop("phaseFile", None)
            record.pop("format", None)
            record["phase"] = base64.b64encode(phase.tobytes()).decode("ascii")
            record["runMetadata"] = base64.b64encode(
                json.dumps(record["runMetadata"], sort_keys=True).encode()
            ).decode("ascii")
            total_bytes += len(json.dumps(record))
            if total_bytes > 32 << 20:
                raise ValueError(
                    "Saved SSB results exceed the 32 MB transfer limit. Import individual result pairs."
                )
            results.append(record)
        return {"schemaVersion": 1, "sourceIdentity": identity, "results": results}
