"""Publish complete acquisition files without replacing existing user data."""

import ctypes
import errno
import os
import sys
from pathlib import Path


def publish_file(temporary: str | Path, destination: str | Path) -> None:
    """Publish a complete sibling file without exposing a partial copy."""
    try:
        os.link(temporary, destination)
    except OSError as error:
        if sys.platform != "darwin" or error.errno not in {
            errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM
        }:
            raise
        # exFAT cannot hard-link. RENAME_EXCL atomically publishes the complete
        # sibling file and refuses replacement if another writer won the race.
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename = library.renamex_np
        except AttributeError as missing:
            raise OSError(
                errno.ENOTSUP,
                "This volume cannot publish .qem files without exposing partial data; "
                "save to a volume with atomic rename support.",
                os.fspath(destination),
            ) from missing
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        rename_exclusive = 4  # RENAME_EXCL from macOS sys/stdio.h
        if rename(
            os.fsencode(temporary), os.fsencode(destination), rename_exclusive
        ) == 0:
            return
        code = ctypes.get_errno()
        if code in {errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise OSError(
                code,
                "This volume cannot publish .qem files without exposing partial data; "
                "save to a volume with atomic rename support.",
                os.fspath(destination),
            )
        raise OSError(code, os.strerror(code), os.fspath(destination))
