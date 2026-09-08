"""Compile the exact compact-mask planner without a CUDA dependency.

The bundled C++ source is unchanged from ``native_planner167.cpp`` in the
validated native detector planner experiment. Its SHA-256 is recorded beside
the source in the package provenance. Linux, a C++17 compiler (``CXX`` or
``c++``), and a writable ``XDG_CACHE_HOME`` (default ``~/.cache``) are required.
Compilation happens once per source, compiler version, and machine architecture.
It uses an argument list, an interprocess file lock, and an atomic installation;
no compiler or scientific work runs merely by importing this module.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Self

import numpy as np


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    """Compile once into the user's source-addressed cache, then load it."""
    import fcntl

    source = Path(__file__).with_name("kernels") / "planner.cpp"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    executable = shutil.which(compiler[0]) if compiler else None
    if executable is None:
        raise RuntimeError(
            "The compact detector planner needs a C++17 compiler. Install "
            "g++ or clang++, or set CXX to its executable; "
            f"got CXX={os.environ.get('CXX', 'c++')!r}."
        )
    compiler[0] = executable
    try:
        version = subprocess.run(
            [*compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Cannot identify the C++ compiler {compiler!r}; check CXX."
        ) from error
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"]
    identity = json.dumps(
        [compiler, version, flags, platform.system(), platform.machine()]
    ).encode()
    digest = hashlib.sha256(source.read_bytes() + identity).hexdigest()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache / "quantem" / "compact-planner" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "planner.so"
    with (cache / "compile.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / "planner.so"
                try:
                    subprocess.run(
                        [*compiler, *flags, str(source), "-o", str(output)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        "The compact detector planner could not compile. "
                        "Check that CXX supports C++17 and shared libraries.\n"
                        f"{error.stderr}"
                    ) from error
                output.replace(target)
    library = ctypes.CDLL(str(target))
    pointer = ctypes.c_void_p
    library.plan167.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_double,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.plan167.restype = ctypes.c_int
    return library


def _array(
    value: np.ndarray, name: str, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Check pointer layout before entering a fixed-layout native function."""
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(dtype)
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            f"{name} must be a C-contiguous {dtype} array of shape {shape}; "
            f"got shape {getattr(value, 'shape', None)} and "
            f"dtype {getattr(value, 'dtype', None)}. "
            "Load metadata for the supported 192 by 192 compact format."
        )
    return value


class NativePlanner:
    """Choose exact residual and tile coefficients for one detector mask.

    Instances own small reusable host outputs. Call from one serial worker;
    a returned array remains valid only until the next call on that instance.
    The planner never transfers scientific data or initializes CUDA.
    """

    @classmethod
    def from_metadata(
        cls,
        *,
        valid: np.ndarray,
        leaf_of: np.ndarray,
        column_cost: np.ndarray,
        tile_cost: float,
        parents: np.ndarray,
        omitted: np.ndarray,
        stored_positions: np.ndarray,
    ) -> Self:
        """Prepare the fixed 192 by 192 planner from encoded-format metadata.

        Metadata remains owned by the caller and is borrowed for the planner's
        lifetime. Keep these arrays immutable while the planner is in use.

        Parameters
        ----------
        valid, leaf_of, column_cost : numpy.ndarray
            Flattened validity, fine-leaf assignment, and decode cost arrays,
            with dtypes int8, int64, and float64, respectively.
        tile_cost : float
            Cost of reading one stored interaction tile.
        parents, omitted, stored_positions : numpy.ndarray
            int32 arrays encoding the 152 split parents and their four children.

        Returns
        -------
        NativePlanner
            Serial planner with reusable residual and coefficient arrays.

        Examples
        --------
        >>> planner = NativePlanner.from_metadata(**metadata)
        >>> seed, residual, fine, coarse, cost = planner(mask, previous_mask)
        """
        self = cls()
        layouts = (
            ("valid", valid, "int8", (36864,)),
            ("leaf_of", leaf_of, "int64", (36864,)),
            ("column_cost", column_cost, "float64", (36864,)),
            ("parents", parents, "int32", (152,)),
            ("omitted", omitted, "int32", (152,)),
            ("stored_positions", stored_positions, "int32", (152, 3)),
        )
        for name, value, dtype, shape in layouts:
            setattr(self, name, _array(value, name, dtype, shape))
        if (
            np.any((valid < 0) | (valid > 1))
            or np.any((leaf_of < 0) | (leaf_of >= 1184))
            or np.any((parents < 0) | (parents >= 576))
            or np.any((omitted < 0) | (omitted >= 4))
            or np.any((stored_positions < 0) | (stored_positions >= 4))
        ):
            raise ValueError(
                "Compact planner metadata has invalid validity or tree indices; "
                "load a complete supported 192 by 192 encoded dataset."
            )
        self.tile_cost = float(tile_cost)
        self.library = _library()
        self.outputs = tuple(np.empty(size, np.int8) for size in (36864, 1032, 36))
        return self

    def __call__(
        self, current: np.ndarray, previous: np.ndarray | None = None
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
        """Plan a valid binary mask using zero, total, or the previous image.

        Parameters
        ----------
        current, previous : numpy.ndarray or None
            C-contiguous int8 masks of shape (36864,). Invalid detector pixels
            must already be zero. ``previous=None`` forces an independent sum.

        Returns
        -------
        tuple
            Seed name, residual pixel coefficients, fine tile coefficients,
            coarse tile coefficients, and estimated cost. Arrays alias this
            planner's reusable host buffers and are overwritten on its next call.

        Examples
        --------
        >>> seed, residual, fine, coarse, cost = planner(mask)
        >>> saved_residual = residual.copy()
        """
        _array(current, "current", "int8", (36864,))
        if previous is not None:
            _array(previous, "previous", "int8", (36864,))
        cost = ctypes.c_double()
        seed = self.library.plan167(
            current.ctypes.data,
            None if previous is None else previous.ctypes.data,
            self.valid.ctypes.data,
            self.leaf_of.ctypes.data,
            1184,
            self.column_cost.ctypes.data,
            self.tile_cost,
            self.parents.ctypes.data,
            self.omitted.ctypes.data,
            self.stored_positions.ctypes.data,
            152,
            *(output.ctypes.data for output in self.outputs),
            ctypes.byref(cost),
        )
        if seed not in (0, 1, 2):
            raise RuntimeError(
                "The compact detector planner rejected its coefficient bounds "
                f"(code {seed}); provide binary masks with invalid pixels zero."
            )
        return ("zero", "total", "previous")[seed], *self.outputs, cost.value
