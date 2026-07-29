"""MPS Metal buffers must be returned to the system when a load is dropped.

PyObjC does not release buffers created by ``newBufferWithLength_options_`` when
the Python wrapper is collected, so without an explicit release every
``load(backend="mps")`` permanently retains its output. Seven no-bin tilts then
exhaust a 128 GB Mac even though only one tilt is ever meant to be resident.
"""

import gc
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="needs an Apple GPU",
)


def _allocated_bytes() -> int:
    return int(torch.mps.driver_allocated_memory())


def test_owner_releases_buffer_when_dropped():
    """An owned buffer returns its memory once the owner goes away.

    ``newBufferWithLength_options_`` hands back a +1-retained object on top of
    the retain PyObjC takes for its wrapper, so the memory comes back only when
    both are undone: the explicit release, then the dropped Python reference.
    Dropping the reference alone is what leaked ~45 GB per tilt load.
    """
    from quantem.gpu.io.backends.mps import decoder as be

    nbytes = 1 << 30
    baseline = _allocated_bytes()
    owner = be._MtlOwner(be._metal_buffer_alloc(nbytes))
    assert _allocated_bytes() - baseline >= nbytes // 2
    del owner
    gc.collect()
    assert _allocated_bytes() - baseline < nbytes // 2


def test_release_is_idempotent():
    """A second release must be a no-op, not an over-release.

    Buffers are reachable from a returned array and a scratch pool at the same
    time, so a double release would be a use-after-free rather than a leak.
    """
    from quantem.gpu.io.backends.mps import decoder as be

    owner = be._MtlOwner(be._metal_buffer_alloc(1 << 20))
    owner.release()
    owner.release()
    be._release_metal_buffer(None)


def test_mtl_array_view_keeps_buffer_alive():
    """Slices must not outlive the buffer they read from.

    ``_MtlArray`` had no ``__array_finalize__``, so a view silently lost ``_mtl``.
    That was invisible while every buffer leaked; once release works it is a
    use-after-free, so views must carry the owner.
    """
    from quantem.gpu.io.backends.mps import decoder as be

    buf = be._metal_buffer_alloc(4096)
    arr = be._mtl_array_from_buffer(buf, np.uint16, (32, 32))
    view = arr[:8]
    assert view._mtl is arr._mtl, "view dropped its Metal buffer owner"
    assert arr.reshape(16, 64)._mtl is arr._mtl


def test_compressed_buffer_bound_uses_file_metadata(tmp_path):
    """Sizing compressed input must not pre-scan every HDF5 chunk layout."""
    from quantem.gpu.io.backends.mps import decoder as be

    small = tmp_path / "small.h5"
    large = tmp_path / "large.h5"
    small.write_bytes(b"0")
    large.write_bytes(b"0" * 1024)
    plan = SimpleNamespace(chunk_files=(str(small), str(large)))

    assert be._max_compressed_bytes_for_plan(plan) == 150 * 1024 * 1024

    os.truncate(large, 200 * 1024 * 1024)
    assert be._max_compressed_bytes_for_plan(plan) == 201 * 1024 * 1024


MAPED_TEST_DIR = os.environ.get("MAPED_TEST_DIR", "")


@pytest.mark.skipif(not os.path.isdir(MAPED_TEST_DIR), reason="needs MAPED_TEST_DIR")
def test_repeated_load_does_not_accumulate():
    """Loading tilts one at a time must not grow memory without bound."""
    import glob

    from quantem.gpu.io import load

    masters = sorted(glob.glob(os.path.join(MAPED_TEST_DIR, "*_master.h5")))[:3]
    if len(masters) < 2:
        pytest.skip("needs at least 2 masters")

    result = load(masters[0], det_bin=1, backend="mps", verbose=False)
    one_tilt = _allocated_bytes()
    del result
    gc.collect()
    for path in masters[1:]:
        result = load(path, det_bin=1, backend="mps", verbose=False)
        del result
        gc.collect()
    assert _allocated_bytes() <= one_tilt * 1.5, (
        f"memory grew from {one_tilt / 1e9:.1f} GB to "
        f"{_allocated_bytes() / 1e9:.1f} GB across {len(masters)} sequential loads"
    )
