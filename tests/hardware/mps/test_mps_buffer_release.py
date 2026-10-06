"""MPS Metal buffers must be returned to the system when a load is dropped.

PyObjC does not release buffers created by ``newBufferWithLength_options_`` when
the Python wrapper is collected, so without an explicit release every
``load(backend="mps")`` permanently retains its output. Seven no-bin tilts then
exhaust a 128 GB Mac even though only one tilt is ever meant to be resident.
"""

import gc
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="needs an Apple GPU",
)


def _allocated_bytes() -> int:
    return int(torch.mps.driver_allocated_memory())


def test_mtl_array_view_keeps_buffer_alive():
    """Slices must not outlive the buffer they read from.

    ``SharedArray`` once had no ``__array_finalize__``, so a view silently lost
    ``_mtl``. That was invisible while every buffer leaked; once release works it
    is a use-after-free, so views must carry the owner.
    """
    from quantem.gpu.device import metal_runtime

    buf = metal_runtime.allocate_shared(4096)
    arr = metal_runtime.numpy_view(buf, np.uint16, 1024).reshape(32, 32).view(metal_runtime.SharedArray)
    arr._mtl = buf
    view = arr[:8]
    assert view._mtl is arr._mtl, "view dropped its Metal buffer owner"
    assert arr.reshape(16, 64)._mtl is arr._mtl


def _decode_master(master, decoder=None):
    """Decode every frame of ``master`` through the bounded prepared-frame path."""
    from quantem.gpu.formats.hdf5.reads import FrameReader
    from quantem.gpu.io.hdf5.mps import decode as be

    with FrameReader(str(master)) as reader:
        prepared = reader.prepare(np.arange(int(reader.source_starts[-1])))
    if decoder is None:
        return be.load_prepared_frames(prepared)
    return decoder.load_prepared_frames(prepared)


def _release(be, array) -> None:
    """Release the one Metal buffer a decoded array owns."""
    from quantem.gpu.device.metal_runtime import release_buffer

    buffer, array._mtl = array._mtl, None
    release_buffer(buffer)


def test_tensor_buffer_reads_before_any_metal_import():
    """A Torch tensor's Metal buffer reads as an array in a fresh process that never imported Metal.

    PyObjC returns ``contents()`` as a bare int until the Metal metadata loads, so
    SSB on an MPS tensor failed when its test file ran on its own.
    """
    import subprocess
    import sys

    code = (
        "import sys, torch\n"
        "from quantem.gpu.device.metal_runtime import numpy_view, tensor_buffer\n"
        "assert 'Metal' not in sys.modules\n"
        "values = torch.arange(16, dtype=torch.float32, device='mps')\n"
        "torch.mps.synchronize()\n"
        "print(numpy_view(tensor_buffer(values), 'float32', 16).tolist())\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str([float(value) for value in range(16)])


def test_partial_bitshuffle_tail_preserves_uint32(tmp_path):
    """The uint32 tail port matches the canonical CUDA element mapping."""
    from quantem.gpu.io.hdf5.mps import decode as be

    values = (
        np.arange(2 * 48 * 48, dtype=np.uint32).reshape(2, 48, 48) * 1009
    )
    master = _write_bslz4_master(tmp_path, "tail_uint32", values)
    result = _decode_master(master)
    np.testing.assert_array_equal(result, values)
    _release(be, result)


def test_partial_bitshuffle_tail_only_frame_decodes_exactly(tmp_path):
    """A frame shorter than one 8 KiB block is decoded by the tail kernel alone."""
    from quantem.gpu.io.hdf5.mps import decode as be

    values = (np.arange(2 * 32 * 64, dtype=np.uint16) % 600).reshape(2, 32, 64)
    values[:, -2:, -2:] = 20_000
    master = _write_bslz4_master(tmp_path, "tail_only", values)
    result = _decode_master(master)
    np.testing.assert_array_equal(result, values)
    _release(be, result)


def _write_bslz4_master(
    root,
    name: str,
    values: np.ndarray,
    *,
    pixel_mask: np.ndarray | None = None,
):
    """Write one real bitshuffle-LZ4 detector frame and Arina-style master."""
    import h5py
    import hdf5plugin

    data_path = root / f"{name}_data_000001.h5"
    master_path = root / f"{name}_master.h5"
    with h5py.File(data_path, "w") as data_file:
        data_file.create_dataset(
            "entry/data/data",
            data=values,
            chunks=(1, values.shape[-2], values.shape[-1]),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with h5py.File(master_path, "w") as master_file:
        master_file["entry/data/data_000001"] = h5py.ExternalLink(
            str(data_path),
            "/entry/data/data",
        )
        master_file.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger",
            data=np.uint32(values.shape[0]),
        )
        if pixel_mask is not None:
            master_file.create_dataset(
                "entry/instrument/detector/detectorSpecific/pixel_mask",
                data=np.asarray(pixel_mask, dtype=np.uint8),
            )
    return master_path


def _write_bslz4_sharded_master(
    root,
    name: str,
    values: np.ndarray,
    split: int,
):
    """Write two external bitshuffle-LZ4 shards for output-offset coverage."""
    import h5py
    import hdf5plugin

    master_path = root / f"{name}_master.h5"
    data_paths = []
    for index, shard in enumerate((values[:split], values[split:]), start=1):
        data_path = root / f"{name}_data_{index:06d}.h5"
        with h5py.File(data_path, "w") as data_file:
            data_file.create_dataset(
                "entry/data/data",
                data=shard,
                chunks=(1, values.shape[-2], values.shape[-1]),
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
        data_paths.append(data_path)
    with h5py.File(master_path, "w") as master_file:
        for index, data_path in enumerate(data_paths, start=1):
            master_file[f"entry/data/data_{index:06d}"] = h5py.ExternalLink(
                str(data_path),
                "/entry/data/data",
            )
        master_file.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger",
            data=np.uint32(values.shape[0]),
        )
    return master_path


def test_partial_bitshuffle_tail_preserves_selective_order_and_duplicates(
    tmp_path,
):
    """Native indexing preserves tail counts and repeated position requests."""
    from quantem.gpu.io import load

    values = (np.arange(4 * 48 * 96, dtype=np.uint16) % 1000).reshape(
        4,
        48,
        96,
    )
    master = _write_bslz4_master(tmp_path, "tail_selective", values)
    with load(
        str(master),
        scan_shape=(2, 2),
        backend="mps",
        verbose=False,
    ) as result:
        selected = np.stack([
            result[row, col].cpu().numpy()
            for row, col in [(1, 1), (0, 1), (1, 1), (0, 0)]
        ])
        np.testing.assert_array_equal(selected, values[[3, 1, 3, 0]])


def test_partial_bitshuffle_tail_after_multiple_full_blocks_and_direct_load(
    tmp_path,
):
    """The tail remains exact after two full blocks in the direct decoder."""
    from quantem.gpu.io.hdf5.mps import decode as be

    values = (np.arange(2 * 96 * 96, dtype=np.uint16) % 1000).reshape(
        2,
        96,
        96,
    )
    master = _write_bslz4_master(tmp_path, "tail_direct", values)
    data_path = tmp_path / "tail_direct_data_000001.h5"
    decoder = be.MPSDecompressor(
        max_compressed_bytes=max(1 << 20, data_path.stat().st_size),
        max_frames=values.shape[0],
        frame_bytes=values[0].nbytes,
        n_blocks_per_frame=3,
    )
    try:
        direct = _decode_master(master, decoder)
        np.testing.assert_array_equal(direct, values)
        _release(be, direct)

        public = _decode_master(master)
        np.testing.assert_array_equal(public, values)
        _release(be, public)
    finally:
        decoder.free()


def test_partial_bitshuffle_tail_preserves_sharded_output_offsets(tmp_path):
    """Two source shards decode into their own frame offsets."""
    from quantem.gpu.io.hdf5.mps import decode as be

    values = (np.arange(5 * 96 * 96, dtype=np.uint16) % 100).reshape(
        5,
        96,
        96,
    )
    master = _write_bslz4_sharded_master(
        tmp_path,
        "tail_sharded",
        values,
        split=2,
    )
    result = _decode_master(master)
    np.testing.assert_array_equal(result, values)
    _release(be, result)


def test_partial_bitshuffle_tail_rejects_non_byte_aligned_elements():
    """Unsupported tail geometry fails closed with a corrective next step."""
    from quantem.gpu.io.hdf5.mps import decode as be

    with pytest.raises(ValueError, match="multiple of 8 elements"):
        be._bitshuffle_tail_elements(33 * 33 * 2, 2)


def test_partial_bitshuffle_tail_rejects_unsupported_source_before_allocation(
    monkeypatch,
):
    """Public prepared IO validates source width and tail before construction."""
    from quantem.gpu.io.hdf5.mps import decode as be

    def reject_construction(*args, **kwargs):
        pytest.fail("invalid source reached Metal allocation")

    monkeypatch.setattr(be, "MPSDecompressor", reject_construction)
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        be.load_prepared_frames(
            {
                "frame_bytes": 33 * 33 * 2,
                "dtype": np.dtype(np.uint16),
            }
        )
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        be.load_prepared_frames(
            {
                "frame_bytes": 33 * 33,
                "dtype": np.dtype(np.uint8),
            }
        )
    with pytest.raises(ValueError, match="1-byte uint8, 2-byte uint16 and 4-byte uint32"):
        be.load_prepared_frames(
            {
                "frame_bytes": 32 * 32 * 8,
                "dtype": np.dtype(np.uint64),
            }
        )


MAPED_TEST_DIR = os.environ.get("MAPED_TEST_DIR", "")


@pytest.mark.skipif(not os.path.isdir(MAPED_TEST_DIR), reason="needs MAPED_TEST_DIR")
def test_repeated_load_does_not_accumulate():
    """Loading tilts one at a time must not grow memory without bound."""
    import glob

    from quantem.gpu.io import load

    masters = sorted(glob.glob(os.path.join(MAPED_TEST_DIR, "*_master.h5")))[:3]
    if len(masters) < 2:
        pytest.skip("needs at least 2 masters")

    result = load(masters[0], backend="mps", verbose=False)
    one_tilt = _allocated_bytes()
    del result
    gc.collect()
    for path in masters[1:]:
        result = load(path, backend="mps", verbose=False)
        del result
        gc.collect()
    assert _allocated_bytes() <= one_tilt * 1.5, (
        f"memory grew from {one_tilt / 1e9:.1f} GB to "
        f"{_allocated_bytes() / 1e9:.1f} GB across {len(masters)} sequential loads"
    )
