"""Save encoded acquisitions as ``.qem`` copies and arrays as Arina HDF5."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.formats.qem.reference import save_array
from quantem.gpu.io import qem
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.hdf5.write import (
    _DEFAULT_BATCH_SIZE,
    H5Writer,
    _default_save_dtype,
    _infer_save_dtype,
    _is_mps_array,
    _normalize_save_dtype,
    _output_file_size,
    save_compressed_arina_h5,
    wait_for_saves,
)
from quantem.gpu.io.precision import precision_name, save_precision
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.float_ans import FloatANSResident
from quantem.gpu.resident.mps.counts import MPSStreamedCounts

try:
    import torch
except ImportError:  # pragma: no cover - only for minimal IO-only installs
    torch = None


@dataclass
class SaveResult:
    """Deterministic completion handle returned by :func:`quantem.gpu.io.save`."""

    path: str
    backend: str
    complete: bool = False

    def wait(self) -> Self:
        """Wait for queued writes, surface errors, and return this result."""
        wait_for_saves()
        self.complete = True
        return self


def save(
    filepath: str | Path,
    data: object,
    scan_shape: tuple[int, int] | None = None,
    metadata: dict | None = None,
    dtype: str | type | np.dtype | None = None,
    batch_size: int | None = None,
    wait: bool = True,
    verbose: bool = False,
    source_master: str | None = None,
    frames_per_file: int = 32768,
    format: str = "arina",
    backend: str = "auto",
    compression: str = "auto",
    compression_level: int = 0,
) -> SaveResult:
    """Save encoded acquisitions as QEM or array results as HDF5.

    ``format="quantem"`` instead writes one self-contained ``.qem`` copy, not an
    HDF5 master/shard set. A complete encoded CUDA or MPS/Metal resident saves
    its existing bytes without re-encoding. Explicit ``backend="cpu"`` accepts
    a 4D NumPy uint8/uint16 or float32 array, using the
    portable reference encoder. Calibration is retained and an
    existing destination is never replaced. Saved copies reopen on CUDA or
    MPS/Metal as encoded counts, or explicitly on CPU as original dense measurements.
    The HDF5-specific options and discussion below do not apply to ``.qem``.
    ``compression="auto"`` preserves the existing default encoding: ANS for
    QuantEM files and bitshuffle/LZ4 for Arina files.
    A ``.qem`` destination selects QuantEM format automatically. Reopening
    preserves the encoded workflow::

        with io.load("acquisition.h5") as acquisition:
            io.save("acquisition.qem", acquisition)
        # Later: io.load("acquisition.qem")

    Output: a master HDF5 file pointing to ``*_data_NNNNNN.h5`` external files
    with per-frame HDF5 chunks. Matches Arina row/column native chunking. The
    public compression method is Bitshuffle/LZ4; ``backend="auto"`` chooses the
    CUDA or MPS/Metal implementation from the input data. The CPU reference
    writer is available only when requested explicitly.

    ``format="arina"`` names the acquisition/layout and writes HDF5 files;
    compression is selected separately.

    Preserve numerical values
    -------------------------
    Omit ``dtype`` to retain the input precision. Drift correction and other
    processing can produce fractional or negative values; converting them to
    unsigned counts changes those values and does not restore the original
    detector statistics. Float-to-integer export rounds to nearest, ties to
    even, and clips to the requested range. Rounding error is at most half a
    count only for finite values within that range; clipping can be larger.
    Validate any approximate export for the intended scientific analysis.

    Approximate precision exports
    -----------------------------
    ``dtype="scaled_uint16"`` calibrates bounded scan regions and records
    GPU-measured conversion errors. Save this approximate result as HDF5;
    the QEM precision codec is not implemented. It reopens through ``io.load``
    as ANS-encoded scaled counts in the original intensity units.
    Keep float32 for an unchanged scientific archive. ``dtype`` changes storage
    precision at this boundary, not the precision of the upstream algorithm.
    Scaled uint16 uses a uniform step within each automatically selected region.
    Plain uint16 conversion does not provide this calibration. The supported
    spelling is ``"scaled_uint16"``, not ``"uint16_scaled"``.
    Compression is lossless relative to the converted stored codes. Reopening
    returns float32 reconstructed intensities, not the original
    pre-conversion float32 values. RMSE and maximum error describe this storage
    difference. Omit ``dtype`` when saving an already-loaded precision resident
    to preserve its codes and calibration without another conversion.

    Drift metadata co-saved with the 4D-STEM
    ----------------------------------------
    Save BOTH the spline knot positions (compact, model-of-record) and the
    dense per-position offsets (ready for ptycho without re-evaluation). Pass
    them via ``metadata=`` (root attrs) or write into the master file::

        save("corrected_master.h5", merged_f32, metadata={
            "drift_model": "spline_n16",
            "drift_knots": knots,                     # (n_imgs, 2, n_knots)
            "drift_probe_positions_px": probe_pos,    # (N_scan_pos, 2)
        })

    Knots = small (kilobytes), regenerable, source of truth.
    Probe positions = dense (~2 MB at 512²), consumed directly by ptycho.

    Parameters
    ----------
    filepath : str
        Output file path. For Arina, external data files are written next to
        the master with the same prefix. QuantEM writes a standalone file.
    data : object
        4D-STEM data. Shape (N, det_row, det_col) or (scan_row, scan_col,
        det_row, det_col). CuPy and Torch accelerator tensors save without a
        host copy. Algorithm packages may also provide a re-readable 4D block
        source with ``shape``, ``dtype``, and ``blocks()``. Each call to
        ``blocks()`` must yield the complete data as ordered accelerator frame
        blocks; scaled integer export calls it twice to measure one global
        range and then encode. Optional ``save_metadata`` attributes are copied
        into the output. This is the public bridge for bounded algorithm output;
        the I/O package does not own the algorithm that produces the blocks.
    scan_shape : tuple[int, int] | None
        Scan grid shape. Required for 3D inputs; inferred from 4D inputs.
    dtype : str or np.dtype or None
        Output dtype. ``None`` uses input dtype. Short aliases such as
        ``"u16"`` and ``"f32"`` are accepted. Changing the dtype can change
        measurements; preserve float32 when exact fractional values matter.
    batch_size : int
        Frames compressed per GPU pass. ``None`` uses the backend default.
    format : {"arina", "quantem"}
        Output file layout. ``"arina"`` writes the QuantEM/Arina-style
        master/data layout. ``"quantem"`` writes the standalone ``.qem`` copy of
        an encoded resident.
    backend : {"auto", "cuda", "mps", "cpu"}
        Compression/write backend. ``"auto"`` keeps CUDA CuPy arrays on CUDA
        and MPS tensors on Metal.
    compression : {"auto", "ans", "lz4", "bslz4", "bitshuffle_lz4"}
        File compression, independent of the loaded resident representation.
        ``"auto"`` uses ANS for QuantEM or bitshuffle/LZ4 for Arina, preserving
        previous default file encodings. ANS is not an Arina/HDF5 filter.
        Other codecs are retained only for internal archival helpers.
    compression_level : int
        Codec level. 0 = codec default. Ignored for LZ4.
    metadata : dict | None
        Saved as root attributes on the master file. Use for drift knots,
        probe positions, calibration.

    See also
    --------
    quantem.gpu.io.load : Round-trip read of these files; bit-exact for
        lossless storage. Precision conversions require their own error checks.
    """
    if torch is not None and isinstance(data, torch.Tensor) and data.is_cuda:
        data = cp.from_dlpack(data.detach())

    if precision_name(dtype) or (isinstance(data, Dataset4dstemGPU) and "precision" in data.metadata):
        if Path(filepath).suffix.lower() == ".qem":
            raise NotImplementedError(
                "QEM precision codecs are not implemented. Save this scaled/quantized "
                "result as HDF5, or save an exact supported acquisition as .qem."
            )
        precision_backend = save_precision(filepath, data, dtype=dtype, scan_shape=scan_shape,
            metadata=metadata, backend=backend, format=format, compression=compression,
            frames_per_file=frames_per_file, verbose=verbose, source_master=source_master)
        return SaveResult(str(filepath), precision_backend, complete=True)

    normalized_format = str(format).lower()
    if Path(filepath).suffix.lower() == ".qem":
        normalized_format = "quantem"
    if normalized_format not in {"arina", "quantem"}:
        raise ValueError(
            f"Unsupported save format {format!r}; use format='arina' or 'quantem'."
        )
    if normalized_format == "quantem" and Path(filepath).suffix.lower() != ".qem":
        raise ValueError(
            "format='quantem' writes saved copies with the .qem extension; "
            f"got {Path(filepath).name!r}."
        )
    if isinstance(compression, str):
        compression = compression.lower()
    if compression == "auto":
        compression = "ans" if normalized_format == "quantem" else "lz4"

    if normalized_format == "quantem":
        if compression != "ans":
            raise ValueError(
                f"format='quantem' does not support compression={compression!r}; "
                "use compression='ans', or format='arina' for bitshuffle/LZ4."
            )
        if compression_level not in (None, 0):
            raise ValueError(
                f"ANS has no compression_level control; got {compression_level!r}. "
                "Remove compression_level to preserve the exact codec contract."
            )
        if isinstance(data, Dataset4dstemGPU):
            metadata = dict(data.metadata) if metadata is None else metadata
            data = data.data
            if torch is not None and isinstance(data, torch.Tensor) and data.is_cuda:
                data = cp.from_dlpack(data.detach())
        if dtype is not None or scan_shape is not None or source_master is not None:
            raise ValueError("Saving a native 4D acquisition preserves its own geometry; remove dtype, scan_shape, and source_master controls.")
        if backend == "cpu" and isinstance(data, np.ndarray):
            save_array(filepath, data, metadata, chunk_scans=512 if batch_size is None else batch_size)
            return SaveResult(str(filepath), "cpu", complete=True)
        if isinstance(data, FloatANSResident):
            if backend not in ("auto", data.backend) or batch_size is not None:
                raise ValueError("Save the float resident on its own backend without batch_size.")
            qem.save_float(filepath, data, metadata)
            return SaveResult(str(filepath), data.backend, complete=True)

        if isinstance(data, (StreamedCounts, MPSStreamedCounts)):
            resident_backend = "mps" if isinstance(data, MPSStreamedCounts) else "cuda"
            if backend not in ("auto", resident_backend) or batch_size is not None:
                raise ValueError(f"Save resident ANS with backend='{resident_backend}' and no batch_size; its exact chunk layout is retained.")
            qem.save_streamed(filepath, data, metadata)
            return SaveResult(str(filepath), resident_backend, complete=True)
        raise NotImplementedError(
            "Writing a .qem copy requires a complete encoded CUDA or Metal resident. "
            "Load the acquisition with representation='encoded' first; nothing was written."
        )

    t0 = time.perf_counter()
    if compression == "ans":
        raise ValueError(
            "compression='ans' is not supported for format='arina'; "
            "use format='quantem', or compression='bitshuffle_lz4' for Arina."
        )
    if compression in {"bslz4", "bitshuffle_lz4"}:
        compression = "lz4"
    normalized_backend = str(backend).lower()
    if normalized_backend not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("backend must be 'auto', 'cuda', 'mps', or 'cpu'")

    if normalized_backend == "auto":
        if _is_cuda_array(data):
            normalized_backend = "cuda"
        elif _is_mps_array(data):
            normalized_backend = "mps"
        else:
            raise RuntimeError(
                "backend='auto' could not infer an accelerated writer from the "
                "input. Pass a CuPy CUDA array or MPS tensor, or request "
                "backend='cpu' explicitly for the reference writer."
            )

    if normalized_backend in {"mps", "cpu"}:
        compression_backend = "mps" if normalized_backend == "mps" else "hdf5"
        save_compressed_arina_h5(
            filepath,
            data,
            scan_shape=scan_shape,
            metadata=metadata,
            dtype=_infer_save_dtype(data) if dtype is None else dtype,
            batch_size=_DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size),
            frames_per_file=frames_per_file,
            compression=compression,
            compression_level=None if compression_level == 0 else compression_level,
            source_master=source_master,
            compression_backend=compression_backend,
        )
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"Saved {filepath} [{normalized_backend}/{compression}] in {elapsed:.2f}s")
        result = SaveResult(str(filepath), normalized_backend, complete=True)
        return result.wait() if wait else result

    if not _is_cuda_array(data):
        raise ValueError(
            "backend='cuda' requires a CuPy CUDA array. Use backend='auto', "
            "'mps', or 'cpu' for non-CUDA inputs."
        )

    cuda_batch_size = 4096 if batch_size is None else int(batch_size)
    data_gpu, dtype, scan_shape = _prepare_save_data(data, dtype, scan_shape)
    n_frames, det_row, det_col = (int(x) for x in data_gpu.shape)

    writer = H5Writer(
        filepath,
        n_frames=n_frames,
        det_shape=(det_row, det_col),
        scan_shape=scan_shape,
        metadata=metadata,
        dtype=dtype,
        source_master=source_master,
        frames_per_file=frames_per_file,
        compression=compression,
        compression_level=compression_level,
    )
    try:
        for start in range(0, n_frames, cuda_batch_size):
            writer.write(data_gpu[start:start + cuda_batch_size])
    finally:
        writer.close(wait=wait)

    if verbose:
        elapsed = time.perf_counter() - t0
        file_size = _output_file_size(filepath)
        raw = n_frames * det_row * det_col * dtype.itemsize
        codec = writer._compression
        if codec != "lz4":
            codec = f"{codec}@{writer._compression_level}"
        print(
            f"Saved {filepath} [{codec}]: {raw / 1e9:.2f} GB -> "
            f"{file_size / 1e9:.2f} GB ({raw / file_size:.1f}x) in {elapsed:.2f}s"
        )
    result = SaveResult(str(filepath), "cuda", complete=bool(wait))
    return result.wait() if wait else result


def _prepare_save_data(data, dtype, scan_shape):
    """Return contiguous ``(n_frames, det_row, det_col)`` CuPy frames in the save dtype.

    Floats saved as integers are rounded to nearest and clipped, and 4D input
    supplies the scan shape that the master file records.
    """
    input_dtype = data.dtype if isinstance(data, cp.ndarray) else np.asarray(data).dtype
    dtype = _normalize_save_dtype(dtype if dtype is not None else _default_save_dtype(input_dtype))
    data_gpu = data if isinstance(data, cp.ndarray) else cp.asarray(np.asarray(data))
    if data_gpu.dtype != dtype:
        if (np.issubdtype(data_gpu.dtype, np.floating)
                and np.issubdtype(dtype, np.integer)):
            lo, hi = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
            data_gpu = cp.clip(cp.rint(data_gpu), lo, hi).astype(dtype)
        else:
            data_gpu = data_gpu.astype(dtype)
    if data_gpu.ndim == 4:
        inferred_scan = tuple(int(x) for x in data_gpu.shape[:2])
        if scan_shape is not None and tuple(scan_shape) != inferred_scan:
            raise ValueError(f"scan_shape={scan_shape} does not match data shape {inferred_scan}")
        scan_shape = inferred_scan
        data_gpu = data_gpu.reshape(-1, data_gpu.shape[-2], data_gpu.shape[-1])
    elif data_gpu.ndim != 3:
        raise ValueError("save() expects 3D frames or 4D-STEM data")
    data_gpu = cp.ascontiguousarray(data_gpu)
    return data_gpu, dtype, scan_shape


def _is_cuda_array(data) -> bool:
    """Return True for CuPy/CUDA arrays without importing CUDA on CPU/Mac hosts."""
    return cp is not None and isinstance(data, cp.ndarray)
