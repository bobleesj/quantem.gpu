"""DigitalMicrograph metadata and bounded native-count CUDA loading."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class NoDiffractionImage(ValueError):
    """The DM document contains only survey images or spectra."""


@dataclass(frozen=True)
class DMSource:
    """One calibrated 4D image, excluding thumbnails and survey images."""

    path: Path
    offset: int
    shape: tuple[int, int, int, int]
    dtype: np.dtype
    metadata: dict[str, object]
    signature: tuple[int, int]

    def memmap(self) -> np.memmap:
        """Map original bytes in scan-row, scan-column, detector-row, column order."""
        return np.memmap(self.path, mode="r", offset=self.offset,
                         dtype=self.dtype, shape=self.shape)

    def assert_unchanged(self) -> None:
        """Reject a source rewritten while its detector data was being loaded."""
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.signature:
            raise ValueError(f"{self.path} changed during loading; reopen the complete file.")


def read_dm_source(
    path: str | Path, scan_shape: tuple[int, int] | None = None,
) -> DMSource:
    """Select the unique calibrated diffraction image using only DM tags."""
    try:
        from ncempy.io import dm
    except ImportError as error:
        raise ImportError(
            "DigitalMicrograph files need ncempy. Install quantem.gpu[dm] "
            "in the environment running this viewer."
        ) from error

    path = Path(path).expanduser().resolve()
    stat = path.stat()
    with dm.fileDM(path) as source:
        candidates = [i for i, ndim in enumerate(source.dataShape) if ndim == 4]
        if not candidates:
            raise NoDiffractionImage(f"{path.name} contains no four-dimensional diffraction image.")
        if len(candidates) != 1:
            raise ValueError(
                f"{path.name} contains {len(candidates)} four-dimensional images; "
                "export one calibrated 4D-STEM image per DM file."
            )
        index = candidates[0]
        shape = tuple(int(value) for value in (
            source.zSize2[index], source.zSize[index],
            source.ySize[index], source.xSize[index],
        ))
        if any(size <= 0 for size in shape):
            raise ValueError(f"{path.name} has invalid image dimensions {shape}; re-export the acquisition.")
        start = sum(source.dataShape[:index])
        units = list(source.scaleUnit[start:start + 4])[::-1]
        sampling = [float(value) for value in source.scale[start:start + 4]][::-1]
        origins = [float(value) for value in source.origin[start:start + 4]][::-1]
        reciprocal = {"1/nm", "1/Å", "1/A", "1/Angstrom", "1/um", "1/µm"}
        if len(units) != 4 or not all(unit in reciprocal for unit in units[2:]):
            raise ValueError(
                f"{path.name} has unsupported DM axis units {units}; export the "
                "4D image with the two diffraction axes fastest and reciprocal calibration."
            )
        if scan_shape is not None and tuple(scan_shape) != shape[:2]:
            raise ValueError(
                f"scan_shape={scan_shape} disagrees with DM geometry {shape[:2]}; "
                "omit scan_shape to retain the acquisition."
            )
        endian = "<" if int(source._endianType[0]) else ">"
        dtype = np.dtype(source._DM2NPDataType(source.dataType[index])).newbyteorder(endian)
        offset = int(source.dataOffset[index])
        size = math.prod(shape) * dtype.itemsize
        if size != int(source.dataSize[index]) or offset + size > stat.st_size:
            raise ValueError(f"{path.name} has an incomplete detector payload; finish the download.")
        prefix = f".ImageList.{index + 1}."
        tags = {key[len(prefix):]: value for key, value in source.allTags.items()
                if key.startswith(prefix) and ".Data." not in key and not key.endswith(".ImageData.Data")}
        retained = {
            f"dm{int(source._dmType)}." + key: value if isinstance(value, str) else json.dumps(
                value, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item)
            )
            for key, value in tags.items()
        }
        voltage = tags.get("ImageTags.Microscope Info.Voltage")
        metadata = dict(
            source_metadata=retained,
            camera_model=tags.get("ImageTags.Acquisition.Device.Source Model"),
            camera_id=tags.get("ImageTags.Acquisition.Device.Source ID"),
            acquisition_date=tags.get("ImageTags.SI.Acquisition.Date"),
            acquisition_processing=tags.get("ImageTags.Acquisition.Parameters.High Level.Processing"),
            median_correction_applied=False,
            source_kind="digitalmicrograph", source_path=str(path),
            source_shape=shape, working_shape=shape, scan_shape=shape[:2],
            detector_shape=shape[2:], source_dtype=dtype.name, dtype=dtype.name,
            n_frames=math.prod(shape[:2]), dm_image_index=index,
            dm_data_offset=offset, sampling=sampling, units=units,
            pixel_origin=origins, representation="dense",
            voltage_kV=float(voltage) / 1000 if voltage is not None else None,
            dm_data_order_swapped=bool(tags.get("ImageTags.Meta Data.Data Order Swapped", False)),
            axis_order=["scan_row", "scan_col", "detector_row", "detector_col"],
        )
        if all(unit in {"nm", "µm", "um", "Å", "A"} for unit in units[:2]):
            factors = {"nm": 10, "µm": 10000, "um": 10000, "Å": 1, "A": 1}
            metadata["scan_sampling_A"] = [sampling[i] * factors[units[i]] for i in range(2)]
        if units[2:] == ["1/nm", "1/nm"]:
            metadata["detector_sampling_inv_A"] = [value / 10 for value in sampling[2:]]
    return DMSource(path, offset, shape, dtype, metadata, (stat.st_size, stat.st_mtime_ns))


def load_dm(path, *, backend, representation, scan_shape, device, verbose):
    """Stream native DM detector bytes once through pinned staging into CUDA ANS."""
    from .backends import resolve_backend
    from .models import FourDSTEMData
    from .representation import DataRepresentation

    started = time.perf_counter()
    source = read_dm_source(path, scan_shape)
    backend = resolve_backend(backend)
    representation = DataRepresentation.parse(
        representation or ("dense" if backend == "cpu" else "encoded")
    )
    metadata = dict(source.metadata)
    if backend == "cpu" and representation is DataRepresentation.DENSE:
        metadata.update(backend="cpu", residency="host", file_counts_exact=True,
                        lossless_exact=True, scan_bin=1, detector_bin=1, crop=None)
        return FourDSTEMData(source.memmap(), metadata)
    if (backend in {"cuda", "mps"} and representation is DataRepresentation.ENCODED
            and source.dtype == np.dtype("float32")):
        from ._array_resident import load_array_resident, read_frame_block

        mapped = source.memmap()
        loaded = None
        try:
            loaded = load_array_resident(
                source.shape, source.dtype,
                lambda first, stop: read_frame_block(mapped, source.shape, first, stop),
                metadata, backend=backend, device=device, verbose=verbose,
            )
            source.assert_unchanged()
            return loaded
        except BaseException:
            if loaded is not None:
                loaded.close()
            raise
        finally:
            mapped._mmap.close()
    if backend == "mps" and representation is DataRepresentation.ENCODED:
        from ._camera_mps import load_camera_mps

        if source.dtype.name not in {"uint8", "uint16"} or not source.dtype.isnative:
            raise TypeError("DM Metal ANS requires native uint8/uint16 counts.")
        return load_camera_mps(source, verbose=verbose)
    if backend != "cuda" or representation is not DataRepresentation.ENCODED:
        raise NotImplementedError(
            "DigitalMicrograph currently supports CUDA encoded counts or explicit "
            "backend='cpu', representation='dense' reference access."
        )
    if source.dtype.name not in {"uint8", "uint16"} or not source.dtype.isnative:
        raise TypeError(
            f"DM ANS requires native uint8/uint16 counts; got {source.dtype}. "
            "Use backend='cpu', representation='dense' to inspect without conversion."
        )
    import cupy as cp
    from quantem.gpu._compact.streamed import StreamedCounts

    selected = cp.cuda.Device().id if device is None else int(str(device).removeprefix("cuda:"))
    with cp.cuda.Device(selected):
        resident = StreamedCounts(source.shape, source.dtype)
        # Two 512-frame buffers bound staging even for large camera frames.
        frames = math.prod(source.shape[:2])
        chunk_frames = min(512, frames)
        block_shape = (chunk_frames, *source.shape[2:])
        block_bytes = math.prod(block_shape) * source.dtype.itemsize
        owners = [cp.cuda.alloc_pinned_memory(block_bytes) for _ in range(2)]
        staging = [np.frombuffer(owner, source.dtype, count=math.prod(block_shape)).reshape(block_shape)
                   for owner in owners]
        raw = cp.empty(block_shape, source.dtype)
        stream = cp.cuda.get_current_stream()
        read_seconds = 0.0

        def read(first, slot):
            before = time.perf_counter()
            count = min(chunk_frames, frames - first)
            with source.path.open("rb", buffering=0) as file:
                file.seek(source.offset + first * math.prod(source.shape[2:]) * source.dtype.itemsize)
                destination = memoryview(staging[slot][:count]).cast("B")
                cursor = 0
                while cursor < len(destination):
                    length = file.readinto(destination[cursor:])
                    if not length:
                        raise ValueError(f"{source.path} ended inside its detector payload.")
                    cursor += length
            return count, time.perf_counter() - before

        try:
            with ThreadPoolExecutor(max_workers=1) as reader:
                pending = reader.submit(read, 0, 0)
                for number, first in enumerate(range(0, frames, chunk_frames)):
                    slot = number % 2
                    count, elapsed = pending.result()
                    read_seconds += elapsed
                    if first + count < frames:
                        pending = reader.submit(read, first + count, 1 - slot)
                    raw[:count].set(staging[slot][:count], stream=stream)
                    resident.append(raw[:count])
                    # append completes both the upload and encoding before the slot is reused.
            source.assert_unchanged()
            metadata.update(
                backend="cuda", device=f"cuda:{selected}", representation="encoded",
                residency="device", working_dtype=source.dtype.name,
                source_logical_tensor_bytes=math.prod(source.shape) * source.dtype.itemsize,
                working_logical_tensor_bytes=math.prod(source.shape) * source.dtype.itemsize,
                physical_resident_bytes=resident.nbytes, index_bytes=resident.index_nbytes,
                resident_profile="runtime-column-rans-spatial-v2", source_read_passes=1,
                lossless_exact=True, file_counts_exact=True, working_counts_exact=True,
                detector_mask_policy="preserve-stored-counts", scan_bin=1, detector_bin=1, crop=None,
                load_timings=dict(resident.load_metrics, read_seconds=read_seconds,
                                  resident_ready_seconds=time.perf_counter() - started,
                                  max_chunk_scans=chunk_frames, pinned_staging_bytes=2 * block_bytes),
            )
            if verbose:
                print(f"Loaded {source.path.name} {source.shape} {source.dtype.name} into "
                      f"lossless CUDA ANS in {metadata['load_timings']['resident_ready_seconds']:.2f} s.")
            return FourDSTEMData(resident, metadata)
        except BaseException:
            resident.release()
            raise
        finally:
            stream.synchronize()
