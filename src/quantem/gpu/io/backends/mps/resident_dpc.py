"""GPU-resident Python MPS orchestration for the shared DPC Metal kernels.

The numerical kernels live in the Swift package's ``dpc.metal`` resource and
are also shipped in Python distributions. This module only owns Python buffer
lifecycle, dispatch, and synchronized metrics.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MPSDPCConfiguration",
    "MPSDPCMetrics",
    "MPSDPCProcessor",
    "MPSDPCResidentResult",
]

_PIPELINE_NAMES = (
    "dpc_pack_complex",
    "fft_bit_reverse_rows",
    "fft_bit_reverse_columns",
    "fft_butterfly_rows",
    "fft_butterfly_columns",
    "fft_normalize_2d",
    "dpc_poisson_frequency",
    "dpc_extract_phase",
)
_pipeline_cache: dict[int, tuple[dict[str, Any], float]] = {}


@dataclass(frozen=True)
class MPSDPCConfiguration:
    """Fixed consumer-selected DPC alignment for one complete scan plane."""

    scan_rows: int
    scan_columns: int
    rotation_degrees: float
    transpose_components: bool = False

    @property
    def count(self) -> int:
        """Return the validated number of scan positions."""

        rows = int(self.scan_rows)
        columns = int(self.scan_columns)
        if (
            rows <= 0
            or columns <= 0
            or rows & (rows - 1)
            or columns & (columns - 1)
            or rows * columns > np.iinfo(np.uint32).max
            or not math.isfinite(float(self.rotation_degrees))
        ):
            raise ValueError(
                "MPS DPC requires finite rotation and positive power-of-two "
                "scan dimensions within uint32."
            )
        return rows * columns


@dataclass(frozen=True)
class MPSDPCMetrics:
    """Synchronized attribution for one resident DPC/iDPC publication."""

    wall_ms: float
    gpu_ms: float
    fft_dispatch_count: int
    total_dispatch_count: int
    upload_bytes: int
    readback_bytes: int
    synchronization_count: int
    device_allocated_bytes_before: int
    device_allocated_bytes_after: int


class MPSDPCResidentResult:
    """Caller-owned MPS buffers for iDPC and Fourier products."""

    def __init__(
        self,
        *,
        configuration: MPSDPCConfiguration,
        phase_buffer,
        gradient_fft_buffer,
        phase_fft_buffer,
        metrics: MPSDPCMetrics,
    ) -> None:
        self.configuration = configuration
        self.phase_buffer = phase_buffer
        self.gradient_fft_buffer = gradient_fft_buffer
        self.phase_fft_buffer = phase_fft_buffer
        self.metrics = metrics
        self._released = False

    @property
    def is_released(self) -> bool:
        """Whether the result's owned buffers have been explicitly released."""

        return self._released

    def release(self) -> None:
        """Release the three Metal buffers owned by this result."""

        if self._released:
            return
        for buffer in (
            self.phase_buffer,
            self.gradient_fft_buffer,
            self.phase_fft_buffer,
        ):
            _release(buffer)
        self.phase_buffer = None
        self.gradient_fft_buffer = None
        self.phase_fft_buffer = None
        self._released = True


class MPSDPCProcessor:
    """Dispatch the shared Metal FFT/Poisson kernels from Python MPS.

    Parameters
    ----------
    device
        Optional PyObjC ``MTLDevice``. The system default is used otherwise.

    Examples
    --------
    ``process_buffers`` connects directly to compact resident DPC buffers. The
    consumer retains the returned buffers until presentation is complete.
    """

    def __init__(self, device=None) -> None:
        self._Metal = _metal_module()
        self._device = device or self._Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Python MPS DPC requires a physical Metal device")
        self._pipelines, self.pipeline_compile_ms = _make_pipelines(
            self._device, self._Metal
        )
        self._queue = self._device.newCommandQueue()
        if self._queue is None:
            raise RuntimeError("Python MPS DPC could not create a command queue")

    def process(
        self,
        centered_row: np.ndarray,
        centered_column: np.ndarray,
        configuration: MPSDPCConfiguration,
    ) -> MPSDPCResidentResult:
        """Upload centered float32 DPC maps and return resident Metal products."""

        count = configuration.count
        row = np.asarray(centered_row, dtype=np.float32).reshape(-1).copy()
        column = np.asarray(centered_column, dtype=np.float32).reshape(-1).copy()
        if row.size != count or column.size != count:
            raise ValueError("DPC maps require one value per complete scan position")
        if not np.all(np.isfinite(row)) or not np.all(np.isfinite(column)):
            raise ValueError("DPC maps must contain only finite float32 values")
        row_buffer = _allocate_shared(self._device, self._Metal, row.nbytes, "row DPC")
        column_buffer = _allocate_shared(
            self._device, self._Metal, column.nbytes, "column DPC"
        )
        try:
            _buffer_view(row_buffer)[:] = row.view(np.uint8)
            _buffer_view(column_buffer)[:] = column.view(np.uint8)
            return self._process_buffers(
                row_buffer,
                column_buffer,
                configuration,
                upload_bytes=row.nbytes + column.nbytes,
            )
        finally:
            _release(row_buffer)
            _release(column_buffer)

    def process_buffers(
        self,
        centered_row_buffer,
        centered_column_buffer,
        configuration: MPSDPCConfiguration,
    ) -> MPSDPCResidentResult:
        """Return resident products from caller-owned float32 Metal buffers."""

        return self._process_buffers(
            centered_row_buffer,
            centered_column_buffer,
            configuration,
            upload_bytes=0,
        )

    def _process_buffers(
        self,
        row_buffer,
        column_buffer,
        configuration: MPSDPCConfiguration,
        *,
        upload_bytes: int,
    ) -> MPSDPCResidentResult:
        count = configuration.count
        scalar_bytes = count * np.dtype(np.float32).itemsize
        complex_bytes = count * np.dtype(np.complex64).itemsize
        registry_id = int(self._device.registryID())
        if (
            int(row_buffer.device().registryID()) != registry_id
            or int(column_buffer.device().registryID()) != registry_id
            or int(row_buffer.length()) < scalar_bytes
            or int(column_buffer.length()) < scalar_bytes
        ):
            raise ValueError(
                "DPC buffers must be complete float32 maps on the processor device"
            )

        allocated_before = int(self._device.currentAllocatedSize())
        gradient = _allocate(
            self._device,
            self._Metal,
            complex_bytes,
            self._Metal.MTLResourceStorageModePrivate,
            "DPC gradient FFT",
        )
        phase_fft = _allocate(
            self._device,
            self._Metal,
            complex_bytes,
            self._Metal.MTLResourceStorageModePrivate,
            "iDPC phase FFT",
        )
        phase = _allocate_shared(
            self._device, self._Metal, scalar_bytes, "iDPC phase"
        )
        try:
            command = self._queue.commandBuffer()
            if command is None:
                raise RuntimeError("Python MPS could not encode DPC/iDPC")
            encoder = command.computeCommandEncoder()
            if encoder is None:
                raise RuntimeError("Python MPS could not encode DPC/iDPC")
            started = time.perf_counter()
            self._encode_pack(
                encoder,
                row_buffer,
                column_buffer,
                gradient,
                configuration,
                count,
            )
            encoder.memoryBarrierWithScope_(self._Metal.MTLBarrierScopeBuffers)
            self._encode_fft(
                encoder,
                gradient,
                configuration.scan_rows,
                configuration.scan_columns,
                inverse=False,
            )
            shape = struct.pack(
                "<4I",
                configuration.scan_columns,
                configuration.scan_rows,
                count,
                0,
            )
            encoder.setComputePipelineState_(
                self._pipelines["dpc_poisson_frequency"]
            )
            encoder.setBuffer_offset_atIndex_(gradient, 0, 0)
            encoder.setBuffer_offset_atIndex_(phase_fft, 0, 1)
            encoder.setBytes_length_atIndex_(shape, len(shape), 2)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                self._Metal.MTLSizeMake(count, 1, 1),
                self._Metal.MTLSizeMake(256, 1, 1),
            )
            encoder.memoryBarrierWithScope_(self._Metal.MTLBarrierScopeBuffers)
            self._encode_fft(
                encoder,
                phase_fft,
                configuration.scan_rows,
                configuration.scan_columns,
                inverse=True,
            )
            encoder.setComputePipelineState_(self._pipelines["dpc_extract_phase"])
            encoder.setBuffer_offset_atIndex_(phase_fft, 0, 0)
            encoder.setBuffer_offset_atIndex_(phase, 0, 1)
            count_bytes = struct.pack("<I", count)
            encoder.setBytes_length_atIndex_(count_bytes, len(count_bytes), 2)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                self._Metal.MTLSizeMake(count, 1, 1),
                self._Metal.MTLSizeMake(256, 1, 1),
            )
            encoder.endEncoding()
            gpu_ms = _complete(command, "DPC/iDPC")
            wall_ms = (time.perf_counter() - started) * 1_000.0
        except Exception:
            _release(gradient)
            _release(phase_fft)
            _release(phase)
            raise

        fft_dispatches = 2 * (
            2
            + (configuration.scan_rows.bit_length() - 1)
            + (configuration.scan_columns.bit_length() - 1)
        ) + 1
        return MPSDPCResidentResult(
            configuration=configuration,
            phase_buffer=phase,
            gradient_fft_buffer=gradient,
            phase_fft_buffer=phase_fft,
            metrics=MPSDPCMetrics(
                wall_ms=wall_ms,
                gpu_ms=gpu_ms,
                fft_dispatch_count=fft_dispatches,
                total_dispatch_count=fft_dispatches + 3,
                upload_bytes=int(upload_bytes),
                readback_bytes=0,
                synchronization_count=1,
                device_allocated_bytes_before=allocated_before,
                device_allocated_bytes_after=int(self._device.currentAllocatedSize()),
            ),
        )

    def _encode_pack(
        self,
        encoder,
        row_buffer,
        column_buffer,
        gradient_buffer,
        configuration: MPSDPCConfiguration,
        count: int,
    ) -> None:
        angle = math.radians(float(configuration.rotation_degrees))
        parameters = struct.pack(
            "<4I4f",
            count,
            int(configuration.transpose_components),
            0,
            0,
            math.cos(angle),
            math.sin(angle),
            0,
            0,
        )
        encoder.setComputePipelineState_(self._pipelines["dpc_pack_complex"])
        encoder.setBuffer_offset_atIndex_(row_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(column_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(gradient_buffer, 0, 2)
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 3)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._Metal.MTLSizeMake(count, 1, 1),
            self._Metal.MTLSizeMake(256, 1, 1),
        )

    def _encode_fft(
        self,
        encoder,
        buffer,
        rows: int,
        columns: int,
        *,
        inverse: bool,
    ) -> None:
        width_stages = columns.bit_length() - 1
        height_stages = rows.bit_length() - 1

        def dispatch(
            name: str,
            width: int,
            height: int,
            log2_size: int,
            stage: int,
            row_axis: bool,
        ) -> None:
            parameters = struct.pack(
                "<4IfI",
                columns,
                rows,
                log2_size,
                stage,
                1.0 if inverse else -1.0,
                int(row_axis),
            )
            encoder.setComputePipelineState_(self._pipelines[name])
            encoder.setBuffer_offset_atIndex_(buffer, 0, 0)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 1)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                self._Metal.MTLSizeMake(width, height, 1),
                self._Metal.MTLSizeMake(16, 16, 1),
            )
            encoder.memoryBarrierWithScope_(self._Metal.MTLBarrierScopeBuffers)

        dispatch(
            "fft_bit_reverse_rows",
            columns,
            rows,
            width_stages,
            0,
            True,
        )
        for stage in range(width_stages):
            dispatch(
                "fft_butterfly_rows",
                columns // 2,
                rows,
                width_stages,
                stage,
                True,
            )
        dispatch(
            "fft_bit_reverse_columns",
            columns,
            rows,
            height_stages,
            0,
            False,
        )
        for stage in range(height_stages):
            dispatch(
                "fft_butterfly_columns",
                columns,
                rows // 2,
                height_stages,
                stage,
                False,
            )
        if inverse:
            dispatch("fft_normalize_2d", columns, rows, height_stages, 0, False)


def _metal_module():
    try:
        import Metal
    except ImportError as error:
        raise RuntimeError("Python MPS DPC requires pyobjc-framework-Metal") from error
    return Metal


def _make_pipelines(device, Metal) -> tuple[dict[str, Any], float]:
    registry_id = int(device.registryID())
    cached = _pipeline_cache.get(registry_id)
    if cached is not None:
        return cached
    started = time.perf_counter()
    resource = (
        Path(__file__).resolve().parents[3]
        / "swift"
        / "Sources"
        / "Metal4DSTEMKernels"
        / "Resources"
        / "dpc.metal"
    )
    if not resource.is_file():
        raise RuntimeError(f"Python MPS DPC resource is missing: {resource}")
    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(
        resource.read_text(), options, None
    )
    if library is None or error is not None:
        raise RuntimeError(f"Python MPS DPC Metal compile failed: {error}")
    pipelines: dict[str, Any] = {}
    for name in _PIPELINE_NAMES:
        function = library.newFunctionWithName_(name)
        pipeline, pipeline_error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None or pipeline_error is not None:
            raise RuntimeError(f"Python MPS DPC pipeline {name} failed: {pipeline_error}")
        pipelines[name] = pipeline
    result = pipelines, (time.perf_counter() - started) * 1_000.0
    _pipeline_cache[registry_id] = result
    return result


def _allocate(device, Metal, nbytes: int, options: int, label: str):
    buffer = device.newBufferWithLength_options_(int(nbytes), options)
    if buffer is None:
        raise MemoryError(f"Metal could not allocate {nbytes} bytes for {label}")
    return buffer


def _allocate_shared(device, Metal, nbytes: int, label: str):
    return _allocate(
        device,
        Metal,
        nbytes,
        Metal.MTLResourceStorageModeShared,
        label,
    )


def _buffer_view(buffer, nbytes: int | None = None) -> memoryview:
    count = int(buffer.length()) if nbytes is None else int(nbytes)
    return memoryview(buffer.contents().as_buffer(count))


def _complete(command, operation: str) -> float:
    command.commit()
    command.waitUntilCompleted()
    if command.error() is not None:
        raise RuntimeError(f"Python MPS {operation} failed: {command.error()}")
    return max(0.0, float(command.GPUEndTime()) - float(command.GPUStartTime())) * 1_000


def _release(buffer) -> None:
    if buffer is None:
        return
    try:
        buffer.release()
    except (AttributeError, ValueError):
        pass
