"""Apple Metal MP4 rendering backend for :mod:`quantem.gpu.movie`.

This backend renders grayscale movie grids to NV12 with a Metal compute kernel,
then asks ffmpeg to encode H.264. On macOS the default codec is
``h264_videotoolbox`` with a ``libx264`` fallback.
"""

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantem.gpu.device.metal_runtime import numpy_view, release_buffer
from quantem.gpu.movie.layout import grid_layout, label_font, label_mask, panel_labels

_METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void scale_grid_nv12(
    device const float* stacks [[buffer(0)]],
    device const float* vmin [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device uchar* dst [[buffer(3)]],
    constant uint& frame_idx [[buffer(4)]],
    constant uint& n_panels [[buffer(5)]],
    constant uint& n_frames [[buffer(6)]],
    constant uint& src_width [[buffer(7)]],
    constant uint& src_height [[buffer(8)]],
    constant uint& out_width [[buffer(9)]],
    constant uint& out_height [[buffer(10)]],
    constant uint& frame_width [[buffer(11)]],
    constant uint& frame_height [[buffer(12)]],
    constant uint& label_height [[buffer(13)]],
    constant uint& gap [[buffer(14)]],
    constant uint& cols [[buffer(15)]],
    uint idx [[thread_position_in_grid]]
) {
    const uint luma_size = out_width * out_height;
    const uint total_size = luma_size + (luma_size >> 1);
    if (idx >= total_size) return;
    if (idx >= luma_size) {
        dst[idx] = 128;
        return;
    }

    const uint y = idx / out_width;
    const uint x = idx - y * out_width;
    uchar out = 0;
    const uint cell_w = frame_width + gap;
    const uint cell_h = label_height + frame_height + gap;
    const uint col = cell_w > 0 ? x / cell_w : 0;
    const uint row = cell_h > 0 ? y / cell_h : 0;
    const uint local_x = x - col * cell_w;
    const uint local_y = y - row * cell_h;
    const uint panel = row * cols + col;

    if (
        panel < n_panels &&
        local_x < frame_width &&
        local_y >= label_height &&
        local_y < label_height + frame_height
    ) {
        uint sx = uint((ulong)local_x * src_width / frame_width);
        uint sy = uint((ulong)(local_y - label_height) * src_height / frame_height);
        sx = min(sx, src_width - 1);
        sy = min(sy, src_height - 1);
        const ulong src_idx =
            (((ulong)panel * n_frames + frame_idx) * src_height + sy) * src_width + sx;
        const float value = (stacks[src_idx] - vmin[panel]) * scale[panel];
        out = uchar(clamp(value, 0.0f, 255.0f));
    }
    dst[idx] = out;
}

kernel void stamp_label(
    device uchar* dst [[buffer(0)]],
    device const uchar* white [[buffer(1)]],
    device const uchar* black [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& mask_width [[buffer(5)]],
    constant uint& mask_height [[buffer(6)]],
    constant uint& label_x [[buffer(7)]],
    constant uint& label_y [[buffer(8)]],
    uint midx [[thread_position_in_grid]]
) {
    const uint total = mask_width * mask_height;
    if (midx >= total) return;
    const uint my = midx / mask_width;
    const uint mx = midx - my * mask_width;
    const uint x = label_x + mx;
    const uint y = label_y + my;
    if (x >= width || y >= height) return;
    const uint out_idx = y * width + x;
    if (black[midx] != 0) dst[out_idx] = 0;
    if (white[midx] != 0) dst[out_idx] = 255;
}
"""


@dataclass(frozen=True)
class LabelMask:
    """One label's white and black uint8 Metal buffers, their size, and the (x, y) pixel they stamp at."""

    x: int
    y: int
    width: int
    height: int
    white: object
    black: object


def _imports() -> tuple[object, object, object, object]:
    """Import Metal and the optional imageio-ffmpeg only when a movie is rendered.

    The module then imports on every platform; a missing Mac, PyObjC Metal or
    Apple GPU raises RuntimeError, which ``is_available`` reports as False.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            f"MPS movie export requires macOS; current platform is {sys.platform}."
        )
    try:
        import Metal
    except ImportError as exc:
        raise RuntimeError("MPS movie export requires pyobjc-framework-Metal.") from exc
    try:
        import imageio_ffmpeg
    except ImportError:
        imageio_ffmpeg = None
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MPS movie export requires an Apple Metal device.")
    return Metal, device, device.newCommandQueue(), imageio_ffmpeg


def is_available() -> bool:
    """Return whether the MPS movie backend can be used in this process."""
    try:
        _imports()
    except RuntimeError:
        return False
    return True


def _buffer(device: object, metal: object, nbytes: int) -> object:
    buffer = device.newBufferWithLength_options_(int(nbytes), metal.MTLResourceStorageModeShared)
    if buffer is None:
        raise MemoryError(f"Metal buffer allocation failed ({int(nbytes) / 1e9:.2f} GB).")
    return buffer


def _buffer_from_array(device: object, metal: object, array: np.ndarray) -> object:
    values = np.ascontiguousarray(array)
    buffer = _buffer(device, metal, values.nbytes)
    view = numpy_view(buffer, values.dtype, values.size).reshape(values.shape)
    view[...] = values
    return buffer


def _uint32(value: int) -> bytes:
    return np.array([int(value)], dtype=np.uint32).tobytes()


def _compile_pipelines(device: object) -> tuple[object, object]:
    options = None
    library, error = device.newLibraryWithSource_options_error_(_METAL_SOURCE, options, None)
    if error:
        raise RuntimeError(f"MPS movie shader compile failed: {error}")
    scale_function = library.newFunctionWithName_("scale_grid_nv12")
    label_function = library.newFunctionWithName_("stamp_label")
    scale_pipeline, error = device.newComputePipelineStateWithFunction_error_(scale_function, None)
    if error:
        raise RuntimeError(f"MPS movie scale pipeline compile failed: {error}")
    label_pipeline, error = device.newComputePipelineStateWithFunction_error_(label_function, None)
    if error:
        raise RuntimeError(f"MPS movie label pipeline compile failed: {error}")
    return scale_pipeline, label_pipeline


def _ffmpeg_exe(imageio_ffmpeg: object | None) -> str:
    if imageio_ffmpeg is None:
        return "ffmpeg"
    return imageio_ffmpeg.get_ffmpeg_exe()


def _encode_nv12(
    raw_path: Path,
    mp4_path: Path,
    *,
    imageio_ffmpeg: object | None,
    width: int,
    height: int,
    fps: float,
    codec: str,
    crf: int,
    quality: int,
    faststart: bool,
) -> None:
    """Encode the raw NV12 frames at ``raw_path`` as H.264 with ffmpeg.

    ``codec="auto"`` tries the VideoToolbox hardware encoder first and falls
    back to libx264 when it fails; an explicit codec gets one attempt.
    """
    ffmpeg = _ffmpeg_exe(imageio_ffmpeg)
    codecs = ["h264_videotoolbox", "libx264"] if codec == "auto" else [codec]
    last_error: subprocess.CalledProcessError | None = None
    for name in codecs:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "-s:v",
            f"{int(width)}x{int(height)}",
            "-r",
            str(max(0.1, float(fps))),
            "-i",
            str(raw_path),
            "-c:v",
            str(name),
        ]
        if name == "libx264":
            command.extend(["-pix_fmt", "yuv420p", "-crf", str(int(crf))])
        elif name == "h264_videotoolbox":
            command.extend(["-b:v", "0", "-q:v", str(int(quality))])
        if faststart:
            command.extend(["-movflags", "+faststart"])
        command.append(str(mp4_path))
        try:
            subprocess.run(command, check=True)
            return
        except FileNotFoundError as exc:
            # A RuntimeError lets save_mp4's "auto" backend fall back to the CPU writer.
            raise RuntimeError(
                "MPS MP4 export needs ffmpeg: pip install imageio-ffmpeg, or put ffmpeg on PATH"
            ) from exc
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if codec != "auto":
                break
    raise RuntimeError(f"ffmpeg failed while writing MPS MP4: {last_error}") from last_error


def save_mp4(
    stacks: Sequence[np.ndarray],
    path: str | Path,
    *,
    labels: Sequence[str] | None,
    fps: float,
    gap: int,
    label_height: int,
    max_width: int | None,
    cols: int | None,
    limits: Sequence[tuple[float, float]],
    crf: int = 18,
    quality: int = 65,
    codec: str = "auto",
    faststart: bool = True,
) -> Path:
    """Save a grayscale movie grid as H.264 MP4 using Apple Metal rendering."""
    metal, device, queue, imageio_ffmpeg = _imports()
    if not stacks:
        raise ValueError("movie.mps.save_mp4 requires at least one stack")
    frames, height, width = stacks[0].shape
    n_panels = len(stacks)
    layout = grid_layout(
        n_panels,
        height,
        width,
        cols=cols,
        gap=gap,
        label_height=label_height,
        max_width=max_width,
    )
    out_width, out_height = layout.width, layout.height

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_stacks = np.stack([np.asarray(stack, dtype=np.float32) for stack in stacks], axis=0)
    # PyObjC never frees a new Metal buffer when its wrapper is collected, so every buffer
    # allocated here is released explicitly, also when rendering or encoding fails.
    buffers = []

    def kept(buffer):
        buffers.append(buffer)
        return buffer

    raw_path = None
    try:
        stack_mtl = kept(_buffer_from_array(device, metal, np.ascontiguousarray(panel_stacks)))
        limits_array = np.asarray(limits, dtype=np.float32)
        vmin_mtl = kept(_buffer_from_array(device, metal, np.ascontiguousarray(limits_array[:, 0])))
        scale = np.asarray(
            [255.0 / max(float(hi) - float(lo), 1e-6) for lo, hi in limits],
            dtype=np.float32,
        )
        scale_mtl = kept(_buffer_from_array(device, metal, scale))
        nv12_bytes = out_width * out_height * 3 // 2
        nv12_mtl = kept(_buffer(device, metal, nv12_bytes))
        nv12_np = numpy_view(nv12_mtl, np.uint8, nv12_bytes)
        scale_pipeline, label_pipeline = _compile_pipelines(device)

        label_masks = []
        for frame_labels in panel_labels(layout, labels, n_panels, frames):
            frame_masks = []
            for label in frame_labels:
                white, black = label_mask(label.text, label_font(label.font_size))
                frame_masks.append(
                    LabelMask(
                        label.x,
                        label.y,
                        white.shape[1],
                        white.shape[0],
                        kept(_buffer_from_array(device, metal, white)),
                        kept(_buffer_from_array(device, metal, black)),
                    )
                )
            label_masks.append(frame_masks)

        block = 256
        grid = metal.MTLSizeMake((nv12_bytes + block - 1) // block, 1, 1)
        threads = metal.MTLSizeMake(block, 1, 1)
        with tempfile.NamedTemporaryFile(suffix=".nv12", delete=False) as raw_file:
            raw_path = Path(raw_file.name)
            for frame_index in range(frames):
                command = queue.commandBuffer()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(scale_pipeline)
                encoder.setBuffer_offset_atIndex_(stack_mtl, 0, 0)
                encoder.setBuffer_offset_atIndex_(vmin_mtl, 0, 1)
                encoder.setBuffer_offset_atIndex_(scale_mtl, 0, 2)
                encoder.setBuffer_offset_atIndex_(nv12_mtl, 0, 3)
                encoder.setBytes_length_atIndex_(_uint32(frame_index), 4, 4)
                encoder.setBytes_length_atIndex_(_uint32(n_panels), 4, 5)
                encoder.setBytes_length_atIndex_(_uint32(frames), 4, 6)
                encoder.setBytes_length_atIndex_(_uint32(width), 4, 7)
                encoder.setBytes_length_atIndex_(_uint32(height), 4, 8)
                encoder.setBytes_length_atIndex_(_uint32(out_width), 4, 9)
                encoder.setBytes_length_atIndex_(_uint32(out_height), 4, 10)
                encoder.setBytes_length_atIndex_(_uint32(layout.frame_width), 4, 11)
                encoder.setBytes_length_atIndex_(_uint32(layout.frame_height), 4, 12)
                encoder.setBytes_length_atIndex_(_uint32(layout.label_height), 4, 13)
                encoder.setBytes_length_atIndex_(_uint32(layout.gap), 4, 14)
                encoder.setBytes_length_atIndex_(_uint32(layout.columns), 4, 15)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
                for mask in label_masks[frame_index]:
                    mask_total = int(mask.width * mask.height)
                    mask_grid = metal.MTLSizeMake((mask_total + block - 1) // block, 1, 1)
                    encoder.setComputePipelineState_(label_pipeline)
                    encoder.setBuffer_offset_atIndex_(nv12_mtl, 0, 0)
                    encoder.setBuffer_offset_atIndex_(mask.white, 0, 1)
                    encoder.setBuffer_offset_atIndex_(mask.black, 0, 2)
                    encoder.setBytes_length_atIndex_(_uint32(out_width), 4, 3)
                    encoder.setBytes_length_atIndex_(_uint32(out_height), 4, 4)
                    encoder.setBytes_length_atIndex_(_uint32(mask.width), 4, 5)
                    encoder.setBytes_length_atIndex_(_uint32(mask.height), 4, 6)
                    encoder.setBytes_length_atIndex_(_uint32(mask.x), 4, 7)
                    encoder.setBytes_length_atIndex_(_uint32(mask.y), 4, 8)
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(mask_grid, threads)
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                # 4 is MTLCommandBufferStatusCompleted.
                status = int(command.status())
                if status != 4:
                    raise RuntimeError(f"MPS movie command failed with Metal status={status}.")
                raw_file.write(nv12_np.tobytes())
        _encode_nv12(
            raw_path,
            path,
            imageio_ffmpeg=imageio_ffmpeg,
            width=out_width,
            height=out_height,
            fps=float(fps),
            codec=str(codec).lower(),
            crf=int(crf),
            quality=int(quality),
            faststart=bool(faststart),
        )
    finally:
        for buffer in buffers:
            release_buffer(buffer)
        if raw_path is not None:
            raw_path.unlink(missing_ok=True)
    return path


__all__ = ["is_available", "save_mp4"]
