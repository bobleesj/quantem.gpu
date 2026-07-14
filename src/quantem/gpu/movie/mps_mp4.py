"""Apple Metal MP4 rendering backend for :mod:`quantem.gpu.movie`.

This backend renders grayscale movie grids to NV12 with a Metal compute kernel,
then asks ffmpeg to encode H.264. On macOS the default codec is
``h264_videotoolbox`` with a ``libx264`` fallback.
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Sequence


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
class _LabelMask:
    x: int
    y: int
    width: int
    height: int
    white: object
    black: object


def _imports() -> tuple[object, object, object, object]:
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


def _layout(
    n_panels: int,
    height: int,
    width: int,
    *,
    cols: int | None,
    gap: int,
    label_height: int,
    max_width: int | None,
) -> tuple[int, int, int, int, int, int]:
    n_cols = min(n_panels, 3) if cols is None else max(1, int(cols))
    n_rows = int(math.ceil(n_panels / n_cols))
    gap = max(0, int(gap))
    label_height = max(0, int(label_height))
    total_width = n_cols * width + (n_cols - 1) * gap
    cell_height = height + label_height
    total_height = n_rows * cell_height + (n_rows - 1) * gap
    scale = 1.0
    if max_width is not None and total_width > int(max_width):
        scale = int(max_width) / total_width
    frame_width = max(1, int(round(width * scale)))
    frame_height = max(1, int(round(height * scale)))
    gap_scaled = max(0, int(round(gap * scale)))
    label_height_scaled = max(0, int(round(label_height * scale)))
    out_width = n_cols * frame_width + (n_cols - 1) * gap_scaled
    out_height = n_rows * (frame_height + label_height_scaled) + (n_rows - 1) * gap_scaled
    out_width += out_width % 2
    out_height += out_height % 2
    return out_width, out_height, frame_width, frame_height, gap_scaled, label_height_scaled


def _font(label_height: int) -> ImageFont.ImageFont:
    size = max(10, int(label_height) - 8)
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _buffer(device: object, Metal: object, nbytes: int) -> object:
    buf = device.newBufferWithLength_options_(int(nbytes), Metal.MTLResourceStorageModeShared)
    if buf is None:
        raise MemoryError(f"Metal buffer allocation failed ({int(nbytes) / 1e9:.2f} GB).")
    return buf


def _numpy_view(mtl_buf: object, dtype: np.dtype, count: int) -> np.ndarray:
    mv = mtl_buf.contents().as_buffer(mtl_buf.length())
    return np.frombuffer(mv, dtype=dtype, count=count)


def _buffer_from_array(device: object, Metal: object, array: np.ndarray) -> object:
    arr = np.ascontiguousarray(array)
    buf = _buffer(device, Metal, arr.nbytes)
    view = _numpy_view(buf, arr.dtype, arr.size).reshape(arr.shape)
    view[...] = arr
    return buf


def _uint32(value: int) -> bytes:
    return np.array([int(value)], dtype=np.uint32).tobytes()


def _compile_pipelines(device: object) -> tuple[object, object]:
    options = None
    library, err = device.newLibraryWithSource_options_error_(_METAL_SOURCE, options, None)
    if err:
        raise RuntimeError(f"MPS movie shader compile failed: {err}")
    scale_fn = library.newFunctionWithName_("scale_grid_nv12")
    label_fn = library.newFunctionWithName_("stamp_label")
    scale_pipe, err = device.newComputePipelineStateWithFunction_error_(scale_fn, None)
    if err:
        raise RuntimeError(f"MPS movie scale pipeline compile failed: {err}")
    label_pipe, err = device.newComputePipelineStateWithFunction_error_(label_fn, None)
    if err:
        raise RuntimeError(f"MPS movie label pipeline compile failed: {err}")
    return scale_pipe, label_pipe


def _render_label_mask(
    device: object,
    Metal: object,
    text: str,
    x: int,
    y: int,
    font_size: int,
) -> _LabelMask:
    font = _font(font_size)
    probe = Image.new("L", (1, 1), 0)
    left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad = 4
    mask_width = max(1, right - left + 2 * pad)
    mask_height = max(1, bottom - top + 2 * pad)
    white = Image.new("L", (mask_width, mask_height), 0)
    black = Image.new("L", (mask_width, mask_height), 0)
    white_draw = ImageDraw.Draw(white)
    black_draw = ImageDraw.Draw(black)
    text_x = pad - left
    text_y = pad - top
    for dx in (-1, 1):
        for dy in (-1, 1):
            black_draw.text((text_x + dx, text_y + dy), text, font=font, fill=255)
    white_draw.text((text_x, text_y), text, font=font, fill=255)
    white_arr = np.asarray(white, dtype=np.uint8)
    black_arr = np.asarray(black, dtype=np.uint8)
    return _LabelMask(
        x=int(x),
        y=int(y),
        width=int(mask_width),
        height=int(mask_height),
        white=_buffer_from_array(device, Metal, white_arr),
        black=_buffer_from_array(device, Metal, black_arr),
    )


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
    ffmpeg = _ffmpeg_exe(imageio_ffmpeg)
    codecs = ["h264_videotoolbox", "libx264"] if codec == "auto" else [codec]
    last_error: subprocess.CalledProcessError | None = None
    for item in codecs:
        cmd = [
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
            str(item),
        ]
        if item == "libx264":
            cmd.extend(["-pix_fmt", "yuv420p", "-crf", str(int(crf))])
        elif item == "h264_videotoolbox":
            cmd.extend(["-b:v", "0", "-q:v", str(int(quality))])
        if faststart:
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(mp4_path))
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if codec != "auto":
                break
    raise RuntimeError(f"ffmpeg failed while writing MPS MP4: {last_error}") from last_error


def save_mp4(
    stacks: "Sequence[np.ndarray]",
    path: str | Path,
    *,
    labels: "Sequence[str] | None",
    fps: float,
    gap: int,
    label_height: int,
    max_width: int | None,
    cols: int | None,
    limits: "Sequence[tuple[float, float]]",
    crf: int = 18,
    quality: int = 65,
    codec: str = "auto",
    faststart: bool = True,
) -> Path:
    """Save a grayscale movie grid as H.264 MP4 using Apple Metal rendering."""

    Metal, device, queue, imageio_ffmpeg = _imports()
    if not stacks:
        raise ValueError("mps_mp4.save_mp4 requires at least one stack")
    frames, height, width = stacks[0].shape
    n_panels = len(stacks)
    names = (
        [str(item) for item in labels]
        if labels is not None
        else [f"Movie {i + 1}" for i in range(n_panels)]
    )
    n_cols = min(n_panels, 3) if cols is None else max(1, int(cols))
    out_width, out_height, frame_width, frame_height, gap_scaled, label_height_scaled = _layout(
        n_panels,
        height,
        width,
        cols=cols,
        gap=gap,
        label_height=label_height,
        max_width=max_width,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack4 = np.stack([np.asarray(stack, dtype=np.float32) for stack in stacks], axis=0)
    stack_mtl = _buffer_from_array(device, Metal, np.ascontiguousarray(stack4))
    limits_arr = np.asarray(limits, dtype=np.float32)
    vmin_mtl = _buffer_from_array(device, Metal, np.ascontiguousarray(limits_arr[:, 0]))
    scale = np.asarray(
        [255.0 / max(float(hi) - float(lo), 1e-6) for lo, hi in limits],
        dtype=np.float32,
    )
    scale_mtl = _buffer_from_array(device, Metal, scale)
    nv12_bytes = out_width * out_height * 3 // 2
    nv12_mtl = _buffer(device, Metal, nv12_bytes)
    nv12_np = _numpy_view(nv12_mtl, np.uint8, nv12_bytes)
    scale_pipe, label_pipe = _compile_pipelines(device)

    label_masks: list[list[_LabelMask]] = [[] for _ in range(frames)]
    if label_height_scaled > 0:
        font_size = max(8, int(label_height_scaled) - 8)
        for frame_idx in range(frames):
            for panel_idx in range(n_panels):
                row, col = divmod(panel_idx, n_cols)
                x = col * (frame_width + gap_scaled) + 4
                y = row * (frame_height + label_height_scaled + gap_scaled) + 2
                text = f"{names[panel_idx]} [{frame_idx + 1}/{frames}]"
                label_masks[frame_idx].append(
                    _render_label_mask(device, Metal, text, x, y, font_size)
                )

    block = 256
    grid = Metal.MTLSizeMake((nv12_bytes + block - 1) // block, 1, 1)
    threads = Metal.MTLSizeMake(block, 1, 1)
    with tempfile.NamedTemporaryFile(suffix=".nv12", delete=False) as tmp:
        raw_path = Path(tmp.name)
        try:
            for frame_idx in range(frames):
                cmd = queue.commandBuffer()
                enc = cmd.computeCommandEncoder()
                enc.setComputePipelineState_(scale_pipe)
                enc.setBuffer_offset_atIndex_(stack_mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(vmin_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(scale_mtl, 0, 2)
                enc.setBuffer_offset_atIndex_(nv12_mtl, 0, 3)
                enc.setBytes_length_atIndex_(_uint32(frame_idx), 4, 4)
                enc.setBytes_length_atIndex_(_uint32(n_panels), 4, 5)
                enc.setBytes_length_atIndex_(_uint32(frames), 4, 6)
                enc.setBytes_length_atIndex_(_uint32(width), 4, 7)
                enc.setBytes_length_atIndex_(_uint32(height), 4, 8)
                enc.setBytes_length_atIndex_(_uint32(out_width), 4, 9)
                enc.setBytes_length_atIndex_(_uint32(out_height), 4, 10)
                enc.setBytes_length_atIndex_(_uint32(frame_width), 4, 11)
                enc.setBytes_length_atIndex_(_uint32(frame_height), 4, 12)
                enc.setBytes_length_atIndex_(_uint32(label_height_scaled), 4, 13)
                enc.setBytes_length_atIndex_(_uint32(gap_scaled), 4, 14)
                enc.setBytes_length_atIndex_(_uint32(n_cols), 4, 15)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
                for mask in label_masks[frame_idx]:
                    mask_total = int(mask.width * mask.height)
                    mask_grid = Metal.MTLSizeMake((mask_total + block - 1) // block, 1, 1)
                    enc.setComputePipelineState_(label_pipe)
                    enc.setBuffer_offset_atIndex_(nv12_mtl, 0, 0)
                    enc.setBuffer_offset_atIndex_(mask.white, 0, 1)
                    enc.setBuffer_offset_atIndex_(mask.black, 0, 2)
                    enc.setBytes_length_atIndex_(_uint32(out_width), 4, 3)
                    enc.setBytes_length_atIndex_(_uint32(out_height), 4, 4)
                    enc.setBytes_length_atIndex_(_uint32(mask.width), 4, 5)
                    enc.setBytes_length_atIndex_(_uint32(mask.height), 4, 6)
                    enc.setBytes_length_atIndex_(_uint32(mask.x), 4, 7)
                    enc.setBytes_length_atIndex_(_uint32(mask.y), 4, 8)
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(mask_grid, threads)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                status = int(cmd.status())
                if status != 4:
                    raise RuntimeError(f"MPS movie command failed with Metal status={status}.")
                tmp.write(nv12_np.tobytes())
        finally:
            tmp.close()
    try:
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
        raw_path.unlink(missing_ok=True)
    return path


__all__ = ["is_available", "save_mp4"]
