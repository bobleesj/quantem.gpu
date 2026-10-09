"""CUDA/NVENC MP4 export kernels for :mod:`quantem.gpu.movie`.

This module is intentionally optional. It imports CUDA/NVENC dependencies only
when the caller selects ``backend="cuda"`` or when ``backend="auto"`` probes
for support.
"""

import ast
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantem.gpu.movie.layout import grid_layout, label_font, label_mask, panel_labels


@dataclass
class LabelMask:
    """One label's white and black uint8 masks on the GPU and the (x, y) pixel they stamp at."""

    x: int
    y: int
    white: object
    black: object


class CudaArrayView:
    """Expose a CuPy array through CUDA Array Interface for PyNvVideoCodec."""

    def __init__(self, array: object) -> None:
        self._array = array
        self.__cuda_array_interface__ = array.__cuda_array_interface__


class Nv12Frame:
    """PyNvVideoCodec-compatible NV12 frame backed by one CuPy allocation."""

    def __init__(self, nv12: object) -> None:
        height = nv12.shape[0] * 2 // 3
        width = nv12.shape[1]
        self._nv12 = nv12
        self._planes = [
            CudaArrayView(nv12[:height, :, None]),
            CudaArrayView(nv12[height:, :].reshape(height // 2, width // 2, 2)),
        ]

    def cuda(self) -> list[CudaArrayView]:
        return self._planes


def _imports() -> tuple[object, object, object]:
    """Import CuPy, imageio-ffmpeg and PyNvVideoCodec, or raise RuntimeError saying what is missing.

    :func:`is_available` probes with this, so a missing package or GPU reads
    as unavailable instead of failing the caller.
    """
    # PyNvVideoCodec 2.2 still uses ast.Str, which Python 3.14 removed.
    if sys.version_info >= (3, 14):
        ast.Str = ast.Constant
    try:
        import cupy as cp
        import imageio_ffmpeg
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise RuntimeError(
            "CUDA movie export requires cupy, imageio-ffmpeg, and "
            "PyNvVideoCodec. Use backend='cpu' or install the NVIDIA "
            "movie dependencies in a CUDA environment."
        ) from exc
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("CUDA movie export requires an NVIDIA CUDA device.")
    return cp, imageio_ffmpeg, nvc


def is_available() -> bool:
    """Return whether the CUDA/NVENC backend can be used in this process."""
    try:
        _imports()
    except RuntimeError:
        return False
    return True


def _kernels(cp: object) -> tuple[object, object]:
    """Compile the two kernels that draw one NV12 grid frame on the GPU.

    ``scale_grid_nv12`` writes the luma plane: each pixel of a panel samples
    its movie by nearest neighbor and maps ``(value - vmin) * scale`` to
    0..255, label rows and gaps stay black, and the chroma plane is the
    neutral 128. ``stamp_label`` then draws one label's black outline and white
    text masks into the luma plane. The frame never leaves the GPU before
    NVENC encodes it.
    """
    scale_grid = cp.RawKernel(
        r'''
        extern "C" __global__
        void scale_grid_nv12(
            const float* const* __restrict__ stacks,
            const float* __restrict__ vmin,
            const float* __restrict__ scale,
            unsigned char* __restrict__ dst,
            const int n_panels,
            const int src_width,
            const int src_height,
            const int frame_stride,
            const int out_width,
            const int out_height,
            const int frame_width,
            const int frame_height,
            const int label_height,
            const int gap,
            const int cols
        ) {
            const int idx = blockDim.x * blockIdx.x + threadIdx.x;
            const int luma_size = out_width * out_height;
            const int total_size = luma_size + (luma_size >> 1);
            if (idx >= total_size) {
                return;
            }
            if (idx >= luma_size) {
                dst[idx] = 128;
                return;
            }

            const int y = idx / out_width;
            const int x = idx - y * out_width;
            unsigned char out = 0;
            const int cell_w = frame_width + gap;
            const int cell_h = label_height + frame_height + gap;
            const int col = cell_w > 0 ? x / cell_w : 0;
            const int row = cell_h > 0 ? y / cell_h : 0;
            const int local_x = x - col * cell_w;
            const int local_y = y - row * cell_h;
            const int panel = row * cols + col;

            if (
                panel >= 0 && panel < n_panels &&
                local_x >= 0 && local_x < frame_width &&
                local_y >= label_height &&
                local_y < label_height + frame_height
            ) {
                int sx = (int)(((long long)local_x * src_width) / frame_width);
                int sy = (int)(((long long)(local_y - label_height) * src_height) / frame_height);
                sx = min(max(sx, 0), src_width - 1);
                sy = min(max(sy, 0), src_height - 1);
                const float value = (stacks[panel][sy * src_width + sx] - vmin[panel]) * scale[panel];
                const float clipped = fminf(fmaxf(value, 0.0f), 255.0f);
                out = (unsigned char)(clipped);
            }
            dst[idx] = out;
        }
        ''',
        "scale_grid_nv12",
    )
    stamp_label = cp.RawKernel(
        r'''
        extern "C" __global__
        void stamp_label(
            unsigned char* __restrict__ dst,
            const int width,
            const int height,
            const unsigned char* __restrict__ white,
            const unsigned char* __restrict__ black,
            const int mask_width,
            const int mask_height,
            const int label_x,
            const int label_y
        ) {
            const int midx = blockDim.x * blockIdx.x + threadIdx.x;
            const int total = mask_width * mask_height;
            if (midx >= total) {
                return;
            }
            const int my = midx / mask_width;
            const int mx = midx - my * mask_width;
            const int x = label_x + mx;
            const int y = label_y + my;
            if (x < 0 || x >= width || y < 0 || y >= height) {
                return;
            }
            const int out_idx = y * width + x;
            if (black[midx] != 0) {
                dst[out_idx] = 0;
            }
            if (white[midx] != 0) {
                dst[out_idx] = 255;
            }
        }
        ''',
        "stamp_label",
    )
    return scale_grid, stamp_label


def _ffmpeg_mux_command(
    imageio_ffmpeg: object,
    elementary_path: Path,
    mp4_path: Path,
    fps: float,
    *,
    faststart: bool,
) -> list[str]:
    """Return the ffmpeg command that wraps NVENC's raw H.264 stream in an MP4 container.

    NVENC emits an elementary stream without timestamps, so ffmpeg generates
    them at ``fps`` and copies the video without re-encoding;
    ``faststart`` moves the index to the front so players can start early.
    """
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-f",
        "h264",
        "-r",
        str(max(0.1, float(fps))),
        "-i",
        str(elementary_path),
        "-c:v",
        "copy",
        "-an",
    ]
    if faststart:
        command.extend(["-movflags", "+faststart"])
    command.append(str(mp4_path))
    return command


def _packet_bytes(packets: list[dict[str, object]]) -> bytes:
    return b"".join(bytes(packet["data"]) for packet in packets)


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
    qp: int = 18,
    preset: str = "P3",
    tuning_info: str = "high_quality",
    gpu_id: int = 0,
    faststart: bool = True,
) -> Path:
    """Save a grayscale movie grid as H.264 MP4 using NVIDIA NVENC.

    Raises
    ------
    RuntimeError
        If the CUDA movie dependencies are missing, NVENC cannot encode
        frames of this size, or ffmpeg cannot write the MP4.
    """
    cp, imageio_ffmpeg, nvc = _imports()
    if not stacks:
        raise ValueError("movie.cuda.save_mp4 requires at least one stack")
    # A device context, not Device.use(): the caller's current device is restored on return.
    with cp.cuda.Device(int(gpu_id)):
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
        # scale_grid_nv12 reads float32 frames, minimums and scales.
        device_stacks = [cp.asarray(np.asarray(stack, dtype=np.float32)) for stack in stacks]
        vmin = cp.asarray([low for low, _ in limits], dtype=cp.float32)
        scale = cp.asarray([255.0 / max(high - low, 1e-6) for low, high in limits], dtype=cp.float32)
        label_masks = [
            [
                LabelMask(
                    label.x,
                    label.y,
                    *(cp.asarray(mask) for mask in label_mask(label.text, label_font(label.font_size))),
                )
                for label in frame_labels
            ]
            for frame_labels in panel_labels(layout, labels, n_panels, frames)
        ]

        scale_grid, stamp_label = _kernels(cp)
        nv12_buffers = [
            cp.empty((out_height + out_height // 2, out_width), dtype=cp.uint8)
            for _ in range(4)
        ]
        nv12_frames = [Nv12Frame(buffer) for buffer in nv12_buffers]
        config = {
            "codec": "h264",
            "gpu_id": int(gpu_id),
            "preset": str(preset).upper(),
            "tuning_info": str(tuning_info),
            "rc": "constqp",
            "qp": str(int(qp)),
            # The elementary-stream mux path does not carry reordering metadata.
            # Scientific frame sequences therefore require decode order to match
            # acquisition order exactly.
            "bf": "0",
            "fps": max(0.1, float(fps)),
        }
        # The encoder copies each NV12 frame on this stream, after the kernels
        # that render it and before the kernels that render the frame reusing its
        # buffer; on its own stream the copy could run first and encode the stale
        # frame from four frames earlier under GPU load. A blocking stream also
        # waits for the uploads above, made on the default stream.
        stream = cp.cuda.Stream()
        try:
            encoder = nvc.CreateEncoder(
                out_width, out_height, "NV12", False,
                cudacontext=int(cp.cuda.driver.ctxGetCurrent()), cudastream=stream.ptr, **config,
            )
        except nvc.PyNvVCException as exc:
            # NVENC refuses some frame sizes and runs out of sessions only here;
            # a RuntimeError lets backend="auto" fall back to the CPU writer.
            raise RuntimeError(
                f"NVENC could not start an H.264 encoder for {out_width} x {out_height} frames: {exc}"
            ) from exc
        block = 256
        total = out_width * out_height * 3 // 2
        grid = ((total + block - 1) // block,)

        # The elementary stream is a temporary file; it is removed whether encoding or muxing fails or not.
        elementary_path = None
        try:
            with stream, tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as elementary_stream:
                elementary_path = Path(elementary_stream.name)
                for frame_index in range(frames):
                    ring_index = frame_index % len(nv12_buffers)
                    frame_pointers = cp.asarray(
                        [stack[frame_index].data.ptr for stack in device_stacks],
                        dtype=cp.uintp,
                    )
                    scale_grid(
                        grid,
                        (block,),
                        (
                            frame_pointers,
                            vmin,
                            scale,
                            nv12_buffers[ring_index],
                            np.int32(n_panels),
                            np.int32(width),
                            np.int32(height),
                            np.int32(width * height),
                            np.int32(out_width),
                            np.int32(out_height),
                            np.int32(layout.frame_width),
                            np.int32(layout.frame_height),
                            np.int32(layout.label_height),
                            np.int32(layout.gap),
                            np.int32(layout.columns),
                        ),
                    )
                    for mask in label_masks[frame_index]:
                        mask_height, mask_width = mask.white.shape
                        mask_grid = ((mask_height * mask_width + block - 1) // block,)
                        stamp_label(
                            mask_grid,
                            (block,),
                            (
                                nv12_buffers[ring_index],
                                np.int32(out_width),
                                np.int32(out_height),
                                mask.white,
                                mask.black,
                                np.int32(mask_width),
                                np.int32(mask_height),
                                np.int32(mask.x),
                                np.int32(mask.y),
                            ),
                        )
                    picture_params = nvc.NV_ENC_PIC_PARAMS()
                    picture_params.inputTimeStamp = frame_index
                    elementary_stream.write(_packet_bytes(encoder.Encode(nv12_frames[ring_index], picture_params)))
                elementary_stream.write(_packet_bytes(encoder.EndEncode()))
            command = _ffmpeg_mux_command(imageio_ffmpeg, elementary_path, path, fps, faststart=faststart)
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed while muxing NVENC MP4: {exc}") from exc
        finally:
            if elementary_path is not None:
                elementary_path.unlink(missing_ok=True)
        stream.synchronize()
    return path
