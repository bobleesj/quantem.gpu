from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import quantem.gpu as qg
from quantem.gpu import movie
from quantem.gpu.movie import export


def _stack(offset: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[:10, :12].astype(np.float32)
    return np.stack([(xx + frame) + yy * 0.5 + offset for frame in range(3)]).astype(np.float32)


def _nvenc_stack(offset: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    return np.stack([(xx + frame) + yy * 0.5 + offset for frame in range(3)]).astype(np.float32)


def test_movie_module_is_available_from_package() -> None:
    assert qg.movie.save_mp4 is movie.save_mp4


def test_save_gif_accepts_single_stack(tmp_path: Path) -> None:
    out = movie.save_gif(_stack(), tmp_path / "single.gif", fps=6, label_height=0)

    assert out.exists()
    with Image.open(out) as img:
        assert img.is_animated
        assert img.n_frames == 3
        assert img.size == (12, 10)


def test_save_gif_accepts_four_dimensional_data(tmp_path: Path) -> None:
    data = np.stack([_stack(0), _stack(100)], axis=0)

    out = movie.save_gif(data, tmp_path / "grid.gif", labels=["raw", "tv"], cols=2, gap=2, label_height=0)

    assert out.exists()
    with Image.open(out) as img:
        assert img.n_frames == 3
        assert img.size == (12 * 2 + 2, 10)


def test_save_gif_renders_multiline_labels(tmp_path: Path) -> None:
    out = movie.save_gif(
        _stack(),
        tmp_path / "multiline.gif",
        labels=["TV12 n=300 r*=22.33\nλ_s=1.594 λ_t=35.58\nρ_z=-0.000005 χ²/pixel=0.9973"],
        label_height=84,
    )

    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (12, 10 + 84)


def test_save_movie_dispatches_by_suffix(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["frames"] = len(frames)
        captured["fps"] = fps
        captured["crf"] = crf
        path = Path(path)
        path.write_bytes(b"mp4")
        return path

    monkeypatch.setattr(export, "_write_mp4", fake_write_mp4)

    out = movie.save_movie(_stack(), tmp_path / "movie.mp4", fps=7, crf=21, backend="cpu")

    assert out.read_bytes() == b"mp4"
    assert captured == {"frames": 3, "fps": 7.0, "crf": 21}


def test_save_mp4_accepts_rendered_pil_frames(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["sizes"] = [frame.size for frame in frames]
        path = Path(path)
        path.write_bytes(b"mp4")
        return path

    monkeypatch.setattr(export, "_write_mp4", fake_write_mp4)
    frames = [Image.new("RGB", (11, 9), (idx, idx, idx)) for idx in range(2)]

    out = movie.save_mp4(frames, tmp_path / "frames.mp4", fps=12)

    assert out.read_bytes() == b"mp4"
    assert captured["sizes"] == [(11, 9), (11, 9)]


def test_save_mp4_rejects_unknown_backend(tmp_path: Path) -> None:
    try:
        movie.save_mp4(_stack(), tmp_path / "bad.mp4", backend="bad")
    except ValueError as exc:
        assert "unknown movie backend" in str(exc)
        assert "mps" in str(exc)
    else:
        raise AssertionError("save_mp4 should reject unavailable backends")


def test_save_mp4_auto_uses_cuda_backend_when_available(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda

    captured = {}

    def fake_cuda_writer(stacks, path, **kwargs):
        captured["shape"] = stacks[0].shape
        captured["labels"] = kwargs["labels"]
        captured["qp"] = kwargs["qp"]
        path = Path(path)
        path.write_bytes(b"cuda")
        return path

    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "save_mp4", fake_cuda_writer)

    out = movie.save_mp4(
        _nvenc_stack(),
        tmp_path / "auto.mp4",
        labels=["raw"],
        crf=22,
        label_height=0,
        max_width=None,
    )

    assert out.read_bytes() == b"cuda"
    assert captured == {"shape": (3, 256, 256), "labels": ["raw"], "qp": 22}


def test_save_mp4_auto_falls_back_when_cuda_unavailable(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda
    from quantem.gpu.movie import mps

    captured = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["frames"] = len(frames)
        path = Path(path)
        path.write_bytes(b"cpu")
        return path

    monkeypatch.setattr(cuda, "is_available", lambda: False)
    monkeypatch.setattr(mps, "is_available", lambda: False)
    monkeypatch.setattr(export, "_write_mp4", fake_write_mp4)

    out = movie.save_mp4(_stack(), tmp_path / "fallback.mp4")

    assert out.read_bytes() == b"cpu"
    assert captured == {"frames": 3}


def test_cuda_backend_rejects_rendered_frames(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (11, 9), (idx, idx, idx)) for idx in range(2)]

    try:
        movie.save_mp4(frames, tmp_path / "frames.mp4", backend="cuda")
    except ValueError as exc:
        assert "requires array movie data" in str(exc)
    else:
        raise AssertionError("CUDA backend should reject pre-rendered frames")


@pytest.mark.skipif(
    not __import__("quantem.gpu.movie", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA/NVENC movie export is unavailable",
)
def test_cuda_movie_preserves_frame_count_and_order(tmp_path: Path) -> None:
    """NVENC output must retain every scientific frame in acquisition order."""
    import cv2

    source = np.stack(
        [np.full((256, 256), value, dtype=np.float32) for value in range(8)]
    )
    path = movie.save_mp4(
        source,
        tmp_path / "ordered.mp4",
        backend="cuda",
        shared_contrast=False,
        percentile=(0, 100),
        label_height=0,
    )
    capture = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(float(frame[..., 0].mean()))
    capture.release()

    assert len(decoded) == len(source)
    assert np.all(np.diff(decoded) > 0), decoded


@pytest.mark.skipif(
    not __import__("quantem.gpu.movie", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA/NVENC movie export is unavailable",
)
def test_cuda_movie_encodes_each_frame_after_it_is_rendered(tmp_path: Path, monkeypatch) -> None:
    """The encoder copies a frame only after the kernels that render it.

    A rendering kernel delayed by a 10 ms spin stands in for a busy GPU; the
    encoder used to copy the NV12 buffer on its own stream before the kernel
    wrote it, encoding stale frames (every frame black here).
    """
    import cupy as cp
    import cv2

    from quantem.gpu.movie import cuda

    spin = cp.RawKernel(
        r"""extern "C" __global__ void spin(long long cycles) {
            const long long start = clock64();
            while (clock64() - start < cycles) {}
        }""",
        "spin",
    )
    kernels = cuda._kernels

    def delayed_kernels(cp_module):
        scale_grid, stamp_label = kernels(cp_module)

        def delayed_scale_grid(grid, block, args):
            spin((1,), (1,), (np.int64(20_000_000),))
            scale_grid(grid, block, args)

        return delayed_scale_grid, stamp_label

    monkeypatch.setattr(cuda, "_kernels", delayed_kernels)
    source = np.stack([np.full((256, 256), value, dtype=np.float32) for value in range(8)])
    path = movie.save_mp4(
        source, tmp_path / "ordered.mp4", backend="cuda",
        shared_contrast=False, percentile=(0, 100), label_height=0,
    )
    capture = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(float(frame[..., 0].mean()))
    capture.release()

    assert len(decoded) == len(source)
    assert np.all(np.diff(decoded) > 0), decoded


@pytest.mark.skipif(
    not __import__("quantem.gpu.movie", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA/NVENC movie export is unavailable",
)
def test_cuda_movie_reports_an_unencodable_frame_size_as_runtime_error(tmp_path: Path) -> None:
    """NVENC refuses tiny frames; the writer says so as a RuntimeError, not PyNvVideoCodec's own type."""
    with pytest.raises(RuntimeError, match="NVENC could not start"):
        movie.save_mp4(
            _stack(), tmp_path / "tiny.mp4", backend="cuda", label_height=0, max_width=None
        )


def test_save_mp4_auto_falls_back_to_cpu_when_the_gpu_writer_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from quantem.gpu.movie import cuda, mps

    def failing_cuda_writer(stacks, path, **kwargs):
        raise RuntimeError("NVENC could not start")

    def fake_write_mp4(frames, path, fps, *, crf=18):
        path = Path(path)
        path.write_bytes(b"cpu")
        return path

    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "save_mp4", failing_cuda_writer)
    monkeypatch.setattr(mps, "is_available", lambda: False)
    monkeypatch.setattr(export, "_write_mp4", fake_write_mp4)

    out = movie.save_mp4(_nvenc_stack(), tmp_path / "fallback.mp4", label_height=0, max_width=None)

    assert out.read_bytes() == b"cpu"


def test_save_mp4_auto_lets_programming_errors_rise(tmp_path: Path, monkeypatch) -> None:
    """Only encoder failures fall back to the CPU writer; a bug in the GPU writer is reported."""
    from quantem.gpu.movie import cuda

    def broken_cuda_writer(stacks, path, **kwargs):
        raise TypeError("bug in the writer")

    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "save_mp4", broken_cuda_writer)

    with pytest.raises(TypeError, match="bug in the writer"):
        movie.save_mp4(_nvenc_stack(), tmp_path / "broken.mp4", label_height=0, max_width=None)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "mps"])
def test_movie_labels_have_one_size_on_every_backend(
    tmp_path: Path, monkeypatch, backend: str
) -> None:
    """The CPU, CUDA and Metal writers draw labels at one point size (Metal drew 8 points smaller)."""
    from quantem.gpu.movie import cuda, layout, mps

    if backend == "cuda" and not cuda.is_available():
        pytest.skip("CUDA/NVENC movie export is unavailable")
    if backend == "mps" and not mps.is_available():
        pytest.skip("Metal movie export is unavailable")
    sizes = []
    font = layout.label_font

    def recording_font(size):
        sizes.append(size)
        return font(size)

    for module in (export, cuda, mps):
        monkeypatch.setattr(module, "label_font", recording_font)
    if backend == "cpu":
        movie.save_gif(_nvenc_stack(), tmp_path / "labels.gif", labels=["raw"])
    else:
        movie.save_mp4(_nvenc_stack(), tmp_path / "labels.mp4", labels=["raw"], backend=backend)

    # The default 28-pixel label row fits one line of 24-point text.
    assert sizes
    assert set(sizes) == {24}


def test_save_mp4_auto_uses_mps_when_cuda_unavailable(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda
    from quantem.gpu.movie import mps

    captured = {}

    def fake_mps_writer(stacks, path, **kwargs):
        captured["shape"] = stacks[0].shape
        captured["labels"] = kwargs["labels"]
        captured["crf"] = kwargs["crf"]
        captured["quality"] = kwargs["quality"]
        path = Path(path)
        path.write_bytes(b"mps")
        return path

    monkeypatch.setattr(cuda, "is_available", lambda: False)
    monkeypatch.setattr(mps, "is_available", lambda: True)
    monkeypatch.setattr(mps, "save_mp4", fake_mps_writer)

    out = movie.save_mp4(
        _stack(),
        tmp_path / "auto-mps.mp4",
        labels=["raw"],
        crf=20,
        quality=71,
    )

    assert out.read_bytes() == b"mps"
    assert captured == {
        "shape": (3, 10, 12),
        "labels": ["raw"],
        "crf": 20,
        "quality": 71,
    }


def test_mps_backend_rejects_rendered_frames(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (11, 9), (idx, idx, idx)) for idx in range(2)]

    try:
        movie.save_mp4(frames, tmp_path / "frames.mp4", backend="mps")
    except ValueError as exc:
        assert "requires array movie data" in str(exc)
    else:
        raise AssertionError("MPS backend should reject pre-rendered frames")


@pytest.mark.skipif(
    not __import__("quantem.gpu.movie", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA/NVENC movie export is unavailable",
)
def test_cuda_movie_removes_its_elementary_stream_when_encoding_fails(tmp_path: Path, monkeypatch) -> None:
    """A failed NVENC frame leaves no temporary .h264 file behind and keeps the caller's current device."""
    import tempfile

    import cupy as cp

    from quantem.gpu.movie import cuda

    imports = cuda._imports

    def failing_encoder_imports():
        cp_module, imageio_ffmpeg, nvc = imports()

        class FailingNvc:
            NV_ENC_PIC_PARAMS = nvc.NV_ENC_PIC_PARAMS
            PyNvVCException = nvc.PyNvVCException

            @staticmethod
            def CreateEncoder(*args, **kwargs):
                encoder = nvc.CreateEncoder(*args, **kwargs)

                class Failing:
                    def Encode(self, frame, parameters):
                        if parameters.inputTimeStamp == 1:
                            raise RuntimeError("encoder failed on frame 1")
                        return encoder.Encode(frame, parameters)

                    def EndEncode(self):
                        return encoder.EndEncode()

                return Failing()

        return cp_module, imageio_ffmpeg, FailingNvc

    monkeypatch.setattr(cuda, "_imports", failing_encoder_imports)
    current = cp.cuda.Device().id
    before = set(Path(tempfile.gettempdir()).glob("*.h264"))
    with pytest.raises(RuntimeError, match="encoder failed on frame 1"):
        cuda.save_mp4(
            [_nvenc_stack()], tmp_path / "failed.mp4", labels=None, fps=4, gap=0, label_height=0,
            max_width=None, cols=None, limits=[(0.0, 600.0)],
        )
    assert set(Path(tempfile.gettempdir()).glob("*.h264")) == before
    assert cp.cuda.Device().id == current


@pytest.mark.skipif(
    not __import__("quantem.gpu.movie", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA/NVENC movie export is unavailable",
)
def test_cuda_movie_on_another_gpu_restores_the_current_device(tmp_path: Path) -> None:
    """gpu_id selects the encoding device for this call only; Device.use() used to leave it current."""
    import cupy as cp

    from quantem.gpu.movie import cuda

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Requires two CUDA devices")
    with cp.cuda.Device(0):
        cuda.save_mp4(
            [_nvenc_stack()], tmp_path / "other.mp4", labels=None, fps=4, gap=0, label_height=0,
            max_width=None, cols=None, limits=[(0.0, 600.0)], gpu_id=1,
        )
        assert cp.cuda.Device().id == 0


@pytest.mark.skipif(
    not __import__("quantem.gpu.movie", fromlist=["mps"]).mps.is_available(),
    reason="MPS movie export is unavailable",
)
@pytest.mark.parametrize("fail", [False, True])
def test_mps_movie_releases_every_metal_buffer(tmp_path: Path, monkeypatch, fail: bool) -> None:
    """Each Metal buffer save_mp4 allocates is released once, also when encoding fails, and no .nv12 file remains.

    PyObjC never frees a new Metal buffer when its wrapper is collected.
    """
    import tempfile

    from quantem.gpu.movie import mps

    allocated, released = [], []
    buffer, release = mps._buffer, mps.release_buffer

    def counted_buffer(device, metal, nbytes):
        allocated.append(buffer(device, metal, nbytes))
        return allocated[-1]

    def counted_release(value):
        released.append(value)
        release(value)

    monkeypatch.setattr(mps, "_buffer", counted_buffer)
    monkeypatch.setattr(mps, "release_buffer", counted_release)
    if fail:
        def failing_encode(*args, **kwargs):
            raise RuntimeError("encode failed")

        monkeypatch.setattr(mps, "_encode_nv12", failing_encode)
    before = set(Path(tempfile.gettempdir()).glob("*.nv12"))
    arguments = dict(labels=["a", "b"], fps=4, gap=2, label_height=16, max_width=None, cols=None,
                     limits=[(0.0, 600.0), (0.0, 600.0)])
    if fail:
        with pytest.raises(RuntimeError, match="encode failed"):
            mps.save_mp4([_nvenc_stack(), _nvenc_stack(1.0)], tmp_path / "labels.mp4", **arguments)
    else:
        mps.save_mp4([_nvenc_stack(), _nvenc_stack(1.0)], tmp_path / "labels.mp4", **arguments)
    assert len(allocated) > 4 and sorted(map(id, released)) == sorted(map(id, allocated))
    assert set(Path(tempfile.gettempdir()).glob("*.nv12")) == before


@pytest.mark.parametrize(("width", "panels", "max_width"), [(21, 3, 61), (10, 4, 25), (13, 2, None)])
def test_cpu_movie_frames_use_the_gpu_writers_grid(width: int, panels: int, max_width: int | None) -> None:
    """The portable writer's canvas and panel placement equal grid_layout, as the CUDA and MPS writers use.

    It used to round the whole canvas separately from the panels, so a scaled grid could clip its last
    column (61 wide for three 21-pixel panels that need 64) or end one pixel short of the GPU movie.
    """
    from quantem.gpu.movie.layout import grid_layout

    stacks = [np.full((2, 9, width), 100.0 * (index + 1), np.float32) for index in range(panels)]
    frames = export._movie_frames(
        stacks, labels=None, gap=3, label_height=0, max_width=max_width, cols=None,
        shared_contrast=True, ref_stacks=[np.array([0.0, 100.0 * (panels + 1)], np.float32)], percentile=(0.0, 100.0),
    )
    layout = grid_layout(panels, 9, width, cols=None, gap=3, label_height=0, max_width=max_width)
    assert frames[0].size == (layout.width, layout.height)
    pixels = np.asarray(frames[0])[..., 0]
    for index in range(panels):
        row, col = divmod(index, layout.columns)
        top = row * (layout.frame_height + layout.label_height + layout.gap) + layout.label_height
        left = col * (layout.frame_width + layout.gap)
        panel = pixels[top:top + layout.frame_height, left:left + layout.frame_width]
        assert panel.shape == (layout.frame_height, layout.frame_width) and panel.min() > 0


def test_mps_movie_without_ffmpeg_raises_the_error_auto_falls_back_on(tmp_path, monkeypatch):
    """No ffmpeg on PATH and no imageio-ffmpeg: a RuntimeError naming the fix, so backend="auto" uses the CPU writer."""
    from quantem.gpu.movie import mps

    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="needs ffmpeg: pip install imageio-ffmpeg"):
        mps._encode_nv12(tmp_path / "frames.nv12", tmp_path / "movie.mp4", imageio_ffmpeg=None, width=16, height=16,
                         fps=10.0, codec="auto", crf=20, quality=65, faststart=True)
