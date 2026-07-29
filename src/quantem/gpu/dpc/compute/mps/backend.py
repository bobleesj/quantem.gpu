"""Metal center-of-mass dispatch."""

from quantem.gpu.detector.compute.mps.kernels import ChunkedFrames


def prepare_frames(data) -> ChunkedFrames:
    """Return a Metal chunk session without copying detector data."""

    return ChunkedFrames(data)
