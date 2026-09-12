"""Expose prepared-series storage through the common loaded-data contract."""

import math
from pathlib import Path

from .models import FourDSTEMData, _release_owned_storage


def load_prepared_series(
    path: Path, *, device: int | str | None, verbose: bool
) -> FourDSTEMData:
    """Load an already validated request without converting the prepared source."""
    from quantem.gpu._compact.load import load

    selected_device = 0 if device is None else int(str(device).removeprefix("cuda:"))
    if verbose:
        print(f"Loading complete encoded series onto cuda:{selected_device}.")
    data = load(path, device=selected_device)
    try:
        logical_bytes = math.prod(data.shape) * data.dtype.itemsize
        metadata = {
            "backend": "cuda",
            "device": f"cuda:{selected_device}",
            "representation": "encoded",
            "residency": "device",
            "resident_codec": "tans",
            "resident_profile": data.storage_format,
            "storage_format": data.storage_format,
            "scan_shape": data.scan_shape,
            "detector_shape": data.det_shape,
            "series_shape": data.series_shape,
            "n_frames": data.n_frames,
            "source_shape": data.shape,
            "working_shape": data.shape,
            "source_dtype": data.dtype.str,
            "working_dtype": data.dtype.name,
            "source_logical_tensor_bytes": logical_bytes,
            "working_logical_tensor_bytes": logical_bytes,
            "physical_resident_bytes": data.nbytes,
            "lossless_exact": True,
            "detector_mask_policy": "preserve-stored-counts",
            "scan_bin": 1,
            "detector_bin": 1,
            "crop": None,
            "valid_pixels": data.valid_pixels,
            "load_seconds": data.load_seconds,
            "load_timing": data.load_timing,
        }
        return FourDSTEMData(data, metadata)
    except BaseException as error:
        _release_owned_storage(data, failure=error)
        raise
