"""Named FP8 encodings; byte storage never identifies a floating format alone."""

import ml_dtypes
import numpy as np

FLOAT8_DTYPES = {
    name: np.dtype(getattr(ml_dtypes, name))
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
}
FORMAT_CODES = {name: index for index, name in enumerate(FLOAT8_DTYPES)}


def is_float8(dtype) -> bool:
    """Recognize the supported NumPy or Torch dtype without casting values."""
    return str(dtype).removeprefix("torch.") in FLOAT8_DTYPES


def torch_dtype(dtype):
    """Resolve a typed FP8 read without silently substituting integer bytes."""
    import torch

    name = str(dtype).removeprefix("torch.")
    result = getattr(torch, name, None)
    if result is None:
        raise NotImplementedError(
            f"This Torch build cannot represent {name}. Install a Torch build "
            "with this FP8 type, or use read(output='native') for original bytes."
        )
    return result


def hdf5_dtype(data) -> np.dtype:
    """Read an explicit FP8 name attached to an HDF5 byte measurement array."""
    name = data.attrs.get("quantem_dtype")
    if name is None:
        return data.dtype
    if isinstance(name, bytes):
        name = name.decode("ascii")
    if not isinstance(name, str) or name not in FLOAT8_DTYPES:
        raise ValueError(
            f"Unsupported quantem_dtype={name!r}; name the original FP8 variant."
        )
    if data.dtype not in (np.dtype("uint8"), np.dtype("V1")):
        raise ValueError(
            "Tagged FP8 HDF5 requires one raw byte per value; do not cast or quantize on load."
        )
    return FLOAT8_DTYPES[name]
