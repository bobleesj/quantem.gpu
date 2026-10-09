"""Scaled-precision conversion primitives, dispatched to the CUDA or Metal implementation.

Precision export and loading convert blocks that already sit on one GPU: a
CuPy array or Torch CUDA tensor, or a Metal staging ``MetalArray`` or Torch
MPS tensor. These three steps (restore intensities, encode codes, measure the
error) pick the kernel for the block's backend, so ``io.precision`` streams
blocks without knowing which device holds them.
"""

from quantem.gpu.resident.cuda.precision import (
    encode_scaled_uint16,
    measure_scaled_uint16,
)
from quantem.gpu.resident.mps import precision as metal
from quantem.gpu.resident.mps.arrays import MetalArray


def restore(values, report, backend: str):
    """Restore float32 scientific units from stored codes, or pass float32 intensities through.

    A version-2 (regional) report has one calibration per region, so a block
    carrying it must already have been restored by the regional reader.
    """
    if report and report.get("version") == 2 and "regions" in report:
        if str(values.dtype).removeprefix("torch.") != "float32":
            raise ValueError("Regional codes require their per-frame calibration; use the regional reader.")
        report = None  # The regional reader has already restored this block.
    if backend == "mps":
        if isinstance(values, MetalArray):
            return metal.restore(values, report)
        return metal.tensor_restore(values, report)
    import cupy as cp

    if not isinstance(values, cp.ndarray):
        values = cp.from_dlpack(values.detach())
    if report and report["storage"] == "scaled_uint16":
        # The calibration applies in float64 and rounds once to float32.
        return (values.astype(cp.float64) * report["scale"] + report["offset"]).astype(
            cp.float32
        )
    return values.astype(cp.float32, copy=False)


def encode(values, report, backend: str):
    """Convert float32 intensities to the requested stored codes (float16 or scaled uint16)."""
    if backend == "mps":
        if isinstance(values, MetalArray):
            return metal.encode(values, report)
        return metal.tensor_encode(values, report)
    import cupy as cp

    if report["storage"] == "float16":
        return values.astype(cp.float16)

    return encode_scaled_uint16(values, report)


def measure(original, restored, report, backend: str, *, encoded=None) -> None:
    """Accumulate the encoding error of one block into ``report``.

    For scaled uint16 on CUDA the caller passes the codes as ``encoded`` and
    no ``restored`` block: one reduction computes every report field without
    materializing a restored or difference array.
    """
    if backend == "mps":
        if isinstance(original, MetalArray):
            metal.measure(original, restored, report)
        else:
            metal.tensor_measure(original, restored, report)
        return
    import cupy as cp

    if report.get("storage") == "scaled_uint16" and encoded is not None:
        measure_scaled_uint16(original, encoded, report)
        return
    # A float64 difference keeps float32 rounding out of the measured error.
    difference = restored.astype(cp.float64) - original.astype(cp.float64)
    report["values"] += original.size
    report["squared_error"] += float(cp.sum(difference * difference).get())
    report["max_abs_error"] = max(
        report["max_abs_error"], float(cp.max(cp.abs(difference)).get())
    )
    report["positive_to_zero"] += int(
        cp.count_nonzero((original > 0) & (restored == 0)).get()
    )
    report["changed"] += int(cp.count_nonzero(original != restored).get())
    report["overflow"] += int(cp.count_nonzero(~cp.isfinite(restored)).get())
