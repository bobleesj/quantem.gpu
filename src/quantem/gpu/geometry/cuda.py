"""Scan rotation on CUDA: the inverse-mapped dense kernels and the encoded band gather.

The NumPy reference in :mod:`quantem.gpu.geometry.rotation` fixes the
conventions; these kernels compute the same inverse mapping for every output
position in one pass over the CuPy array.
"""

import math
from functools import cache

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.resident.cuda.counts import StreamedCounts

# Decoded frames held at once per output band, and again per source read, while
# rotating an encoded acquisition.
ENCODED_BAND_BYTES = 256 * 1024**2


def rotate_array(
    data,
    output_shape: tuple[int, int],
    angle_degrees: float,
    interpolation: str,
    fill_value: float,
):
    """Rotate a CuPy array's scan plane into ``output_shape`` with one inverse-mapped kernel.

    ``interpolation`` is ``"nearest"``, which keeps the dtype, or
    ``"bilinear"``, which returns float32. Returns a CuPy array.
    """
    source = cp.ascontiguousarray(data)
    source_rows, source_columns = (int(value) for value in source.shape[:2])
    output_rows, output_columns = output_shape
    detector_size = int(np.prod(source.shape[2:], dtype=np.int64))
    angle_radians = math.radians(angle_degrees)
    nearest_kernel, bilinear_kernel = _kernels()
    output_dtype = source.dtype if interpolation == "nearest" else cp.float32
    output = cp.empty((*output_shape, *source.shape[2:]), dtype=output_dtype)
    common = (
        source,
        np.int64(source_rows),
        np.int64(source_columns),
        np.int64(output_columns),
        np.int64(detector_size),
        np.float64(math.cos(angle_radians)),
        np.float64(math.sin(angle_radians)),
        np.float64((source_rows - 1) / 2.0),
        np.float64((source_columns - 1) / 2.0),
        np.float64((output_rows - 1) / 2.0),
        np.float64((output_columns - 1) / 2.0),
    )
    if interpolation == "nearest":
        if source.dtype.kind in "iu" and not float(fill_value).is_integer():
            raise ValueError(
                "fill_value must be an integer when nearest interpolation "
                f"preserves integer data; got {fill_value!r}."
            )
        nearest_kernel(*common, source.dtype.type(fill_value), output)
    else:
        bilinear_kernel(*common, np.float32(fill_value), output)
    return output


def gather_scan_positions(
    dataset: Dataset4dstemGPU,
    index_map: np.ndarray,
    fill_value: int,
) -> StreamedCounts:
    """Build a new encoded acquisition whose frame at each output position is a source frame.

    ``index_map`` holds, per output scan position, the flat source scan index
    or -1 for ``fill_value`` frames. The dense form of an encoded acquisition
    often exceeds GPU memory, so each band of output rows gathers its frames
    from bounded reads of whole source rows and is encoded at once: at most
    ``ENCODED_BAND_BYTES`` of frames per band and per read are ever decoded.
    """
    source = dataset.data
    scan_rows, scan_cols, *detector_shape = dataset.shape
    output_rows, output_cols = index_map.shape[:2]
    frame_bytes = math.prod(detector_shape) * source.dtype.itemsize
    band_rows = max(1, ENCODED_BAND_BYTES // (output_cols * frame_bytes))
    read_rows = max(1, ENCODED_BAND_BYTES // (scan_cols * frame_bytes))
    with cp.cuda.Device(source.device):
        rotated = StreamedCounts(
            (output_rows, output_cols, *detector_shape), source.dtype, source.valid_pixels
        )
        for first in range(0, output_rows, band_rows):
            wanted = index_map[first : first + band_rows].reshape(-1)
            band = cp.full((len(wanted), *detector_shape), fill_value, source.dtype)
            inside = np.flatnonzero(wanted >= 0)
            wanted_rows = wanted[inside] // scan_cols
            for read_first in np.unique(wanted_rows // read_rows) * read_rows:
                read_stop = min(read_first + read_rows, scan_rows)
                take = inside[(wanted_rows >= read_first) & (wanted_rows < read_stop)]
                frames = cp.from_dlpack(
                    dataset.read(scan_region=(read_first, read_stop, 0, scan_cols))
                ).reshape(-1, *detector_shape)
                band[cp.asarray(take)] = frames[cp.asarray(wanted[take] - read_first * scan_cols)]
            rotated.append(band)
    return rotated


@cache
def _kernels():
    """Compile the nearest and bilinear rotation kernels once; CuPy specializes them per dtype."""
    nearest = cp.ElementwiseKernel(
        "raw T source, int64 source_rows, int64 source_columns, "
        "int64 output_columns, int64 detector_size, float64 cosine, "
        "float64 sine, float64 source_center_row, "
        "float64 source_center_column, float64 output_center_row, "
        "float64 output_center_column, T fill_value",
        "T rotated",
        """
            const long long detector_index = i % detector_size;
            const long long output_scan_index = i / detector_size;
            const long long output_row = output_scan_index / output_columns;
            const long long output_column = output_scan_index % output_columns;
            const double row = (double)output_row - output_center_row;
            const double column = (double)output_column - output_center_column;
            const double source_column = cosine * column - sine * row
                + source_center_column;
            const double source_row = sine * column + cosine * row
                + source_center_row;
            const long long nearest_row = llrint(source_row);
            const long long nearest_column = llrint(source_column);
            if (nearest_row >= 0 && nearest_row < source_rows
                    && nearest_column >= 0 && nearest_column < source_columns) {
                const long long source_index =
                    (nearest_row * source_columns + nearest_column) * detector_size
                    + detector_index;
                rotated = source[source_index];
            } else {
                rotated = fill_value;
            }
            """,
        "quantem_rotate_scan_nearest",
    )
    bilinear = cp.ElementwiseKernel(
        "raw T source, int64 source_rows, int64 source_columns, "
        "int64 output_columns, int64 detector_size, float64 cosine, "
        "float64 sine, float64 source_center_row, "
        "float64 source_center_column, float64 output_center_row, "
        "float64 output_center_column, float32 fill_value",
        "float32 rotated",
        """
            const long long detector_index = i % detector_size;
            const long long output_scan_index = i / detector_size;
            const long long output_row = output_scan_index / output_columns;
            const long long output_column = output_scan_index % output_columns;
            const double row = (double)output_row - output_center_row;
            const double column = (double)output_column - output_center_column;
            const double source_column = cosine * column - sine * row
                + source_center_column;
            const double source_row = sine * column + cosine * row
                + source_center_row;
            const long long row0 = (long long)floor(source_row);
            const long long column0 = (long long)floor(source_column);
            const float row_fraction = (float)(source_row - (double)row0);
            const float column_fraction = (float)(source_column - (double)column0);
            float value = fill_value;
            const long long sample_rows[4] = {row0, row0, row0 + 1, row0 + 1};
            const long long sample_columns[4] = {
                column0, column0 + 1, column0, column0 + 1
            };
            const float weights[4] = {
                (1.0f - row_fraction) * (1.0f - column_fraction),
                (1.0f - row_fraction) * column_fraction,
                row_fraction * (1.0f - column_fraction),
                row_fraction * column_fraction
            };
            for (int sample = 0; sample < 4; ++sample) {
                const long long sample_row = sample_rows[sample];
                const long long sample_column = sample_columns[sample];
                if (sample_row >= 0 && sample_row < source_rows
                        && sample_column >= 0 && sample_column < source_columns) {
                    const long long source_index =
                        (sample_row * source_columns + sample_column) * detector_size
                        + detector_index;
                    value += ((float)source[source_index] - fill_value)
                        * weights[sample];
                }
            }
            rotated = value;
            """,
        "quantem_rotate_scan_bilinear",
    )
    return nearest, bilinear
