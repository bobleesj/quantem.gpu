"""Shared host admission for exact block-indexed count-rANS arrays."""

import numpy as np


def _host_vector(value: np.ndarray, dtype: str, name: str) -> np.ndarray:
    """Validate an external table without narrowing scientific values."""
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a one-dimensional {dtype} array.")
    return np.ascontiguousarray(array)


def _validate_arrays(
    *,
    shape,
    block_frames,
    scale,
    payload,
    offsets,
    model_ids,
    context_offsets,
    symbols,
    cumulative,
    frequencies,
    literal,
) -> tuple[tuple[int, int, int, int], tuple[np.ndarray, ...]]:
    """Admit only bounded native-uint16 streams and complete probability tables."""
    if len(shape) != 4 or any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
        for size in shape
    ):
        raise ValueError("shape must contain four positive integer dimensions.")
    shape = tuple(map(int, shape))
    if (
        isinstance(block_frames, (bool, np.bool_))
        or not isinstance(block_frames, (int, np.integer))
        or not 1 <= block_frames < 2**32
    ):
        raise ValueError("block_frames must be an integer from 1 to 2**32 - 1.")
    if (
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, np.integer))
        or not 1 <= scale <= 15
    ):
        raise ValueError("rANS scale must be an integer from 1 to 15.")
    scan_count = shape[0] * shape[1]
    detector_count = shape[2] * shape[3]
    if detector_count >= 2**32 or scan_count * detector_count >= 2**63:
        raise ValueError("The declared shape exceeds the supported count range.")
    arrays = tuple(
        _host_vector(value, dtype, name)
        for value, dtype, name in (
            (payload, "uint8", "payload"),
            (offsets, "uint64", "offsets"),
            (model_ids, "uint32", "model_ids"),
            (context_offsets, "uint32", "context_offsets"),
            (symbols, "uint16", "symbols"),
            (cumulative, "uint16", "cumulative"),
            (frequencies, "uint16", "frequencies"),
            (literal, "uint8", "literal"),
        )
    )
    (
        payload,
        offsets,
        model_ids,
        context_offsets,
        symbols,
        cumulative,
        frequencies,
        literal,
    ) = arrays
    block_count = (scan_count + block_frames - 1) // block_frames
    streams = block_count * detector_count
    if len(offsets) != streams + 1 or len(model_ids) != streams:
        raise ValueError(
            "Stream offsets and model selectors disagree with shape and block_frames."
        )
    if (
        offsets[0] != 0
        or offsets[-1] != len(payload)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(
            "Stream offsets must cover the payload once in increasing order."
        )
    if not len(literal) or len(context_offsets) != len(literal) + 1:
        raise ValueError("Each model must have a literal flag and context interval.")
    if np.any(model_ids >= len(literal)) or np.any(literal > 1):
        raise ValueError(
            "Model selectors or literal flags are outside their declared ranges."
        )
    if len(symbols) != len(cumulative) or len(symbols) != len(frequencies):
        raise ValueError(
            "Symbols, cumulative starts, and frequencies must have matching lengths."
        )
    if (
        context_offsets[0] != 0
        or context_offsets[-1] != len(symbols)
        or np.any(context_offsets[1:] < context_offsets[:-1])
    ):
        raise ValueError(
            "Context offsets must cover the model tables once in increasing order."
        )
    for model in range(len(literal)):
        first, stop = map(int, context_offsets[model : model + 2])
        if literal[model]:
            if first != stop:
                raise ValueError(
                    "Literal models must not contain entropy table entries."
                )
            continue
        weights = frequencies[first:stop].astype(np.uint32)
        if not len(weights) or np.any(weights == 0) or int(weights.sum()) != 1 << scale:
            raise ValueError(
                "Every entropy model needs positive frequencies summing to 2**scale."
            )
        expected = np.cumsum(weights, dtype=np.uint32) - weights
        if not np.array_equal(cumulative[first:stop], expected):
            raise ValueError(
                "Cumulative starts must partition all entropy-model slots exactly."
            )
        values = symbols[first:stop]
        if np.any(values[1:] <= values[:-1]):
            raise ValueError("Entropy-model symbols must be unique and increasing.")
    for block in range(block_count):
        first = block * detector_count
        last = first + detector_count
        count = min(block_frames, scan_count - block * block_frames)
        sizes = offsets[first + 1 : last + 1] - offsets[first:last]
        raw = literal[model_ids[first:last]] != 0
        if np.any(sizes[raw] != 2 * count) or np.any(sizes[~raw] < 4):
            raise ValueError(
                "A stream's byte length cannot represent its declared scan block."
            )
    return shape, arrays
