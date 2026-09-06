"""Small independent inverse recurrence for exact count-rANS conformance.

No production encoder, decoder or model builder is imported here. The frozen
symbol law contains rare uint16 literals, a constant column, a nonuniform model,
multiple blocks and a short tail. It is unit support, not a real-data benchmark.
"""

import numpy as np


def make_fixture(*, dtype="uint16"):
    """Encode known counts independently in the retained column-stream layout."""
    scale = 4
    block_frames = 4
    frames = 15
    detector_count = 6
    counts = np.zeros((frames, detector_count), dtype=np.uint16)
    scan = np.arange(frames)
    counts[:, 0] = scan % 4
    counts[:, 1] = (scan * 3 + 1) % 4
    counts[:, 3] = np.asarray([0, 65535, 256, 32768, 1] * 3, np.uint16)
    counts[:, 4] = (scan // 2) % 4
    counts[:, 5] = (scan + 2) % 4
    if dtype == "uint8":
        counts = (counts % 256).astype(np.uint8)
    parameters = {
        "shape": (3, 5, 2, 3),
        "block_frames": block_frames,
        "scale": scale,
        "context_offsets": np.asarray([0, 4, 5, 5], np.uint32),
        "symbols": np.asarray([0, 1, 2, 3, 0], np.uint16),
        "cumulative": np.asarray([0, 7, 12, 15, 0], np.uint16),
        "frequencies": np.asarray([7, 5, 3, 1, 16], np.uint16),
        "literal": np.asarray([0, 0, 1], np.uint8),
    }
    model_ids, offsets, streams = [], [0], []
    for block_start in range(0, frames, block_frames):
        for pixel, model in enumerate([0, 0, 1, 2, 0, 0]):
            values = counts[block_start : block_start + block_frames, pixel]
            if model == 2:
                stream = values.astype("<u2").tobytes()
            else:
                state, emitted = 1 << 23, []
                for value in values[::-1]:
                    slot = int(value) if model == 0 else 4
                    frequency = int(parameters["frequencies"][slot])
                    cumulative = int(parameters["cumulative"][slot])
                    limit = (((1 << 23) >> scale) << 8) * frequency
                    while state >= limit:
                        emitted.append(state & 255)
                        state >>= 8
                    state = (
                        ((state // frequency) << scale) + state % frequency + cumulative
                    )
                stream = state.to_bytes(4, "little") + bytes(reversed(emitted))
            streams.append(stream)
            offsets.append(offsets[-1] + len(stream))
            model_ids.append(model)
    parameters.update(
        payload=np.frombuffer(b"".join(streams), np.uint8).copy(),
        offsets=np.asarray(offsets, np.uint64),
        model_ids=np.asarray(model_ids, np.uint32),
    )
    return counts.reshape(parameters["shape"]), parameters


def decode_reference(parameters):
    """Decode all fixture counts with explicit scalar arithmetic on the host."""
    shape = parameters["shape"]
    frames = shape[0] * shape[1]
    pixels = shape[2] * shape[3]
    result = np.empty((frames, pixels), dtype=np.uint16)
    payload = parameters["payload"].tobytes()
    scale = parameters["scale"]
    block_frames = parameters["block_frames"]
    for stream, model in enumerate(parameters["model_ids"]):
        start, stop = map(int, parameters["offsets"][stream : stream + 2])
        block, pixel = divmod(stream, pixels)
        first_frame = block * block_frames
        count = min(block_frames, frames - first_frame)
        if parameters["literal"][model]:
            result[first_frame : first_frame + count, pixel] = np.frombuffer(
                payload[start:stop], dtype="<u2"
            )
            continue
        state = int.from_bytes(payload[start : start + 4], "little")
        cursor = start + 4
        first, end = map(int, parameters["context_offsets"][model : model + 2])
        for frame in range(first_frame, first_frame + count):
            slot = state % (1 << scale)
            symbol = next(
                index
                for index in range(first, end)
                if (
                    parameters["cumulative"][index]
                    <= slot
                    < int(parameters["cumulative"][index])
                    + int(parameters["frequencies"][index])
                )
            )
            result[frame, pixel] = parameters["symbols"][symbol]
            state = (
                int(parameters["frequencies"][symbol]) * (state >> scale)
                + slot
                - int(parameters["cumulative"][symbol])
            )
            while state < 1 << 23:
                if cursor >= stop:
                    raise ValueError(
                        "A frozen fixture stream ended before normalization."
                    )
                state = state * 256 + payload[cursor]
                cursor += 1
        if cursor != stop or state != 1 << 23:
            raise ValueError("A frozen fixture stream did not terminate exactly.")
    return result.reshape(shape)
