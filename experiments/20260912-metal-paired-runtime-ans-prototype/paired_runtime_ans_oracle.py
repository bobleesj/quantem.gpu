"""Independent CPU oracle for the proposed exact paired runtime tANS ABI.

This is deliberately standalone experiment code. It does not import a CUDA,
Metal, application, or current runtime-ANS implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


MODELS = 32
STATES = 1024
SYMBOLS = 1089
ESCAPE = 1088
INTERVAL = 512


class InvalidStream(ValueError):
    """Raised when encoded bytes violate the proposed ABI."""


@dataclass(frozen=True)
class EncodedStream:
    """One exact encoded detector-column stream."""

    mode: int
    payload: bytes
    transitions: int


def _rank_descending(values: np.ndarray) -> np.ndarray:
    """Return deterministic descending ranks, breaking ties by symbol index."""

    order = np.lexsort((np.arange(values.size), -values))
    ranks = np.empty(values.size, np.int64)
    ranks[order] = np.arange(values.size)
    return ranks


def build_frequencies() -> np.ndarray:
    """Build the fixed 32-model, 1024-state pair frequency table."""

    means = np.exp(np.linspace(np.log(0.002), np.log(32.0), MODELS))
    choices = np.asarray([4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768])
    k = np.arange(33, dtype=np.float64)
    factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, 33)))))
    probability = np.exp(k[None, :] * np.log(means[:, None]) - factorial[None, :] - means[:, None])
    a = np.arange(SYMBOLS) // 33
    b = np.arange(SYMBOLS) % 33
    joint = np.where((a < 32) & (b < 32), probability[:, a] * probability[:, b], 0.0)
    ranking = np.stack([_rank_descending(row) for row in joint])

    candidates: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    for choice in choices:
        supported = ranking < choice
        supported[:, ESCAPE] = True
        selected = np.where(supported, joint, 0.0)
        selected[:, ESCAPE] = np.maximum(0.0, 1.0 - selected.sum(axis=1))
        count = supported.sum(axis=1)
        allocation = selected * (STATES - count)[:, None]
        frequency = np.where(supported, np.floor(allocation).astype(np.uint32) + 1, 0)
        remainder = STATES - frequency.sum(axis=1, dtype=np.int64)
        fractions = np.where(supported, allocation - np.floor(allocation), -np.inf)
        for model in range(MODELS):
            ranks = _rank_descending(fractions[model])
            frequency[model] += (ranks < remainder[model]).astype(np.uint32)
        candidates.append(frequency)
        cost = (
            selected
            * np.log2(STATES / np.maximum(frequency, np.uint32(1)))
        ).sum(axis=1) + 13.0 * selected[:, ESCAPE]
        costs.append(cost)

    all_frequencies = np.stack(candidates)
    choice_for_model = np.argmin(np.stack(costs), axis=0)
    result = all_frequencies[choice_for_model, np.arange(MODELS)]
    if not np.all(result.sum(axis=1, dtype=np.uint64) == STATES):
        raise AssertionError("Every tANS model must contain exactly 1024 states")
    return result


def build_tables(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build encoding states and packed decoding transitions."""

    if frequencies.shape != (MODELS, SYMBOLS) or frequencies.dtype != np.uint32:
        raise ValueError("Expected uint32 frequencies with shape (32, 1089)")
    encoding = np.zeros((MODELS, STATES), np.uint16)
    decoding = np.full((MODELS, STATES), np.uint32(0xFFFF_FFFF), np.uint32)
    for model in range(MODELS):
        starts = np.cumsum(frequencies[model], dtype=np.uint32) - frequencies[model]
        for symbol in np.flatnonzero(frequencies[model]):
            frequency = int(frequencies[model, symbol])
            start = int(starts[symbol])
            rank = 0
            for state in range(STATES):
                sequence = (state * 43) & (STATES - 1)
                if not start <= sequence < start + frequency:
                    continue
                n = frequency + rank
                bits = 10 - (n.bit_length() - 1)
                base = (n << bits) - STATES
                pair = 4095 if symbol == ESCAPE else symbol // 33 | ((symbol % 33) << 6)
                decoding[model, state] = np.uint32(pair | (bits << 12) | (base << 16))
                encoding[model, start + rank] = np.uint16(state)
                rank += 1
            if rank != frequency:
                raise AssertionError("Frequency table did not assign every encoding rank")
    if np.any(decoding == np.uint32(0xFFFF_FFFF)):
        raise AssertionError("Decoding table contains an unassigned state")
    return encoding, decoding


def _model(values: np.ndarray) -> int:
    mean = np.float32(np.minimum(values.astype(np.uint32), 32).sum(dtype=np.uint64) / values.size)
    scaled = np.float32(
        (np.log(max(mean, np.float32(0.002))) - np.log(np.float32(0.002)))
        * np.float32((MODELS - 1) / math.log(16000.0))
    )
    return int(np.clip(np.rint(scaled), 0, MODELS - 1))


def _append_bits(state: list[int | bytearray], value: int, count: int) -> None:
    if count == 0:
        return
    if count < 0 or value < 0 or value >= 1 << count:
        raise AssertionError("Bit-stack append is out of range")
    buffer, available, output = int(state[0]), int(state[1]), state[2]
    assert isinstance(output, bytearray)
    buffer |= value << available
    available += count
    while available >= 8:
        output.append(buffer & 255)
        buffer >>= 8
        available -= 8
    state[0], state[1] = buffer, available


def encode_stream(
    values: np.ndarray,
    frequencies: np.ndarray,
    encoding: np.ndarray,
) -> EncodedStream:
    """Encode one native count stream without changing any value."""

    values = np.asarray(values)
    if values.ndim != 1 or values.size < 1 or values.size > INTERVAL:
        raise ValueError("Encode one nonempty stream of at most 512 counts")
    if values.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("The logical stream dtype must be uint8 or uint16")
    native = values.astype(np.uint16, copy=False)
    low, high = int(native.min()), int(native.max())
    nonzero = int(np.count_nonzero(native))
    if low == high:
        if high == 0:
            return EncodedStream(253, b"", 0)
        return EncodedStream(255, high.to_bytes(2, "little"), 0)
    if high <= 128 and nonzero <= 2:
        events = bytearray()
        for scan, count in enumerate(native):
            if count:
                event = (scan << 7) | (int(count) - 1)
                events += event.to_bytes(2, "little")
        return EncodedStream(252, bytes(events), 0)

    model = _model(native)
    starts = np.cumsum(frequencies[model], dtype=np.uint32) - frequencies[model]
    state_value = 0
    total_bits = 0
    stack: list[int | bytearray] = [0, 0, bytearray()]
    pairs = (native.size + 1) // 2
    for pair_index in range(pairs - 1, -1, -1):
        index = pair_index * 2
        a = int(native[index])
        b = int(native[index + 1]) if index + 1 < native.size else 0
        symbol = a * 33 + b if a < 32 and b < 32 else ESCAPE
        frequency = int(frequencies[model, symbol])
        if symbol == ESCAPE or frequency == 0:
            symbol = ESCAPE
            frequency = int(frequencies[model, ESCAPE])
            if a < 64 and b < 64:
                _append_bits(stack, a | (b << 6), 13)
                total_bits += 13
            else:
                _append_bits(stack, b, 16)
                _append_bits(stack, a, 16)
                _append_bits(stack, 4096, 13)
                total_bits += 45
        start = int(starts[symbol])
        y = STATES + state_value
        bits = 10 - (frequency.bit_length() - 1)
        if y < frequency << bits:
            bits -= 1
        rank = (y >> bits) - frequency
        emitted = y & ((1 << bits) - 1) if bits else 0
        _append_bits(stack, emitted, bits)
        total_bits += bits
        state_value = int(encoding[model, start + rank])

    buffer, available, body = int(stack[0]), int(stack[1]), stack[2]
    assert isinstance(body, bytearray)
    if available:
        body.append(buffer)
    use_entropy = len(body) + 2 < 2 * native.size and total_bits < 16384
    size = len(body) + 2 if use_entropy else 2 * native.size
    if high <= 128 and 2 * nonzero <= size + 1:
        events = bytearray()
        for scan, count in enumerate(native):
            if count:
                event = (scan << 7) | (int(count) - 1)
                events += event.to_bytes(2, "little")
        return EncodedStream(252, bytes(events), 0)
    if not use_entropy:
        return EncodedStream(254, native.astype("<u2", copy=False).tobytes(), 0)
    header = (state_value << 6) | (total_bits & 7)
    return EncodedStream(64 + model, header.to_bytes(2, "little") + body, pairs)


def decode_stream(
    encoded: EncodedStream,
    count: int,
    logical_dtype: np.dtype,
    decoding: np.ndarray,
) -> np.ndarray:
    """Decode one stream and reject malformed terminal state or extents."""

    dtype = np.dtype(logical_dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")) or not 1 <= count <= INTERVAL:
        raise ValueError("Decode 1...512 values to uint8 or uint16")
    mode, payload = encoded.mode, encoded.payload
    values = np.zeros(count, np.uint16)
    if mode == 253:
        if payload:
            raise InvalidStream("All-zero mode must have no payload")
    elif mode == 255:
        if len(payload) != 2:
            raise InvalidStream("Constant mode must contain one uint16")
        values.fill(int.from_bytes(payload, "little"))
    elif mode == 254:
        if len(payload) != 2 * count:
            raise InvalidStream("Literal mode extent differs from the count")
        values[:] = np.frombuffer(payload, "<u2")
    elif mode == 252:
        if len(payload) % 2:
            raise InvalidStream("Sparse event bytes must be uint16 aligned")
        previous = -1
        for offset in range(0, len(payload), 2):
            event = int.from_bytes(payload[offset : offset + 2], "little")
            scan, value = event >> 7, (event & 127) + 1
            if scan <= previous or scan >= count:
                raise InvalidStream("Sparse positions must be strictly increasing and in range")
            values[scan] = value
            previous = scan
    elif 64 <= mode < 64 + MODELS:
        if len(payload) < 2:
            raise InvalidStream("Paired tANS stream is missing its header")
        header = int.from_bytes(payload[:2], "little")
        tail = header & 7
        if (header >> 3) & 7:
            raise InvalidStream("Reserved paired tANS header bits are nonzero")
        state_value = header >> 6
        body = payload[2:]
        if state_value >= STATES or (tail and not body):
            raise InvalidStream("Paired tANS initial state or tail is invalid")
        meaningful = len(body) * 8 - ((8 - tail) & 7)
        bits_value = int.from_bytes(body, "little")
        if meaningful < 0 or bits_value >> meaningful:
            raise InvalidStream("Paired tANS padding bits are nonzero")

        def pop(bits: int) -> int:
            nonlocal meaningful
            if bits < 0 or meaningful < bits:
                raise InvalidStream("Paired tANS bit stack is truncated")
            meaningful -= bits
            return (bits_value >> meaningful) & ((1 << bits) - 1) if bits else 0

        model = mode - 64
        for index in range(0, count, 2):
            code = int(decoding[model, state_value])
            pair, bits, base = code & 4095, (code >> 12) & 15, code >> 16
            state_value = base + pop(bits)
            if pair == 4095:
                word = pop(13)
                if word < 4096:
                    a, b = word & 63, word >> 6
                elif word == 4096:
                    a, b = pop(16), pop(16)
                else:
                    raise InvalidStream("Reserved paired tANS escape marker")
            else:
                a, b = pair & 63, pair >> 6
            values[index] = a
            if index + 1 < count:
                values[index + 1] = b
            elif b:
                raise InvalidStream("Odd stream decoded a nonzero padded count")
        if meaningful != 0 or state_value != 0:
            raise InvalidStream("Paired tANS stream has a nonterminal state or trailing bits")
    else:
        raise InvalidStream(f"Unknown paired runtime mode {mode}")
    if dtype == np.dtype("uint8") and np.any(values > 255):
        raise InvalidStream("Decoded values exceed the declared uint8 logical dtype")
    return values.astype(dtype)


def validate_transition_tables(
    frequencies: np.ndarray,
    encoding: np.ndarray,
    decoding: np.ndarray,
) -> int:
    """Exhaustively prove every supported encoder step inverts through its table."""

    checked = 0
    for model in range(MODELS):
        starts = np.cumsum(frequencies[model], dtype=np.uint32) - frequencies[model]
        for symbol in np.flatnonzero(frequencies[model]):
            frequency = int(frequencies[model, symbol])
            start = int(starts[symbol])
            expected_pair = 4095 if symbol == ESCAPE else symbol // 33 | ((symbol % 33) << 6)
            for old_state in range(STATES):
                y = STATES + old_state
                bits = 10 - (frequency.bit_length() - 1)
                if y < frequency << bits:
                    bits -= 1
                rank = (y >> bits) - frequency
                low = y & ((1 << bits) - 1) if bits else 0
                new_state = int(encoding[model, start + rank])
                code = int(decoding[model, new_state])
                if (code & 4095) != expected_pair or ((code >> 12) & 15) != bits:
                    raise AssertionError("Encoding and decoding tables disagree on symbol or bit count")
                if (code >> 16) + low != old_state:
                    raise AssertionError("Encoding and decoding state transitions are not inverse")
                checked += 1
    return checked


def run_oracle() -> dict[str, object]:
    """Run deterministic exactness and transition-count checks."""

    frequencies = build_frequencies()
    encoding, decoding = build_tables(frequencies)
    transition_cases = validate_transition_tables(frequencies, encoding, decoding)
    cases: list[tuple[str, np.ndarray]] = [
        ("u8-zero", np.zeros(INTERVAL, np.uint8)),
        ("u8-constant", np.full(INTERVAL, 255, np.uint8)),
        ("u16-constant-sentinel", np.full(INTERVAL, 65535, np.uint16)),
        ("u16-sparse", np.asarray([0] * 13 + [128] + [0] * 497 + [1], np.uint16)),
    ]
    for dtype in (np.uint8, np.uint16):
        rng = np.random.default_rng(20260912 + np.dtype(dtype).itemsize)
        cases.append((f"{np.dtype(dtype).name}-poisson", rng.poisson(2.0, INTERVAL).astype(dtype)))
        cases.append((f"{np.dtype(dtype).name}-six-bit", rng.integers(0, 64, INTERVAL, dtype=dtype)))
    rng = np.random.default_rng(481516)
    wide = rng.poisson(2.0, INTERVAL).astype(np.uint16)
    wide[[3, 255, 511]] = [256, 32768, 65535]
    cases.append(("u16-wide-escape", wide))
    cases.append(("u16-incompressible", rng.integers(0, 65536, INTERVAL, dtype=np.uint16)))

    rows = []
    for name, values in cases:
        encoded = encode_stream(values, frequencies, encoding)
        decoded = decode_stream(encoded, values.size, values.dtype, decoding)
        if not np.array_equal(decoded, values):
            raise AssertionError(f"Exact round trip failed for {name}")
        rows.append(
            {
                "case": name,
                "mode": encoded.mode,
                "bytes": len(encoded.payload),
                "paired_transitions": encoded.transitions,
                "scalar_transition_reference": values.size if encoded.transitions else 0,
            }
        )

    entropy = [row for row in rows if row["paired_transitions"]]
    if not entropy or any(row["paired_transitions"] * 2 != row["scalar_transition_reference"] for row in entropy):
        raise AssertionError("Ordinary paired streams must halve dependent transitions")

    # Structural corruption checks do not rewrite valid reference bytes.
    ordinary = next((name, values) for name, values in cases if name == "uint16-poisson")[1]
    encoded = encode_stream(ordinary, frequencies, encoding)
    if not 64 <= encoded.mode < 96:
        raise AssertionError("The corruption fixture must use paired tANS")
    corrupt = bytearray(encoded.payload)
    corrupt[0] |= 1 << 3
    try:
        decode_stream(EncodedStream(encoded.mode, bytes(corrupt), encoded.transitions), INTERVAL, ordinary.dtype, decoding)
    except InvalidStream:
        pass
    else:
        raise AssertionError("Reserved header bits were accepted")
    try:
        decode_stream(EncodedStream(encoded.mode, encoded.payload[:-1], encoded.transitions), INTERVAL, ordinary.dtype, decoding)
    except InvalidStream:
        pass
    else:
        raise AssertionError("Truncated paired payload was accepted")

    return {
        "abi": "metal-runtime-paired-tans-v1",
        "frequency_sha256": hashlib.sha256(frequencies.astype("<u4", copy=False).tobytes()).hexdigest(),
        "encoding_sha256": hashlib.sha256(encoding.astype("<u2", copy=False).tobytes()).hexdigest(),
        "decoding_sha256": hashlib.sha256(decoding.astype("<u4", copy=False).tobytes()).hexdigest(),
        "transition_cases": transition_cases,
        "round_trip_cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_oracle(), indent=2, sort_keys=True))
