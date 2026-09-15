"""CPU oracle for exact pair-boundary restart in paired runtime tANS.

This module is a self-contained research proof. It mirrors the production
table builder, byte writer, and reverse stack reader without importing Metal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

STATE_COUNT = 1_024
MODEL_COUNT = 32
SYMBOL_COUNT = 1_089
ESCAPE_SYMBOL = 1_088
MAX_PAIRS = 256
MAX_ENTROPY_BITS = 1 << 14
COMPACT_CHECKPOINT_BYTES = 3


class CodecError(ValueError):
    """A stream violates the paired runtime tANS byte contract."""


@dataclass(frozen=True)
class Tables:
    """Deterministic paired runtime tANS tables."""

    frequencies: tuple[int, ...]
    starts: tuple[int, ...]
    encoding: tuple[int, ...]
    decoding: tuple[int, ...]

    @classmethod
    def build(cls) -> Tables:
        """Build the same 32 normalized models and inverse transitions as Swift.

        Returns
        -------
        Tables
            Frequency starts, encoder states, and packed decoder entries.

        Examples
        --------
        >>> tables = Tables.build()
        >>> len(tables.decoding) == MODEL_COUNT * STATE_COUNT
        True
        """
        frequencies = _build_frequencies()
        starts: list[int] = []
        encoding = [0] * (MODEL_COUNT * STATE_COUNT)
        decoding = [0] * (MODEL_COUNT * STATE_COUNT)
        for model in range(MODEL_COUNT):
            model_frequencies = frequencies[
                model * SYMBOL_COUNT : (model + 1) * SYMBOL_COUNT
            ]
            model_starts: list[int] = []
            total = 0
            for frequency in model_frequencies:
                model_starts.append(total)
                total += frequency
            if total != STATE_COUNT:
                raise CodecError(f"model {model} has {total} states, expected 1024")
            starts.extend(model_starts)

            symbol_for_sequence = [0] * STATE_COUNT
            for symbol, frequency in enumerate(model_frequencies):
                start = model_starts[symbol]
                for sequence in range(start, start + frequency):
                    symbol_for_sequence[sequence] = symbol

            ranks = [0] * SYMBOL_COUNT
            table_base = model * STATE_COUNT
            for state in range(STATE_COUNT):
                sequence = (state * 43) & (STATE_COUNT - 1)
                symbol = symbol_for_sequence[sequence]
                rank = ranks[symbol]
                ranks[symbol] += 1
                frequency = model_frequencies[symbol]
                n = frequency + rank
                bits = 10 - (n.bit_length() - 1)
                base = (n << bits) - STATE_COUNT
                pair = (
                    4_095
                    if symbol == ESCAPE_SYMBOL
                    else symbol // 33 | ((symbol % 33) << 6)
                )
                decoding[table_base + state] = pair | (bits << 12) | (base << 16)
                encoding[model * STATE_COUNT + model_starts[symbol] + rank] = state
            if any(
                ranks[symbol] != frequency
                for symbol, frequency in enumerate(model_frequencies)
            ):
                raise CodecError(f"model {model} has an unassigned encoder rank")
        return cls(tuple(frequencies), tuple(starts), tuple(encoding), tuple(decoding))

    def hashes(self) -> dict[str, str]:
        """Hash each table in its production little-endian storage form.

        Returns
        -------
        dict[str, str]
            SHA-256 values for frequencies, encoding states, and packed decoding.

        Examples
        --------
        >>> hashes = Tables.build().hashes()
        >>> len(hashes["packed_decoding"])
        64
        """
        return {
            "frequencies": _little_endian_hash(self.frequencies, "I"),
            "encoding": _little_endian_hash(self.encoding, "H"),
            "packed_decoding": _little_endian_hash(
                self.decoding,
                "I",
            ),
        }


def _descending_ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _build_frequencies() -> list[int]:
    choices = (4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768)
    state_count = STATE_COUNT
    lower_mean = 0.002
    upper_mean = 32.0
    log_lower = math.log(lower_mean)
    log_upper = math.log(upper_mean)
    log_factorial = [0.0] * 33
    for value in range(1, 33):
        log_factorial[value] = log_factorial[value - 1] + math.log(value)

    result = [0] * (MODEL_COUNT * SYMBOL_COUNT)
    for model in range(MODEL_COUNT):
        interpolation = model / (MODEL_COUNT - 1)
        mean = math.exp(log_lower + interpolation * (log_upper - log_lower))
        log_mean = math.log(mean)
        probability = [
            math.exp(value * log_mean - log_factorial[value] - mean)
            for value in range(33)
        ]
        joint = [0.0] * SYMBOL_COUNT
        for first in range(32):
            for second in range(32):
                joint[first * 33 + second] = probability[first] * probability[second]
        joint_ranks = _descending_ranks(joint)

        best_cost = math.inf
        best_frequency = [0] * SYMBOL_COUNT
        for choice in choices:
            supported = [rank < choice for rank in joint_ranks]
            supported[ESCAPE_SYMBOL] = True
            selected = [0.0] * SYMBOL_COUNT
            selected_total = 0.0
            supported_count = 0
            for symbol, is_supported in enumerate(supported):
                if is_supported:
                    selected[symbol] = joint[symbol]
                    selected_total += selected[symbol]
                    supported_count += 1
            selected[ESCAPE_SYMBOL] = max(0.0, 1.0 - selected_total)

            available = float(state_count - supported_count)
            allocation = [0.0] * SYMBOL_COUNT
            frequency = [0] * SYMBOL_COUNT
            allocated = 0
            for symbol, is_supported in enumerate(supported):
                if is_supported:
                    allocation[symbol] = selected[symbol] * available
                    frequency[symbol] = math.floor(allocation[symbol]) + 1
                    allocated += frequency[symbol]
            remainder = state_count - allocated
            fractions = [-math.inf] * SYMBOL_COUNT
            for symbol, is_supported in enumerate(supported):
                if is_supported:
                    fractions[symbol] = allocation[symbol] - math.floor(allocation[symbol])
            fraction_ranks = _descending_ranks(fractions)
            for symbol, is_supported in enumerate(supported):
                if is_supported and fraction_ranks[symbol] < remainder:
                    frequency[symbol] += 1

            cost = 13.0 * selected[ESCAPE_SYMBOL]
            for symbol in range(SYMBOL_COUNT):
                cost += selected[symbol] * math.log2(
                    state_count / max(frequency[symbol], 1)
                )
            if cost < best_cost:
                best_cost = cost
                best_frequency = frequency
        if sum(best_frequency) != state_count:
            raise CodecError(f"model {model} did not normalize to 1024 states")
        result[model * SYMBOL_COUNT : (model + 1) * SYMBOL_COUNT] = best_frequency
    return result


def _little_endian_hash(values: tuple[int, ...], code: str) -> str:
    packer = struct.Struct("<" + code)
    digest = hashlib.sha256()
    for value in values:
        digest.update(packer.pack(value))
    return digest.hexdigest()


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def model_for(values: list[int]) -> int:
    """Select the production model from the clipped count mean.

    Parameters
    ----------
    values : list[int]
        One nonempty uint16 stream.

    Returns
    -------
    int
        Model index in the range 0 through 31.

    Examples
    --------
    >>> model_for([0, 1, 2, 3]) in range(32)
    True
    """
    if not values:
        raise CodecError("cannot choose a model for an empty stream")
    clipped_sum = sum(min(value, 32) for value in values)
    mean = _float32(_float32(clipped_sum) / _float32(len(values)))
    scaled_log = _float32(math.log(max(mean, _float32(0.002))))
    lower_log = _float32(math.log(_float32(0.002)))
    factor = _float32(31.0 / math.log(16_000.0))
    scaled = _float32(_float32(scaled_log - lower_log) * factor)
    return min(31, max(0, round(scaled)))


@dataclass(frozen=True)
class EncodedStream:
    """One count stream encoded in a paired runtime tANS mode."""

    mode: int
    payload: bytes
    count: int
    model: int | None
    meaningful_bits: int


def _pair(values: list[int], pair_index: int) -> tuple[int, int]:
    first_index = 2 * pair_index
    first = values[first_index]
    second = values[first_index + 1] if first_index + 1 < len(values) else 0
    return first, second


def _symbol(first: int, second: int, model: int, tables: Tables) -> int:
    symbol = first * 33 + second if first < 32 and second < 32 else ESCAPE_SYMBOL
    if tables.frequencies[model * SYMBOL_COUNT + symbol] == 0:
        return ESCAPE_SYMBOL
    return symbol


def _append_field(bits: list[int], value: int, count: int) -> None:
    bits.extend((value >> bit) & 1 for bit in range(count))


@dataclass(frozen=True)
class EntropyBits:
    """Bit-exact output of the reverse pair encoder."""

    start_state: int
    bits: tuple[int, ...]
    pair_bit_counts: tuple[int, ...]


def _encode_pairs(
    pairs: list[tuple[int, int]], model: int, terminal_state: int, tables: Tables
) -> EntropyBits:
    state = terminal_state
    output_bits: list[int] = []
    pair_bit_counts = [0] * len(pairs)
    frequency_base = model * SYMBOL_COUNT
    encoding_base = model * STATE_COUNT
    for pair_index in range(len(pairs) - 1, -1, -1):
        bit_count_before = len(output_bits)
        first, second = pairs[pair_index]
        symbol = _symbol(first, second, model, tables)
        if symbol == ESCAPE_SYMBOL:
            if first < 64 and second < 64:
                _append_field(output_bits, first | (second << 6), 13)
            else:
                _append_field(output_bits, second, 16)
                _append_field(output_bits, first, 16)
                _append_field(output_bits, 4_096, 13)
        start = tables.starts[frequency_base + symbol]
        frequency = tables.frequencies[frequency_base + symbol]
        y = STATE_COUNT + state
        bits = 10 - (frequency.bit_length() - 1)
        if y < (frequency << bits):
            bits -= 1
        rank = (y >> bits) - frequency
        low = 0 if bits == 0 else y & ((1 << bits) - 1)
        _append_field(output_bits, low, bits)
        state = tables.encoding[encoding_base + start + rank]
        pair_bit_counts[pair_index] = len(output_bits) - bit_count_before
    return EntropyBits(state, tuple(output_bits), tuple(pair_bit_counts))


def _pack_bits(bits: tuple[int, ...]) -> bytes:
    payload = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        payload[index // 8] |= bit << (index & 7)
    return bytes(payload)


def encode_entropy_segment(
    pairs: list[tuple[int, int]], model: int, terminal_state: int, tables: Tables
) -> tuple[int, bytes, EntropyBits]:
    """Encode a pair-aligned segment with an explicit terminal checkpoint state.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Complete logical value pairs, including a zero odd-tail partner.
    model : int
        Production tANS model index.
    terminal_state : int
        State reached after decoding this segment.
    tables : Tables
        The deterministic production tables.

    Returns
    -------
    tuple[int, bytes, EntropyBits]
        Mode byte, byte-aligned segment payload, and exact emitted bit trace.

    Examples
    --------
    >>> tables = Tables.build()
    >>> mode, payload, _ = encode_entropy_segment([(1, 2)], 0, 0, tables)
    >>> 64 <= mode < 96 and len(payload) >= 2
    True
    """
    if not 0 <= model < MODEL_COUNT or not 0 <= terminal_state < STATE_COUNT:
        raise CodecError("model and terminal state must be inside their tANS tables")
    encoded = _encode_pairs(pairs, model, terminal_state, tables)
    header = (encoded.start_state << 6) | (len(encoded.bits) & 7)
    payload = struct.pack("<H", header) + _pack_bits(encoded.bits)
    return 64 + model, payload, encoded


def encode_stream(values: list[int], tables: Tables) -> EncodedStream:
    """Encode one uint16 stream with the paired runtime mode-selection rules.

    Parameters
    ----------
    values : list[int]
        One nonempty stream of at most 512 uint16 detector counts.
    tables : Tables
        Deterministic paired runtime tANS tables.

    Returns
    -------
    EncodedStream
        The production mode, payload, model, and meaningful entropy bit count.

    Examples
    --------
    >>> tables = Tables.build()
    >>> encoded = encode_stream([0] * 8, tables)
    >>> encoded.mode
    253
    """
    if not values or len(values) > 512 or any(not 0 <= value <= 65_535 for value in values):
        raise CodecError("stream must contain 1..512 uint16 values")
    minimum, maximum = min(values), max(values)
    nonzero = sum(value != 0 for value in values)
    if minimum == maximum:
        if maximum == 0:
            return EncodedStream(253, b"", len(values), None, 0)
        return EncodedStream(255, struct.pack("<H", maximum), len(values), None, 0)
    if maximum <= 128 and nonzero <= 2:
        return _encode_sparse(values)

    model = model_for(values)
    pairs = [_pair(values, pair_index) for pair_index in range((len(values) + 1) // 2)]
    entropy = _encode_pairs(pairs, model, 0, tables)
    body = _pack_bits(entropy.bits)
    use_entropy = len(body) + 2 < 2 * len(values) and len(entropy.bits) < 16_384
    if maximum <= 128 and 2 * nonzero <= (len(body) + 2 if use_entropy else 2 * len(values)) + 1:
        return _encode_sparse(values)
    if not use_entropy:
        literal = b"".join(struct.pack("<H", value) for value in values)
        return EncodedStream(254, literal, len(values), model, 0)
    header = (entropy.start_state << 6) | (len(entropy.bits) & 7)
    return EncodedStream(
        64 + model,
        struct.pack("<H", header) + body,
        len(values),
        model,
        len(entropy.bits),
    )


def _encode_sparse(values: list[int]) -> EncodedStream:
    payload = bytearray()
    for scan, value in enumerate(values):
        if value:
            event = (scan << 7) | (value - 1)
            payload.extend(struct.pack("<H", event))
    return EncodedStream(252, bytes(payload), len(values), None, 0)


@dataclass(frozen=True)
class ReaderSnapshot:
    """Serializable reverse-reader state at a pair boundary."""

    cursor: int
    last: int
    reservoir: int
    available: int
    remaining: int
    last_bits: int
    valid: bool


@dataclass(frozen=True)
class Checkpoint:
    """State required to restart an entropy decoder exactly between pairs."""

    mode: int
    model: int
    state: int
    pair_index: int
    reader: ReaderSnapshot


@dataclass(frozen=True)
class CompactCheckpoint:
    """Exact three-byte midpoint state for one 512-value entropy stream.

    The original payload remains immutable. Its unread suffix is addressed by
    the remaining meaningful-bit count, so the active reservoir can be
    reconstructed from the original payload instead of being copied into the
    checkpoint.
    """

    mode: int
    pair_index: int
    state: int
    remaining_bits: int

    def pack(self) -> bytes:
        """Pack the state and unread-bit position into exactly 24 bits.

        Returns
        -------
        bytes
            Three little-endian bytes. Bits 0--9 store the tANS state and bits
            10--23 store the unread meaningful-bit count.

        Examples
        --------
        >>> CompactCheckpoint(64, 128, 7, 321).pack() == (7 | (321 << 10)).to_bytes(3, "little")
        True
        """
        if not 64 <= self.mode < 96:
            raise CodecError("compact checkpoint mode must be entropy mode 64..95")
        if not 0 <= self.pair_index <= MAX_PAIRS:
            raise CodecError("compact checkpoint pair index must be in 0..256")
        if not 0 <= self.state < STATE_COUNT:
            raise CodecError("compact checkpoint state must be in 0..1023")
        if not 0 <= self.remaining_bits < MAX_ENTROPY_BITS:
            raise CodecError("compact checkpoint bit position must fit the 14-bit entropy limit")
        packed = self.state | (self.remaining_bits << 10)
        return packed.to_bytes(COMPACT_CHECKPOINT_BYTES, "little")

    @classmethod
    def unpack(cls, mode: int, pair_index: int, packed: bytes) -> CompactCheckpoint:
        """Unpack one three-byte checkpoint.

        Parameters
        ----------
        mode : int
            Immutable source mode, 64 through 95.
        pair_index : int
            Pair boundary represented by this checkpoint.
        packed : bytes
            The exact three-byte checkpoint record.

        Returns
        -------
        CompactCheckpoint
            State and unread-bit position.

        Examples
        --------
        >>> state = CompactCheckpoint(64, 128, 7, 321)
        >>> CompactCheckpoint.unpack(64, 128, state.pack()) == state
        True
        """
        if len(packed) != COMPACT_CHECKPOINT_BYTES:
            raise CodecError("compact checkpoint must be exactly three bytes")
        word = int.from_bytes(packed, "little")
        return cls(mode, pair_index, word & 1023, word >> 10)


class ReverseReader:
    """CPU equivalent of the production `PRTReverseReader`."""

    def __init__(self, payload: bytes, meaningful: int, last_bits: int) -> None:
        self.payload = payload
        self.body_first = 2
        self.cursor = len(payload)
        self.last = len(payload) - 1
        self.reservoir = 0
        self.available = 0
        self.remaining = meaningful
        self.last_bits = last_bits
        self.valid = True

    @classmethod
    def from_snapshot(cls, payload: bytes, snapshot: ReaderSnapshot) -> ReverseReader:
        reader = cls.__new__(cls)
        reader.payload = payload
        reader.body_first = 2
        reader.cursor = snapshot.cursor
        reader.last = snapshot.last
        reader.reservoir = snapshot.reservoir
        reader.available = snapshot.available
        reader.remaining = snapshot.remaining
        reader.last_bits = snapshot.last_bits
        reader.valid = snapshot.valid
        return reader

    @classmethod
    def from_bit_position(cls, payload: bytes, remaining_bits: int) -> ReverseReader:
        """Rebuild an empty-buffer reader at an exact reverse-bit position.

        `remaining_bits` counts the still-unread meaningful bits in the
        original byte stream. The first future refill reads the byte that
        contains the current partial suffix, if any, followed by prior bytes.

        Parameters
        ----------
        payload : bytes
            Original entropy payload, unchanged.
        remaining_bits : int
            Meaningful bits left at the restart boundary.

        Returns
        -------
        ReverseReader
            A reader that will consume exactly the original unread suffix.

        Examples
        --------
        >>> reader = ReverseReader.from_bit_position(b"\\x01\\x02\\x03", 9)
        >>> reader.remaining
        9
        """
        if remaining_bits < 0 or remaining_bits >= MAX_ENTROPY_BITS:
            raise CodecError("restart bit position must fit the 14-bit entropy limit")
        reader = cls.__new__(cls)
        reader.payload = payload
        reader.body_first = 2
        reader.cursor = reader.body_first + (remaining_bits + 7) // 8
        if reader.cursor > len(payload):
            raise CodecError("restart bit position exceeds the original entropy payload")
        reader.last = reader.cursor - 1
        reader.reservoir = 0
        reader.available = 0
        reader.remaining = remaining_bits
        reader.last_bits = remaining_bits & 7 or 8
        reader.valid = True
        return reader

    def snapshot(self) -> ReaderSnapshot:
        return ReaderSnapshot(
            self.cursor,
            self.last,
            self.reservoir,
            self.available,
            self.remaining,
            self.last_bits,
            self.valid,
        )

    def pop(self, count: int) -> int:
        if not self.valid or count > self.remaining:
            self.valid = False
            return 0
        while self.available < count:
            if self.cursor <= self.body_first:
                self.valid = False
                return 0
            self.cursor -= 1
            bit_count = self.last_bits if self.cursor == self.last else 8
            self.reservoir = (self.payload[self.cursor] & ((1 << bit_count) - 1)) | (
                self.reservoir << bit_count
            )
            self.available += bit_count
        self.remaining -= count
        self.available -= count
        value = (self.reservoir >> self.available) & ((1 << count) - 1)
        self.reservoir &= (1 << self.available) - 1
        return value


def _entropy_begin(mode: int, payload: bytes) -> tuple[int, ReverseReader]:
    if not 64 <= mode < 96 or len(payload) < 3:
        raise CodecError("entropy stream must use mode 64..95 and contain a bit body")
    header = struct.unpack_from("<H", payload)[0]
    tail = header & 7
    state = header >> 6
    body_bytes = len(payload) - 2
    if (header >> 3) & 7 or state >= STATE_COUNT or (tail and body_bytes == 0):
        raise CodecError("entropy header has reserved bits or an invalid state")
    if tail and payload[-1] >> tail:
        raise CodecError("entropy body has nonzero padding above its meaningful tail")
    last_bits = tail or 8
    meaningful = (body_bytes - 1) * 8 + last_bits
    return state, ReverseReader(payload, meaningful, last_bits)


def _decode_pair(
    model: int, state: int, reader: ReverseReader, tables: Tables
) -> tuple[tuple[int, int], int]:
    code = tables.decoding[model * STATE_COUNT + state]
    pair = code & 4_095
    bits = (code >> 12) & 15
    state = (code >> 16) + reader.pop(bits)
    if pair == 4_095:
        word = reader.pop(13)
        if word < 4_096:
            first, second = word & 63, word >> 6
        elif word == 4_096:
            first = reader.pop(16)
            second = reader.pop(16)
        else:
            reader.valid = False
            first = second = 0
    else:
        first, second = pair & 63, pair >> 6
    if not reader.valid or state >= STATE_COUNT:
        raise CodecError("entropy pair ran out of bits or left the state table")
    return (first, second), state


def decode_entropy_segment(
    mode: int,
    payload: bytes,
    pair_count: int,
    tables: Tables,
    checkpoint: Checkpoint | None = None,
) -> tuple[list[tuple[int, int]], Checkpoint]:
    """Decode a fixed number of pairs and return the exact restart checkpoint.

    Parameters
    ----------
    mode : int
        Production entropy mode, 64 through 95.
    payload : bytes
        The byte-aligned entropy stream or segment payload.
    pair_count : int
        Number of complete value pairs to decode in this call.
    tables : Tables
        Deterministic paired runtime tANS tables.
    checkpoint : Checkpoint or None, optional
        State returned by the preceding call over the same payload.

    Returns
    -------
    tuple[list[tuple[int, int]], Checkpoint]
        Decoded pairs and state for an exact continuation.

    Examples
    --------
    >>> tables = Tables.build()
    >>> encoded = encode_stream([1, 2] * 128, tables)
    >>> first, checkpoint = decode_entropy_segment(encoded.mode, encoded.payload, 3, tables)
    >>> len(first), checkpoint.pair_index
    (3, 3)
    """
    if pair_count < 0:
        raise CodecError("pair_count must be nonnegative")
    if checkpoint is None:
        state, reader = _entropy_begin(mode, payload)
        model = mode - 64
        pair_index = 0
    else:
        if mode != checkpoint.mode:
            raise CodecError("checkpoint mode does not match the resumed stream")
        model = checkpoint.model
        state = checkpoint.state
        pair_index = checkpoint.pair_index
        reader = ReverseReader.from_snapshot(payload, checkpoint.reader)
    decoded: list[tuple[int, int]] = []
    for _ in range(pair_count):
        pair, state = _decode_pair(model, state, reader, tables)
        decoded.append(pair)
        pair_index += 1
    next_checkpoint = Checkpoint(mode, model, state, pair_index, reader.snapshot())
    return decoded, next_checkpoint


def decode_stream(encoded: EncodedStream, tables: Tables) -> list[int]:
    """Decode a complete entropy or fallback stream using production mode rules.

    Parameters
    ----------
    encoded : EncodedStream
        Mode, payload, and logical count produced by :func:`encode_stream`.
    tables : Tables
        Deterministic paired runtime tANS tables.

    Returns
    -------
    list[int]
        Exact logical uint16 values.

    Examples
    --------
    >>> tables = Tables.build()
    >>> decode_stream(encode_stream([0] * 4, tables), tables)
    [0, 0, 0, 0]
    """
    mode, payload, count = encoded.mode, encoded.payload, encoded.count
    if mode == 253:
        if payload:
            raise CodecError("zero mode must have an empty payload")
        return [0] * count
    if mode == 255:
        if len(payload) != 2:
            raise CodecError("constant mode must contain one uint16")
        return [struct.unpack("<H", payload)[0]] * count
    if mode == 254:
        if len(payload) != 2 * count:
            raise CodecError("literal mode payload length does not match its count")
        return list(struct.unpack("<" + "H" * count, payload))
    if mode == 252:
        if len(payload) % 2:
            raise CodecError("sparse mode payload must contain complete uint16 events")
        values = [0] * count
        previous = -1
        for (event,) in struct.iter_unpack("<H", payload):
            scan = event >> 7
            value = (event & 127) + 1
            if scan >= count or scan <= previous:
                raise CodecError("sparse events must be in-range and strictly ordered")
            values[scan] = value
            previous = scan
        return values
    if not 64 <= mode < 96:
        raise CodecError(f"unsupported paired runtime mode {mode}")
    pairs, checkpoint = decode_entropy_segment(
        mode, payload, (count + 1) // 2, tables
    )
    if checkpoint.reader.remaining != 0 or checkpoint.state != 0:
        raise CodecError("entropy stream did not finish at zero state and zero remaining bits")
    values = [value for pair in pairs for value in pair]
    if count % 2 and values[count] != 0:
        raise CodecError("odd entropy stream has a nonzero unused pair value")
    return values[:count]
