"""CPU-only checks for exact paired runtime tANS restart boundaries."""

from __future__ import annotations

import hashlib
import struct
import unittest

from paired_runtime_tans_checkpoint import (
    Checkpoint,
    CodecError,
    CompactCheckpoint,
    EncodedStream,
    ReaderSnapshot,
    ReverseReader,
    Tables,
    decode_entropy_segment,
    decode_stream,
    encode_entropy_segment,
    encode_stream,
)


def _entropy_values() -> list[int]:
    return [(index * 17 + index // 7) % 11 for index in range(512)]


def _six_bit_escape_values() -> list[int]:
    return [(index * 37 + 11) % 64 for index in range(512)]


def _wide_escape_values() -> list[int]:
    values = _entropy_values()
    values[3] = 40
    values[127] = 63
    values[255] = 256
    values[383] = 32_768
    values[511] = 65_535
    return values


def _odd_wide_escape_values() -> list[int]:
    values = [(index * 29 + index // 5) % 17 for index in range(509)]
    values[8] = 40
    values[12] = 63
    values[18] = 256
    values[25] = 65_535
    return values


def _literal_values() -> list[int]:
    state = 481_516
    values = []
    for _ in range(512):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        values.append((state >> 8) & 0xFFFF)
    return values


def _meaningful_bits(payload: bytes, count: int) -> tuple[int, ...]:
    header = struct.unpack_from("<H", payload)[0]
    tail = header & 7
    bit_count = (len(payload) - 3) * 8 + (tail or 8)
    if bit_count != count:
        raise AssertionError(f"header indicates {bit_count} bits, expected {count}")
    return tuple((payload[2 + index // 8] >> (index & 7)) & 1 for index in range(bit_count))


class PairedRuntimeTANSCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = Tables.build()

    def test_tables_match_the_frozen_production_abi_hashes(self) -> None:
        self.assertEqual(
            self.tables.hashes(),
            {
                "frequencies": "74a39b914a3d563cec122294bf3d29c4972e089ae64ae42c6500eaaafde4145f",
                "encoding": "b76355bf9e4d4c9bf4949faf412671d7ec881b232df7a303250a5071342253d8",
                "packed_decoding": (
                    "44966937a0352082f194c121b37cbd0dd6883cf65919f2d73f94a893c303becd"
                ),
            },
        )

    def test_cpu_encoder_matches_frozen_synthetic_codec_payloads(self) -> None:
        fixtures = [
            (
                "zero",
                [0] * 512,
                253,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                "constant",
                [255] * 512,
                255,
                "ea5dbf9596d187e9500f23e9a680109475341cf4e81f7e043f7d97152c10772f",
            ),
            (
                "entropy",
                _entropy_values(),
                89,
                "d62c1f09f44fe91d3b9b9f95ad9f87b6c5b8ff11f2f6ff979daca5751c667d5d",
            ),
            (
                "six-bit escape",
                _six_bit_escape_values(),
                94,
                "47f2b897bcb7394a0f54b631df6a34696865c472542ce2988f157c25683a7b84",
            ),
            (
                "wide escape",
                _wide_escape_values(),
                89,
                "5c58b3bf2637e37478c3813dce7b285ae3ef9ba66b429459fa0c1834e18a6c77",
            ),
            (
                "sparse",
                self._sparse_values(),
                252,
                "8eae0aa67f373fa805517b67e1891898e403dfd33578e8d4b5ed229803097838",
            ),
            (
                "literal",
                _literal_values(),
                254,
                "db594deee53f8966db4c1b089ef4edfae85db47b2c7ecf618b0ce07df666fe11",
            ),
        ]
        for name, values, expected_mode, expected_hash in fixtures:
            with self.subTest(name=name):
                encoded = encode_stream(values, self.tables)
                self.assertEqual(encoded.mode, expected_mode)
                self.assertEqual(hashlib.sha256(encoded.payload).hexdigest(), expected_hash)
                self.assertEqual(decode_stream(encoded, self.tables), values)

    def test_every_interior_pair_boundary_restarts_and_reframes_exactly(self) -> None:
        tail_lengths: set[int] = set()
        checked_boundaries = 0
        fixtures = [
            ("ordinary", _entropy_values()),
            ("six-bit escape", _six_bit_escape_values()),
            ("wide escape", _wide_escape_values()),
            ("odd tail and escapes", _odd_wide_escape_values()),
        ]
        for name, values in fixtures:
            with self.subTest(name=name):
                encoded = encode_stream(values, self.tables)
                self.assertTrue(64 <= encoded.mode < 96)
                self.assertIsNotNone(encoded.model)
                pair_count = (len(values) + 1) // 2
                source_pairs = [
                    (
                        values[2 * index],
                        values[2 * index + 1]
                        if 2 * index + 1 < len(values)
                        else 0,
                    )
                    for index in range(pair_count)
                ]
                full_bits = _meaningful_bits(encoded.payload, encoded.meaningful_bits)
                initial_state = struct.unpack_from("<H", encoded.payload)[0] >> 6

                for split in range(1, pair_count):
                    checked_boundaries += 1
                    first_pairs, checkpoint = decode_entropy_segment(
                        encoded.mode, encoded.payload, split, self.tables
                    )
                    resumed_checkpoint = Checkpoint(
                        checkpoint.mode,
                        checkpoint.model,
                        checkpoint.state,
                        checkpoint.pair_index,
                        ReaderSnapshot(
                            checkpoint.reader.cursor,
                            checkpoint.reader.last,
                            checkpoint.reader.reservoir,
                            checkpoint.reader.available,
                            checkpoint.reader.remaining,
                            checkpoint.reader.last_bits,
                            checkpoint.reader.valid,
                        ),
                    )
                    second_pairs, terminal = decode_entropy_segment(
                        encoded.mode,
                        encoded.payload,
                        pair_count - split,
                        self.tables,
                        resumed_checkpoint,
                    )
                    self.assertEqual(first_pairs + second_pairs, source_pairs)
                    self.assertEqual(terminal.state, 0)
                    self.assertEqual(terminal.reader.remaining, 0)
                    self.assertEqual(terminal.pair_index, pair_count)

                    compact = CompactCheckpoint(
                        encoded.mode,
                        split,
                        checkpoint.state,
                        checkpoint.reader.remaining,
                    )
                    compact_bytes = compact.pack()
                    restored_compact = CompactCheckpoint.unpack(
                        encoded.mode, split, compact_bytes
                    )
                    self.assertEqual(restored_compact, compact)
                    compact_reader = ReverseReader.from_bit_position(
                        encoded.payload, restored_compact.remaining_bits
                    )
                    compact_restart = Checkpoint(
                        encoded.mode,
                        encoded.model,
                        restored_compact.state,
                        restored_compact.pair_index,
                        compact_reader.snapshot(),
                    )
                    compact_pairs, compact_terminal = decode_entropy_segment(
                        encoded.mode,
                        encoded.payload,
                        pair_count - split,
                        self.tables,
                        compact_restart,
                    )
                    self.assertEqual(first_pairs + compact_pairs, source_pairs)
                    self.assertEqual(compact_terminal.state, 0)
                    self.assertEqual(compact_terminal.reader.remaining, 0)
                    self.assertEqual(compact_terminal.pair_index, pair_count)

                    first_mode, first_payload, first_trace = encode_entropy_segment(
                        source_pairs[:split], encoded.model, checkpoint.state, self.tables
                    )
                    second_mode, second_payload, second_trace = encode_entropy_segment(
                        source_pairs[split:], encoded.model, 0, self.tables
                    )
                    self.assertEqual(first_mode, encoded.mode)
                    self.assertEqual(second_mode, encoded.mode)
                    self.assertEqual(first_trace.start_state, initial_state)
                    self.assertEqual(second_trace.start_state, checkpoint.state)
                    self.assertEqual(second_trace.bits + first_trace.bits, full_bits)
                    self.assertEqual(
                        len(first_trace.bits),
                        encoded.meaningful_bits - checkpoint.reader.remaining,
                    )
                    self.assertEqual(len(second_trace.bits), checkpoint.reader.remaining)
                    self.assertEqual(
                        struct.unpack_from("<H", first_payload)[0] & 7,
                        len(first_trace.bits) & 7,
                    )
                    self.assertEqual(
                        struct.unpack_from("<H", second_payload)[0] & 7,
                        len(second_trace.bits) & 7,
                    )
                    tail_lengths.add(len(first_trace.bits) & 7)
                    tail_lengths.add(len(second_trace.bits) & 7)

                    decoded_first, first_terminal = decode_entropy_segment(
                        first_mode, first_payload, split, self.tables
                    )
                    decoded_second, second_terminal = decode_entropy_segment(
                        second_mode, second_payload, pair_count - split, self.tables
                    )
                    self.assertEqual(decoded_first + decoded_second, source_pairs)
                    self.assertEqual(first_terminal.state, checkpoint.state)
                    self.assertEqual(first_terminal.reader.remaining, 0)
                    self.assertEqual(second_terminal.state, 0)
                    self.assertEqual(second_terminal.reader.remaining, 0)

                decoded = decode_stream(encoded, self.tables)
                self.assertEqual(decoded, values)
                if len(values) % 2:
                    self.assertEqual(source_pairs[-1][1], 0)

        self.assertEqual(tail_lengths, set(range(8)))
        self.assertEqual(checked_boundaries, 1_019)

    def test_fixture_pairs_exercise_both_escape_forms_and_ordinary_symbols(self) -> None:
        ordinary_values = _entropy_values()
        ordinary = [
            (ordinary_values[index], ordinary_values[index + 1])
            for index in range(0, len(ordinary_values), 2)
        ]
        narrow = [
            (_six_bit_escape_values()[index], _six_bit_escape_values()[index + 1])
            for index in range(0, 512, 2)
        ]
        wide = [
            (_wide_escape_values()[index], _wide_escape_values()[index + 1])
            for index in range(0, 512, 2)
        ]
        self.assertTrue(any(first < 32 and second < 32 for first, second in ordinary))
        self.assertTrue(
            any(
                (first >= 32 or second >= 32) and first < 64 and second < 64
                for first, second in narrow
            )
        )
        self.assertTrue(any(first >= 64 or second >= 64 for first, second in wide))

    def test_fallback_and_malformed_modes_follow_the_production_contract(self) -> None:
        entropy = encode_stream(_entropy_values(), self.tables)
        header = bytearray(entropy.payload)
        header[0] |= 1 << 3
        malformed_header = EncodedStream(
            entropy.mode,
            bytes(header),
            entropy.count,
            entropy.model,
            entropy.meaningful_bits,
        )
        with self.assertRaisesRegex(CodecError, "reserved bits"):
            decode_stream(malformed_header, self.tables)

        tail = struct.unpack_from("<H", entropy.payload)[0] & 7
        self.assertNotEqual(tail, 0)
        bad_padding = bytearray(entropy.payload)
        bad_padding[-1] |= 1 << tail
        malformed_padding = EncodedStream(
            entropy.mode,
            bytes(bad_padding),
            entropy.count,
            entropy.model,
            entropy.meaningful_bits,
        )
        with self.assertRaisesRegex(CodecError, "padding"):
            decode_stream(malformed_padding, self.tables)

        truncated = EncodedStream(
            entropy.mode,
            entropy.payload[:-1],
            entropy.count,
            entropy.model,
            entropy.meaningful_bits,
        )
        with self.assertRaises(CodecError):
            decode_stream(truncated, self.tables)
        unsupported = EncodedStream(
            96, entropy.payload, entropy.count, entropy.model, entropy.meaningful_bits
        )
        with self.assertRaisesRegex(CodecError, "unsupported"):
            decode_stream(unsupported, self.tables)

        literal = encode_stream(_literal_values(), self.tables)
        malformed_literal = EncodedStream(254, literal.payload[:-1], literal.count, None, 0)
        with self.assertRaisesRegex(CodecError, "length"):
            decode_stream(malformed_literal, self.tables)

        with self.assertRaisesRegex(CodecError, "empty payload"):
            decode_stream(EncodedStream(253, b"\0", 8, None, 0), self.tables)
        with self.assertRaisesRegex(CodecError, "one uint16"):
            decode_stream(EncodedStream(255, b"", 8, None, 0), self.tables)

        sparse = encode_stream(self._sparse_values(), self.tables)
        unordered = struct.pack("<HH", 511 << 7, (13 << 7) | 127)
        with self.assertRaisesRegex(CodecError, "strictly ordered"):
            decode_stream(EncodedStream(252, unordered, sparse.count, None, 0), self.tables)
        out_of_range = struct.pack("<H", (5 << 7) | 1)
        with self.assertRaisesRegex(CodecError, "in-range"):
            decode_stream(EncodedStream(252, out_of_range, 4, None, 0), self.tables)

    @staticmethod
    def _sparse_values() -> list[int]:
        values = [0] * 512
        values[13] = 128
        values[511] = 1
        return values


if __name__ == "__main__":
    unittest.main()
