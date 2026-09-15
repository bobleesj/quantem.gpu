"""CPU proof for compact 64-pair checkpoint capture and restart direction."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROOF_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "20260913-apple-m5-ans-checkpoint-restart-proof"
)
sys.path.insert(0, str(PROOF_DIRECTORY))

from paired_runtime_tans_checkpoint import (  # noqa: E402
    Checkpoint,
    CodecError,
    CompactCheckpoint,
    ReverseReader,
    STATE_COUNT,
    Tables,
    decode_entropy_segment,
    encode_stream,
)


def _ordinary_values() -> list[int]:
    return [(index * 17 + index // 7) % 11 for index in range(512)]


def _escape_values() -> list[int]:
    values = [(index * 37 + 11) % 64 for index in range(512)]
    values[3] = 40
    values[127] = 63
    values[255] = 256
    values[383] = 32_768
    values[511] = 65_535
    return values


def _odd_values() -> list[int]:
    values = [(index * 29 + index // 5) % 17 for index in range(509)]
    values[8] = 40
    values[12] = 63
    values[18] = 256
    values[25] = 65_535
    return values


class FourWayCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = Tables.build()

    def test_four_64_pair_segments_match_contiguous_decode_checkpoints(self) -> None:
        fixtures = (
            ("ordinary", _ordinary_values()),
            ("escapes and uint16 maximum", _escape_values()),
            ("odd 509-value stream", _odd_values()),
        )
        for name, values in fixtures:
            with self.subTest(fixture=name):
                encoded = encode_stream(values, self.tables)
                self.assertTrue(64 <= encoded.mode < 96)
                pair_count = (len(values) + 1) // 2
                self.assertGreaterEqual(pair_count, 192)
                checkpoint: Checkpoint | None = None
                decoded_values: list[int] = []
                checksums: list[int] = []
                for segment_index in range(4):
                    first_pair = segment_index * 64
                    segment_pair_count = min(64, pair_count - first_pair)
                    pairs, checkpoint = decode_entropy_segment(
                        encoded.mode,
                        encoded.payload,
                        segment_pair_count,
                        self.tables,
                        checkpoint,
                    )
                    segment_values = [value for pair in pairs for value in pair]
                    valid_count = min(2 * segment_pair_count, len(values) - 2 * first_pair)
                    decoded_values.extend(segment_values[:valid_count])
                    checksum = 1469598103934665603
                    for value in segment_values[:valid_count]:
                        checksum = (
                            (checksum ^ (value & 255)) * 1099511628211
                        ) & 0xFFFF_FFFF_FFFF_FFFF
                        checksum = (
                            (checksum ^ ((value >> 8) & 255)) * 1099511628211
                        ) & 0xFFFF_FFFF_FFFF_FFFF
                    checksums.append(checksum)
                    if segment_index < 3:
                        uninterrupted_pairs, uninterrupted = decode_entropy_segment(
                            encoded.mode,
                            encoded.payload,
                            64 * (segment_index + 1),
                            self.tables,
                        )
                        self.assertEqual(len(uninterrupted_pairs), 64 * (segment_index + 1))
                        self.assertEqual(checkpoint.state, uninterrupted.state)
                        self.assertEqual(
                            checkpoint.reader.remaining,
                            uninterrupted.reader.remaining,
                        )
                        record = CompactCheckpoint(
                            encoded.mode,
                            checkpoint.pair_index,
                            checkpoint.state,
                            checkpoint.reader.remaining,
                        )
                        packed = record.pack()
                        self.assertEqual(len(packed), 3)
                        unpacked = CompactCheckpoint.unpack(
                            encoded.mode, checkpoint.pair_index, packed
                        )
                        self.assertEqual(unpacked, record)
                        checkpoint = Checkpoint(
                            unpacked.mode,
                            unpacked.mode - 64,
                            unpacked.state,
                            unpacked.pair_index,
                            ReverseReader.from_bit_position(
                                encoded.payload, unpacked.remaining_bits
                            ).snapshot(),
                        )
                        self.assertLess(checkpoint.state, STATE_COUNT)
                self.assertEqual(decoded_values, values)
                self.assertEqual(len(checksums), 4)
                self.assertEqual(checkpoint.pair_index, pair_count)
                self.assertEqual(checkpoint.state, 0)
                self.assertEqual(checkpoint.reader.remaining, 0)

    def test_fallback_and_interleaved_modes_do_not_admit_checkpoints(self) -> None:
        for mode in (252, 253, 254, 255, 96, 127):
            with self.subTest(mode=mode), self.assertRaises(CodecError):
                CompactCheckpoint(mode, 64, 0, 0).pack()


if __name__ == "__main__":
    unittest.main()
