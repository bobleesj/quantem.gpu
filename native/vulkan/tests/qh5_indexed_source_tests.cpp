#include "quantem/gpu/vulkan/qh5_indexed_source.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using quantem::gpu::vulkan::Qh5IndexedSegment;
using quantem::gpu::vulkan::Qh5IndexedSource;
using quantem::gpu::vulkan::Shape4D;

namespace {

void require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void append_le32(std::vector<std::uint8_t> &output, const std::uint32_t value) {
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8U));
  output.push_back(static_cast<std::uint8_t>(value >> 16U));
  output.push_back(static_cast<std::uint8_t>(value >> 24U));
}

std::vector<std::uint8_t>
bitshuffle_uint16(const std::vector<std::uint16_t> &values) {
  require(values.size() == 4096U, "test block size");
  std::vector<std::uint8_t> shuffled(8192U, 0);
  for (std::uint32_t bit = 0; bit < 16U; ++bit) {
    for (std::uint32_t group = 0; group < 128U; ++group) {
      std::uint32_t word = 0;
      for (std::uint32_t lane = 0; lane < 32U; ++lane) {
        const std::uint16_t value = values[group * 32U + lane];
        if ((value & static_cast<std::uint16_t>(1U << bit)) != 0)
          word |= 1U << lane;
      }
      const std::size_t offset =
          static_cast<std::size_t>(bit * 128U + group) * 4U;
      shuffled[offset] = static_cast<std::uint8_t>(word);
      shuffled[offset + 1U] = static_cast<std::uint8_t>(word >> 8U);
      shuffled[offset + 2U] = static_cast<std::uint8_t>(word >> 16U);
      shuffled[offset + 3U] = static_cast<std::uint8_t>(word >> 24U);
    }
  }
  return shuffled;
}

std::vector<std::uint8_t>
lz4_literal_block(const std::vector<std::uint8_t> &values) {
  std::vector<std::uint8_t> output;
  output.reserve(values.size() + 40U);
  output.push_back(0xf0U);
  std::size_t remaining = values.size() - 15U;
  while (remaining >= 255U) {
    output.push_back(255U);
    remaining -= 255U;
  }
  output.push_back(static_cast<std::uint8_t>(remaining));
  output.insert(output.end(), values.begin(), values.end());
  return output;
}

void write_all(const int descriptor, const std::vector<std::uint8_t> &bytes) {
  std::size_t written = 0;
  while (written < bytes.size()) {
    const ssize_t count =
        write(descriptor, bytes.data() + written, bytes.size() - written);
    if (count <= 0)
      throw std::runtime_error("test write failed");
    written += static_cast<std::size_t>(count);
  }
}

std::vector<std::uint8_t>
make_index(const std::uint64_t source_bytes, const std::uint32_t first_payload,
           const std::uint32_t first_compressed_bytes,
           const std::uint32_t second_payload,
           const std::uint32_t second_compressed_bytes) {
  const std::string json =
      "{\"sourcePath\":\"fixture.h5\",\"sourceBytes\":" +
      std::to_string(source_bytes) +
      ",\"sourceMtimeNs\":0,\"detRows\":64,\"detCols\":64,"
      "\"nFrames\":2,\"srcDtype\":\"uint16\",\"blockElems\":4096,"
      "\"nBlocksPerFrame\":1,\"chunks\":[{\"startFrame\":0,"
      "\"nFrames\":2,\"rangeStart\":0,\"rangeEnd\":" +
      std::to_string(source_bytes) +
      ",\"metaOffsetWords\":0,\"metaWords\":4}]}";
  std::vector<std::uint8_t> output{'Q', 'H', '5', 'I', 'D', 'X', '0', '1'};
  append_le32(output, static_cast<std::uint32_t>(json.size()));
  append_le32(output, 4U);
  output.insert(output.end(), json.begin(), json.end());
  while ((output.size() & 3U) != 0)
    output.push_back(0);
  append_le32(output, first_payload);
  append_le32(output, first_compressed_bytes);
  append_le32(output, second_payload);
  append_le32(output, second_compressed_bytes);
  return output;
}

} // namespace

int main() {
  std::vector<std::uint16_t> expected(8192U);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] = static_cast<std::uint16_t>(
        index < 4096U ? (index * 17U) & 0xffffU
                      : ((index - 4096U) * 257U + 123U) & 0xffffU);
  }
  const std::vector<std::uint8_t> first = lz4_literal_block(bitshuffle_uint16(
      std::vector<std::uint16_t>(expected.begin(), expected.begin() + 4096)));
  const std::vector<std::uint8_t> second = lz4_literal_block(bitshuffle_uint16(
      std::vector<std::uint16_t>(expected.begin() + 4096, expected.end())));
  constexpr std::uint32_t first_payload = 64U;
  const std::uint32_t second_payload =
      first_payload + static_cast<std::uint32_t>(first.size()) + 7U;
  const std::uint64_t source_bytes = second_payload + second.size();
  std::vector<std::uint8_t> source(static_cast<std::size_t>(source_bytes),
                                   0xa5U);
  std::copy(first.begin(), first.end(), source.begin() + first_payload);
  std::copy(second.begin(), second.end(), source.begin() + second_payload);
  const std::vector<std::uint8_t> index = make_index(
      source_bytes, first_payload, static_cast<std::uint32_t>(first.size()),
      second_payload, static_cast<std::uint32_t>(second.size()));

  char source_name[] = "/tmp/qgpu-qh5-source-XXXXXX";
  char index_name[] = "/tmp/qgpu-qh5-index-XXXXXX";
  const int source_descriptor = mkstemp(source_name);
  const int index_descriptor = mkstemp(index_name);
  require(source_descriptor >= 0 && index_descriptor >= 0,
          "temporary descriptors");
  unlink(source_name);
  unlink(index_name);
  write_all(source_descriptor, source);
  write_all(index_descriptor, index);

  const Qh5IndexedSegment segment{
      source_descriptor, 0, source.size(), index_descriptor, 0, index.size(),
  };
  auto reader = Qh5IndexedSource::open({segment}, Shape4D{1, 2, 64, 64});
  close(source_descriptor);
  close(index_descriptor);
  require(reader->frame_count() == 2U, "indexed frame count");
  require(reader->blocks_per_frame() == 1U, "indexed blocks per frame");
  std::vector<std::uint16_t> decoded(expected.size());
  const auto metrics = reader->read_frames(0, 2, decoded.data());
  require(decoded == expected, "lossless uint16 decode");
  require(metrics.source_frame_count == 2U, "decoded source frame count");
  require(metrics.source_block_count == 2U, "decoded source block count");
  require(metrics.source_bytes_read >= first.size() + second.size(),
          "compressed source byte count");
  require(metrics.storage_read_milliseconds >= 0.0, "storage timing");
  require(metrics.source_decode_milliseconds >= 0.0, "decode timing");

  const auto compressed = reader->read_compressed_frames(0, 2);
  require(compressed.metrics.source_frame_count == 2U,
          "compressed source frame count");
  require(compressed.metrics.source_block_count == 2U,
          "compressed source block count");
  require(compressed.metrics.source_bytes_read >= first.size() + second.size(),
          "compressed storage byte count");
  require(compressed.metrics.source_decode_milliseconds == 0.0,
          "compressed read does not claim CPU decode");
  require(compressed.compressed_byte_count == first.size() + second.size(),
          "compressed payload byte count");
  require(compressed.block_metadata ==
              std::vector<std::uint32_t>{
                  0U, static_cast<std::uint32_t>(first.size()),
                  static_cast<std::uint32_t>(first.size()),
                  static_cast<std::uint32_t>(second.size())},
          "compressed block metadata remains logical and tightly packed");
  require((compressed.compressed_bytes.size() & 3U) == 0U &&
              std::equal(first.begin(), first.end(),
                         compressed.compressed_bytes.begin()) &&
              std::equal(second.begin(), second.end(),
                         compressed.compressed_bytes.begin() + first.size()),
          "compressed GPU staging bytes retain exact LZ4 streams");

  std::cout << "PASS: quantem.gpu indexed QH5 uint16 tests\n";
  return 0;
}
