#include "quantem/gpu/vulkan/qh5_indexed_source.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace quantem::gpu::vulkan {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::array<char, 8> kIndexMagic{'Q', 'H', '5', 'I',
                                          'D', 'X', '0', '1'};
constexpr std::uint32_t kUint16BlockElements = 4096;
constexpr std::uint32_t kUint16BlockBytes = 8192;
constexpr std::uint64_t kMaximumReadSpanBytes = 64ULL * 1024ULL * 1024ULL;

double milliseconds(const Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

std::uint64_t checked_sum(const std::uint64_t left, const std::uint64_t right,
                          const char *label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::invalid_argument(std::string(label) + " exceeds uint64 range");
  }
  return left + right;
}

std::uint64_t checked_product(const std::uint64_t left,
                              const std::uint64_t right, const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " exceeds uint64 range");
  }
  return left * right;
}

struct OwnedDescriptor {
  int value = -1;
  std::uint64_t base_offset = 0;
  std::uint64_t length = 0;

  OwnedDescriptor() = default;
  OwnedDescriptor(const int source, const std::uint64_t offset,
                  const std::uint64_t requested_length)
      : base_offset(offset), length(requested_length) {
    if (source < 0 || requested_length == 0) {
      throw std::invalid_argument(
          "QH5 source and index descriptors are required");
    }
    struct stat status{};
    if (fstat(source, &status) != 0 || status.st_size < 0 ||
        offset > static_cast<std::uint64_t>(status.st_size) ||
        requested_length >
            static_cast<std::uint64_t>(status.st_size) - offset) {
      throw std::invalid_argument("QH5 descriptor range exceeds its file");
    }
    value = dup(source);
    if (value < 0) {
      throw Qh5SourceIoError("failed to duplicate a QH5 file descriptor");
    }
  }
  OwnedDescriptor(const OwnedDescriptor &) = delete;
  OwnedDescriptor &operator=(const OwnedDescriptor &) = delete;
  OwnedDescriptor(OwnedDescriptor &&other) noexcept
      : value(std::exchange(other.value, -1)), base_offset(other.base_offset),
        length(other.length) {}
  OwnedDescriptor &operator=(OwnedDescriptor &&other) noexcept {
    if (this != &other) {
      if (value >= 0)
        close(value);
      value = std::exchange(other.value, -1);
      base_offset = other.base_offset;
      length = other.length;
    }
    return *this;
  }
  ~OwnedDescriptor() {
    if (value >= 0)
      close(value);
  }
};

void read_exact(const OwnedDescriptor &source, const std::uint64_t offset,
                const std::uint64_t bytes, std::uint8_t *destination) {
  if (offset > source.length || bytes > source.length - offset) {
    throw Qh5SourceIoError("QH5 read range exceeds its descriptor");
  }
  std::uint64_t completed = 0;
  while (completed < bytes) {
    const std::uint64_t requested = std::min<std::uint64_t>(
        bytes - completed,
        static_cast<std::uint64_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t count =
        pread(source.value, destination + completed,
              static_cast<std::size_t>(requested),
              static_cast<off_t>(source.base_offset + offset + completed));
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      throw Qh5SourceIoError("QH5 range read ended before completion");
    }
    completed += static_cast<std::uint64_t>(count);
  }
}

std::uint32_t read_le32(const std::uint8_t *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint32_t read_plane_word(const std::uint8_t *bytes,
                              const std::size_t word) {
  return read_le32(bytes + word * sizeof(std::uint32_t));
}

std::size_t key_value_start(const std::string_view json,
                            const std::string_view key) {
  const std::string token = "\"" + std::string(key) + "\"";
  const std::size_t key_position = json.find(token);
  if (key_position == std::string_view::npos)
    throw std::invalid_argument("QH5 index is missing JSON key " +
                                std::string(key));
  const std::size_t colon = json.find(':', key_position + token.size());
  if (colon == std::string_view::npos)
    throw std::invalid_argument("QH5 index has a malformed JSON key " +
                                std::string(key));
  std::size_t position = colon + 1;
  while (position < json.size() &&
         std::isspace(static_cast<unsigned char>(json[position])) != 0) {
    ++position;
  }
  return position;
}

std::uint64_t json_uint(const std::string_view json,
                        const std::string_view key) {
  std::size_t position = key_value_start(json, key);
  if (position >= json.size() ||
      !std::isdigit(static_cast<unsigned char>(json[position]))) {
    throw std::invalid_argument("QH5 index JSON value is not unsigned: " +
                                std::string(key));
  }
  std::uint64_t value = 0;
  while (position < json.size() &&
         std::isdigit(static_cast<unsigned char>(json[position])) != 0) {
    const std::uint32_t digit =
        static_cast<std::uint32_t>(json[position] - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      throw std::invalid_argument("QH5 index JSON integer overflows: " +
                                  std::string(key));
    }
    value = value * 10U + digit;
    ++position;
  }
  return value;
}

std::string json_string(const std::string_view json,
                        const std::string_view key) {
  std::size_t position = key_value_start(json, key);
  if (position >= json.size() || json[position] != '"') {
    throw std::invalid_argument("QH5 index JSON value is not a string: " +
                                std::string(key));
  }
  ++position;
  std::string result;
  while (position < json.size()) {
    const char value = json[position++];
    if (value == '"')
      return result;
    if (value == '\\') {
      if (position >= json.size())
        break;
      const char escaped = json[position++];
      if (escaped == '"' || escaped == '\\' || escaped == '/') {
        result.push_back(escaped);
      } else {
        throw std::invalid_argument(
            "QH5 index uses an unsupported JSON string escape");
      }
    } else {
      result.push_back(value);
    }
  }
  throw std::invalid_argument("QH5 index contains an unterminated JSON string");
}

struct ChunkMetadata {
  std::uint64_t start_frame = 0;
  std::uint64_t frame_count = 0;
  std::uint64_t range_start = 0;
  std::uint64_t range_end = 0;
  std::uint64_t metadata_offset_words = 0;
  std::uint64_t metadata_words = 0;
};

std::vector<ChunkMetadata> json_chunks(const std::string_view json) {
  std::size_t position = key_value_start(json, "chunks");
  if (position >= json.size() || json[position] != '[')
    throw std::invalid_argument("QH5 index chunks value is not an array");
  ++position;
  std::vector<ChunkMetadata> chunks;
  while (position < json.size()) {
    while (position < json.size() &&
           (std::isspace(static_cast<unsigned char>(json[position])) != 0 ||
            json[position] == ',')) {
      ++position;
    }
    if (position < json.size() && json[position] == ']')
      break;
    if (position >= json.size() || json[position] != '{')
      throw std::invalid_argument("QH5 index chunks array is malformed");
    const std::size_t object_start = position;
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (; position < json.size(); ++position) {
      const char value = json[position];
      if (in_string) {
        if (escaped) {
          escaped = false;
        } else if (value == '\\') {
          escaped = true;
        } else if (value == '"') {
          in_string = false;
        }
        continue;
      }
      if (value == '"') {
        in_string = true;
      } else if (value == '{') {
        ++depth;
      } else if (value == '}') {
        if (--depth == 0) {
          ++position;
          break;
        }
      }
    }
    if (depth != 0)
      throw std::invalid_argument("QH5 index chunk object is unterminated");
    const std::string_view object =
        json.substr(object_start, position - object_start);
    chunks.push_back({
        json_uint(object, "startFrame"),
        json_uint(object, "nFrames"),
        json_uint(object, "rangeStart"),
        json_uint(object, "rangeEnd"),
        json_uint(object, "metaOffsetWords"),
        json_uint(object, "metaWords"),
    });
  }
  if (chunks.empty())
    throw std::invalid_argument("QH5 index contains no chunks");
  return chunks;
}

void lz4_decompress_exact(const std::uint8_t *compressed,
                          const std::size_t compressed_bytes,
                          std::uint8_t *output,
                          const std::size_t output_bytes) {
  std::size_t input_position = 0;
  std::size_t output_position = 0;
  const auto extended_length = [&](std::size_t length) {
    if (length != 15U)
      return length;
    std::uint8_t next = 255;
    while (next == 255) {
      if (input_position >= compressed_bytes)
        throw std::invalid_argument("truncated QH5 LZ4 length");
      next = compressed[input_position++];
      length += next;
    }
    return length;
  };

  while (input_position < compressed_bytes) {
    const std::uint8_t token = compressed[input_position++];
    const std::size_t literal_bytes = extended_length(token >> 4U);
    if (literal_bytes > compressed_bytes - input_position ||
        literal_bytes > output_bytes - output_position) {
      throw std::invalid_argument("QH5 LZ4 literal range is invalid");
    }
    std::memcpy(output + output_position, compressed + input_position,
                literal_bytes);
    input_position += literal_bytes;
    output_position += literal_bytes;
    if (input_position == compressed_bytes)
      break;
    if (compressed_bytes - input_position < 2U)
      throw std::invalid_argument("QH5 LZ4 match offset is truncated");
    const std::size_t match_offset =
        static_cast<std::size_t>(compressed[input_position]) |
        (static_cast<std::size_t>(compressed[input_position + 1U]) << 8U);
    input_position += 2U;
    if (match_offset == 0 || match_offset > output_position)
      throw std::invalid_argument("QH5 LZ4 match offset is invalid");
    const std::size_t match_bytes = extended_length(token & 0x0fU) + 4U;
    if (match_bytes > output_bytes - output_position)
      throw std::invalid_argument("QH5 LZ4 match exceeds the decoded block");
    for (std::size_t index = 0; index < match_bytes; ++index) {
      output[output_position + index] =
          output[output_position + index - match_offset];
    }
    output_position += match_bytes;
  }
  if (output_position != output_bytes) {
    throw std::invalid_argument("QH5 LZ4 block does not decode to 8192 bytes");
  }
}

void bitunshuffle_uint16(const std::uint8_t *shuffled, std::uint16_t *output) {
  for (std::uint32_t group = 0; group < 128U; ++group) {
    for (std::uint32_t lane = 0; lane < 32U; ++lane) {
      std::uint16_t value = 0;
      for (std::uint32_t bit = 0; bit < 16U; ++bit) {
        if ((read_plane_word(shuffled, bit * 128U + group) & (1U << lane)) !=
            0) {
          value |= static_cast<std::uint16_t>(1U << bit);
        }
      }
      output[group * 32U + lane] = value;
    }
  }
}

} // namespace

struct Qh5IndexedSource::Impl {
  struct SourceFile {
    OwnedDescriptor source;
  };
  struct Block {
    std::uint32_t source_index = 0;
    std::uint64_t offset = 0;
    std::uint32_t compressed_bytes = 0;
  };

  Shape4D shape;
  std::uint32_t blocks_per_frame = 0;
  std::vector<SourceFile> sources;
  std::vector<Block> blocks;
};

Qh5IndexedSource::Qh5IndexedSource(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

Qh5IndexedSource::Qh5IndexedSource(Qh5IndexedSource &&) noexcept = default;
Qh5IndexedSource &
Qh5IndexedSource::operator=(Qh5IndexedSource &&) noexcept = default;
Qh5IndexedSource::~Qh5IndexedSource() = default;

std::unique_ptr<Qh5IndexedSource>
Qh5IndexedSource::open(const std::vector<Qh5IndexedSegment> &segments,
                       const Shape4D &expected_shape) {
  if (segments.empty())
    throw std::invalid_argument("indexed QH5 loading requires source segments");
  if (expected_shape.detector_pixel_count() % kUint16BlockElements != 0) {
    throw std::invalid_argument(
        "indexed uint16 QH5 loading requires complete 4096-value blocks");
  }
  auto result = std::make_unique<Impl>();
  result->shape = expected_shape;
  result->blocks_per_frame = static_cast<std::uint32_t>(
      expected_shape.detector_pixel_count() / kUint16BlockElements);
  std::uint64_t total_frames = 0;
  result->sources.reserve(segments.size());

  for (std::size_t segment_index = 0; segment_index < segments.size();
       ++segment_index) {
    const auto &input = segments[segment_index];
    OwnedDescriptor source(input.borrowed_source_file_descriptor,
                           input.source_file_offset_bytes,
                           input.source_file_length_bytes);
    OwnedDescriptor index(input.borrowed_index_file_descriptor,
                          input.index_file_offset_bytes,
                          input.index_file_length_bytes);
    if (index.length > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("QH5 index exceeds the process size range");
    std::vector<std::uint8_t> index_bytes(
        static_cast<std::size_t>(index.length));
    read_exact(index, 0, index.length, index_bytes.data());
    if (index_bytes.size() < 16U ||
        !std::equal(kIndexMagic.begin(), kIndexMagic.end(),
                    index_bytes.begin())) {
      throw std::invalid_argument("QH5 index magic is missing or invalid");
    }
    const std::uint64_t json_bytes = read_le32(index_bytes.data() + 8U);
    const std::uint64_t word_count = read_le32(index_bytes.data() + 12U);
    const std::uint64_t json_end = checked_sum(16U, json_bytes, "QH5 JSON end");
    const std::uint64_t binary_start =
        checked_sum(json_end, 3U, "QH5 JSON alignment") & ~3ULL;
    const std::uint64_t binary_bytes =
        checked_product(word_count, sizeof(std::uint32_t), "QH5 word bytes");
    if (json_end > index.length ||
        checked_sum(binary_start, binary_bytes, "QH5 index end") !=
            index.length) {
      throw std::invalid_argument("QH5 index has truncated or trailing bytes");
    }
    const std::string_view json(
        reinterpret_cast<const char *>(index_bytes.data() + 16U),
        static_cast<std::size_t>(json_bytes));
    const std::uint64_t source_bytes = json_uint(json, "sourceBytes");
    const std::uint64_t detector_rows = json_uint(json, "detRows");
    const std::uint64_t detector_columns = json_uint(json, "detCols");
    const std::uint64_t frame_count = json_uint(json, "nFrames");
    const std::string dtype = json_string(json, "srcDtype");
    const std::uint64_t block_elements = json_uint(json, "blockElems");
    const std::uint64_t blocks_per_frame = json_uint(json, "nBlocksPerFrame");
    if (source_bytes != source.length ||
        detector_rows != expected_shape.detector_rows ||
        detector_columns != expected_shape.detector_columns ||
        frame_count == 0 || dtype != "uint16" ||
        block_elements != kUint16BlockElements ||
        blocks_per_frame != result->blocks_per_frame) {
      throw std::invalid_argument(
          "QH5 index disagrees with the declared uint16 source geometry");
    }

    const std::vector<ChunkMetadata> chunks = json_chunks(json);
    std::uint64_t expected_frame = 0;
    std::uint64_t expected_word = 0;
    for (const auto &chunk : chunks) {
      const std::uint64_t expected_chunk_words =
          checked_product(checked_product(chunk.frame_count, blocks_per_frame,
                                          "QH5 chunk blocks"),
                          2U, "QH5 chunk words");
      if (chunk.start_frame != expected_frame || chunk.frame_count == 0 ||
          chunk.range_start >= chunk.range_end ||
          chunk.range_end > source.length ||
          chunk.metadata_offset_words != expected_word ||
          chunk.metadata_words != expected_chunk_words ||
          expected_word > word_count ||
          chunk.metadata_words > word_count - expected_word) {
        throw std::invalid_argument(
            "QH5 chunks do not cover frames and words exactly once");
      }
      std::uint64_t previous_end = chunk.range_start;
      for (std::uint64_t word = chunk.metadata_offset_words;
           word < chunk.metadata_offset_words + chunk.metadata_words;
           word += 2U) {
        const std::uint8_t *pair =
            index_bytes.data() + binary_start + word * sizeof(std::uint32_t);
        const std::uint64_t relative = read_le32(pair);
        const std::uint64_t compressed =
            read_le32(pair + sizeof(std::uint32_t));
        const std::uint64_t payload =
            checked_sum(chunk.range_start, relative, "QH5 payload offset");
        const std::uint64_t payload_end =
            checked_sum(payload, compressed, "QH5 payload end");
        if (compressed == 0 || payload < previous_end ||
            payload_end > chunk.range_end ||
            compressed > std::numeric_limits<std::uint32_t>::max()) {
          throw std::invalid_argument(
              "QH5 compressed block exceeds its source range");
        }
        result->blocks.push_back({
            static_cast<std::uint32_t>(segment_index),
            payload,
            static_cast<std::uint32_t>(compressed),
        });
        previous_end = payload_end;
      }
      if (previous_end != chunk.range_end)
        throw std::invalid_argument(
            "QH5 block metadata does not reach chunk end");
      expected_frame += chunk.frame_count;
      expected_word += chunk.metadata_words;
    }
    if (expected_frame != frame_count || expected_word != word_count) {
      throw std::invalid_argument(
          "QH5 final frame or word coverage is incomplete");
    }
    total_frames = checked_sum(total_frames, frame_count, "QH5 frame coverage");
    result->sources.push_back({std::move(source)});
  }

  if (total_frames != expected_shape.scan_count() ||
      result->blocks.size() != checked_product(total_frames,
                                               result->blocks_per_frame,
                                               "QH5 total blocks")) {
    throw std::invalid_argument(
        "QH5 indexes do not cover the declared logical scan exactly once");
  }
  return std::unique_ptr<Qh5IndexedSource>(
      new Qh5IndexedSource(std::move(result)));
}

std::uint64_t Qh5IndexedSource::frame_count() const {
  return implementation_->shape.scan_count();
}

std::uint32_t Qh5IndexedSource::blocks_per_frame() const {
  return implementation_->blocks_per_frame;
}

Qh5ReadMetrics Qh5IndexedSource::read_frames(const std::uint64_t first_frame,
                                             const std::uint32_t frame_count,
                                             std::uint16_t *destination) const {
  if (destination == nullptr || frame_count == 0 ||
      first_frame > this->frame_count() ||
      frame_count > this->frame_count() - first_frame) {
    throw std::invalid_argument("QH5 frame read is outside the logical scan");
  }
  const std::uint64_t first_block = checked_product(
      first_frame, implementation_->blocks_per_frame, "QH5 first block");
  const std::uint64_t block_count = checked_product(
      frame_count, implementation_->blocks_per_frame, "QH5 read blocks");
  const std::uint64_t stop_block =
      checked_sum(first_block, block_count, "QH5 stop block");
  Qh5ReadMetrics metrics;
  metrics.source_frame_count = frame_count;
  metrics.source_block_count = block_count;
  std::array<std::uint8_t, kUint16BlockBytes> shuffled{};
  std::uint64_t cursor = first_block;
  while (cursor < stop_block) {
    const auto &first =
        implementation_->blocks[static_cast<std::size_t>(cursor)];
    std::uint64_t group_stop = cursor + 1U;
    std::uint64_t span_end =
        checked_sum(first.offset, first.compressed_bytes, "QH5 read span");
    while (group_stop < stop_block) {
      const auto &next =
          implementation_->blocks[static_cast<std::size_t>(group_stop)];
      if (next.source_index != first.source_index || next.offset < first.offset)
        break;
      const std::uint64_t next_end =
          checked_sum(next.offset, next.compressed_bytes, "QH5 read span");
      if (next_end - first.offset > kMaximumReadSpanBytes)
        break;
      span_end = std::max(span_end, next_end);
      ++group_stop;
    }
    const std::uint64_t span_bytes = span_end - first.offset;
    std::vector<std::uint8_t> compressed(static_cast<std::size_t>(span_bytes));
    const auto read_started = Clock::now();
    read_exact(implementation_->sources[first.source_index].source,
               first.offset, span_bytes, compressed.data());
    metrics.storage_read_milliseconds +=
        milliseconds(Clock::now() - read_started);
    metrics.source_bytes_read += span_bytes;

    const auto decode_started = Clock::now();
    for (std::uint64_t block_index = cursor; block_index < group_stop;
         ++block_index) {
      const auto &block =
          implementation_->blocks[static_cast<std::size_t>(block_index)];
      const std::uint64_t relative = block.offset - first.offset;
      if (relative > span_bytes ||
          block.compressed_bytes > span_bytes - relative)
        throw std::logic_error("QH5 block escaped its retained read span");
      lz4_decompress_exact(compressed.data() + relative, block.compressed_bytes,
                           shuffled.data(), shuffled.size());
      const std::uint64_t requested_block = block_index - first_block;
      const std::uint64_t requested_frame =
          requested_block / implementation_->blocks_per_frame;
      const std::uint64_t block_in_frame =
          requested_block % implementation_->blocks_per_frame;
      const std::uint64_t detector_offset = checked_sum(
          checked_product(requested_frame,
                          implementation_->shape.detector_pixel_count(),
                          "QH5 decoded frame offset"),
          checked_product(block_in_frame, kUint16BlockElements,
                          "QH5 decoded block offset"),
          "QH5 decoded output offset");
      bitunshuffle_uint16(shuffled.data(), destination + detector_offset);
    }
    metrics.source_decode_milliseconds +=
        milliseconds(Clock::now() - decode_started);
    cursor = group_stop;
  }
  return metrics;
}

Qh5CompressedBatch Qh5IndexedSource::read_compressed_frames(
    const std::uint64_t first_frame, const std::uint32_t frame_count) const {
  if (frame_count == 0 || first_frame > this->frame_count() ||
      frame_count > this->frame_count() - first_frame) {
    throw std::invalid_argument(
        "QH5 compressed frame read is outside the logical scan");
  }
  const std::uint64_t first_block = checked_product(
      first_frame, implementation_->blocks_per_frame,
      "QH5 compressed first block");
  const std::uint64_t block_count = checked_product(
      frame_count, implementation_->blocks_per_frame,
      "QH5 compressed read blocks");
  const std::uint64_t stop_block = checked_sum(
      first_block, block_count, "QH5 compressed stop block");
  std::uint64_t payload_bytes = 0;
  for (std::uint64_t block_index = first_block; block_index < stop_block;
       ++block_index) {
    payload_bytes = checked_sum(
        payload_bytes,
        implementation_->blocks[static_cast<std::size_t>(block_index)]
            .compressed_bytes,
        "QH5 compressed payload bytes");
  }
  if (payload_bytes > std::numeric_limits<std::uint32_t>::max() ||
      block_count > std::numeric_limits<std::uint32_t>::max() / 2U ||
      payload_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(
        "QH5 compressed batch exceeds the Vulkan uint32 address range");
  }

  Qh5CompressedBatch batch;
  batch.compressed_bytes.reserve(
      static_cast<std::size_t>((payload_bytes + 3U) & ~3ULL));
  batch.block_metadata.reserve(static_cast<std::size_t>(block_count) * 2U);
  batch.metrics.source_frame_count = frame_count;
  batch.metrics.source_block_count = block_count;

  std::uint64_t cursor = first_block;
  while (cursor < stop_block) {
    const auto &first =
        implementation_->blocks[static_cast<std::size_t>(cursor)];
    std::uint64_t group_stop = cursor + 1U;
    std::uint64_t span_end =
        checked_sum(first.offset, first.compressed_bytes,
                    "QH5 compressed read span");
    while (group_stop < stop_block) {
      const auto &next =
          implementation_->blocks[static_cast<std::size_t>(group_stop)];
      if (next.source_index != first.source_index || next.offset < first.offset)
        break;
      const std::uint64_t next_end =
          checked_sum(next.offset, next.compressed_bytes,
                      "QH5 compressed read span");
      if (next_end - first.offset > kMaximumReadSpanBytes)
        break;
      span_end = std::max(span_end, next_end);
      ++group_stop;
    }
    const std::uint64_t span_bytes = span_end - first.offset;
    std::vector<std::uint8_t> source_span(
        static_cast<std::size_t>(span_bytes));
    const auto read_started = Clock::now();
    read_exact(implementation_->sources[first.source_index].source,
               first.offset, span_bytes, source_span.data());
    batch.metrics.storage_read_milliseconds +=
        milliseconds(Clock::now() - read_started);
    batch.metrics.source_bytes_read += span_bytes;

    for (std::uint64_t block_index = cursor; block_index < group_stop;
         ++block_index) {
      const auto &block =
          implementation_->blocks[static_cast<std::size_t>(block_index)];
      const std::uint64_t relative = block.offset - first.offset;
      if (relative > span_bytes ||
          block.compressed_bytes > span_bytes - relative ||
          batch.compressed_bytes.size() >
              std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error(
            "QH5 compressed block escaped its retained read span");
      }
      batch.block_metadata.push_back(
          static_cast<std::uint32_t>(batch.compressed_bytes.size()));
      batch.block_metadata.push_back(block.compressed_bytes);
      batch.compressed_bytes.insert(
          batch.compressed_bytes.end(), source_span.begin() + relative,
          source_span.begin() + relative + block.compressed_bytes);
    }
    cursor = group_stop;
  }

  if (batch.compressed_bytes.size() != payload_bytes ||
      batch.block_metadata.size() != block_count * 2U) {
    throw std::logic_error(
        "QH5 compressed batch does not cover its requested blocks exactly");
  }
  batch.compressed_byte_count = static_cast<std::uint32_t>(payload_bytes);
  batch.compressed_bytes.resize(
      static_cast<std::size_t>((payload_bytes + 3U) & ~3ULL), 0);
  return batch;
}

} // namespace quantem::gpu::vulkan
