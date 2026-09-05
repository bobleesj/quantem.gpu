#include "quantem/gpu/vulkan/c_api.h"
#include "quantem/gpu/vulkan/exact_products.hpp"

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

namespace {

constexpr std::uint32_t kScanRows = 8;
constexpr std::uint32_t kScanColumns = 8;
constexpr std::uint32_t kDetectorRows = 192;
constexpr std::uint32_t kDetectorColumns = 192;
constexpr std::size_t kDetectorValues =
    static_cast<std::size_t>(kDetectorRows) * kDetectorColumns;
constexpr std::size_t kBlockValues = 4096;
constexpr std::uint32_t kBlocksPerFrame =
    static_cast<std::uint32_t>(kDetectorValues / kBlockValues);

int executor_creation_count = 0;

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

std::uint16_t expected_value(const std::uint32_t frame,
                             const std::uint32_t detector_index) {
  return static_cast<std::uint16_t>(
      (frame * 1009U + detector_index * 31U + (detector_index / 127U) * 17U) &
      0xffffU);
}

std::vector<std::uint8_t>
bitshuffle_uint16(const std::vector<std::uint16_t> &values) {
  require(values.size() == kBlockValues, "test block size");
  std::vector<std::uint8_t> shuffled(kBlockValues * sizeof(std::uint16_t), 0);
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

struct Fixture {
  int source_descriptor = -1;
  int index_descriptor = -1;
  std::uint64_t source_bytes = 0;
  std::uint64_t one_frame_source_bytes = 0;
  std::uint64_t index_bytes = 0;
};

Fixture make_fixture() {
  constexpr std::uint32_t frame_count = kScanRows * kScanColumns;
  constexpr std::uint32_t block_count = frame_count * kBlocksPerFrame;
  std::vector<std::uint8_t> source;
  std::vector<std::uint32_t> metadata;
  metadata.reserve(static_cast<std::size_t>(block_count) * 2U);
  std::vector<std::uint16_t> values(kBlockValues);
  std::uint64_t one_frame_source_bytes = 0;
  for (std::uint32_t frame = 0; frame < frame_count; ++frame) {
    for (std::uint32_t block = 0; block < kBlocksPerFrame; ++block) {
      for (std::uint32_t index = 0; index < kBlockValues; ++index) {
        values[index] = expected_value(
            frame, block * static_cast<std::uint32_t>(kBlockValues) + index);
      }
      const std::vector<std::uint8_t> compressed =
          lz4_literal_block(bitshuffle_uint16(values));
      require(source.size() <= UINT32_MAX, "test source offset range");
      metadata.push_back(static_cast<std::uint32_t>(source.size()));
      metadata.push_back(static_cast<std::uint32_t>(compressed.size()));
      source.insert(source.end(), compressed.begin(), compressed.end());
      if (frame == 0)
        one_frame_source_bytes += compressed.size();
    }
  }
  const std::string json =
      "{\"sourcePath\":\"selected-diffraction-fixture.h5\","
      "\"sourceBytes\":" +
      std::to_string(source.size()) +
      ",\"sourceMtimeNs\":0,\"detRows\":192,\"detCols\":192,"
      "\"nFrames\":64,\"srcDtype\":\"uint16\",\"blockElems\":4096,"
      "\"nBlocksPerFrame\":9,\"chunks\":[{\"startFrame\":0,"
      "\"nFrames\":64,\"rangeStart\":0,\"rangeEnd\":" +
      std::to_string(source.size()) + ",\"metaOffsetWords\":0,\"metaWords\":" +
      std::to_string(metadata.size()) + "}]}";
  std::vector<std::uint8_t> index{'Q', 'H', '5', 'I', 'D', 'X', '0', '1'};
  append_le32(index, static_cast<std::uint32_t>(json.size()));
  append_le32(index, static_cast<std::uint32_t>(metadata.size()));
  index.insert(index.end(), json.begin(), json.end());
  while ((index.size() & 3U) != 0)
    index.push_back(0);
  for (const std::uint32_t word : metadata)
    append_le32(index, word);

  char source_name[] = "/tmp/qgpu-selected-source-XXXXXX";
  char index_name[] = "/tmp/qgpu-selected-index-XXXXXX";
  const int source_descriptor = mkstemp(source_name);
  const int index_descriptor = mkstemp(index_name);
  require(source_descriptor >= 0 && index_descriptor >= 0,
          "temporary descriptors");
  unlink(source_name);
  unlink(index_name);
  write_all(source_descriptor, source);
  write_all(index_descriptor, index);
  return {source_descriptor, index_descriptor, source.size(),
          one_frame_source_bytes, index.size()};
}

void require_exact_frame(const std::vector<std::uint16_t> &actual,
                         const std::uint32_t row, const std::uint32_t column) {
  const std::uint32_t frame = row * kScanColumns + column;
  for (std::uint32_t index = 0; index < kDetectorValues; ++index) {
    if (actual[index] != expected_value(frame, index)) {
      std::cerr << "FAIL: frame mismatch at (" << row << ", " << column
                << ") detector index " << index << '\n';
      std::exit(1);
    }
  }
}

} // namespace

namespace quantem::gpu::vulkan {

std::unique_ptr<ExactProductExecutor> ExactProductExecutor::create() {
  ++executor_creation_count;
  throw std::runtime_error(
      "selected-diffraction host test must not initialize Vulkan");
}

} // namespace quantem::gpu::vulkan

int main() {
  Fixture fixture = make_fixture();
  qgpu_source_segment source{};
  source.struct_size = sizeof(source);
  source.borrowed_file_descriptor = fixture.source_descriptor;
  source.length_bytes = fixture.source_bytes;
  for (std::size_t index = 0; index < QGPU_SHA256_BYTES; ++index)
    source.sha256[index] = static_cast<std::uint8_t>(index + 1U);
  qgpu_qh5_index_segment qh5_index{};
  qh5_index.struct_size = sizeof(qh5_index);
  qh5_index.borrowed_file_descriptor = fixture.index_descriptor;
  qh5_index.length_bytes = fixture.index_bytes;
  for (std::size_t index = 0; index < QGPU_SHA256_BYTES; ++index)
    qh5_index.sha256[index] = static_cast<std::uint8_t>(0x80U + index);

  qgpu_vulkan_open_request open{};
  open.struct_size = sizeof(open);
  open.abi_version = QGPU_VULKAN_ABI_VERSION;
  open.generation = 10;
  open.container = QGPU_SOURCE_QH5_INDEXED_BITSHUFFLE_LZ4;
  open.dataset.struct_size = sizeof(open.dataset);
  open.dataset.scan_rows = kScanRows;
  open.dataset.scan_columns = kScanColumns;
  open.dataset.detector_rows = kDetectorRows;
  open.dataset.detector_columns = kDetectorColumns;
  open.dataset.source_dtype = QGPU_SOURCE_UINT16;
  open.dataset.scan_bin = 1;
  open.dataset.detector_bin = 1;
  open.dataset.crop_is_none = true;
  for (std::size_t index = 0; index < QGPU_SHA256_BYTES; ++index) {
    open.dataset.source_identity_sha256[index] =
        static_cast<std::uint8_t>(0x40U + index);
  }
  open.ordered_source_segments = &source;
  open.source_segment_count = 1;
  open.borrowed_cache_directory_file_descriptor = -1;
  open.ordered_qh5_index_segments = &qh5_index;
  open.qh5_index_segment_count = 1;

  qgpu_error error{};
  error.struct_size = sizeof(error);
  qgpu_vulkan_session *session = nullptr;
  require(qgpu_vulkan_open_v1(&open, &session, &error) == QGPU_STATUS_OK,
          "open indexed QH5 session");
  require(session != nullptr, "open session result");
  require(executor_creation_count == 0, "open does not initialize Vulkan");
  close(fixture.source_descriptor);
  close(fixture.index_descriptor);

  qgpu_vulkan_event event{};
  event.struct_size = sizeof(event);
  require(qgpu_vulkan_poll_event_v1(session, &event, &error) ==
                  QGPU_STATUS_OK &&
              event.type == QGPU_EVENT_OPENED,
          "open event");
  event.struct_size = sizeof(event);
  require(qgpu_vulkan_poll_event_v1(session, &event, &error) ==
              QGPU_STATUS_NO_EVENT,
          "no queued product event before reads");

  std::vector<std::uint16_t> destination(kDetectorValues, 0);
  qgpu_vulkan_selected_diffraction_request request{};
  request.struct_size = sizeof(request);
  request.generation = 11;
  request.destination_uint16 = destination.data();
  request.destination_value_capacity = destination.size();
  const std::array<std::array<std::uint32_t, 2>, 6> coordinates{{
      {0, 0},
      {0, 7},
      {7, 0},
      {7, 7},
      {4, 4},
      {2, 5},
  }};
  for (const auto &coordinate : coordinates) {
    request.scan_row = coordinate[0];
    request.scan_column = coordinate[1];
    qgpu_vulkan_selected_diffraction_result result{};
    result.struct_size = sizeof(result);
    require(qgpu_vulkan_read_selected_diffraction_v1(session, &request, &result,
                                                     &error) == QGPU_STATUS_OK,
            "selected diffraction read");
    require_exact_frame(destination, request.scan_row, request.scan_column);
    require(result.generation == request.generation,
            "selected generation identity");
    require(result.scan_row == request.scan_row &&
                result.scan_column == request.scan_column,
            "selected coordinate identity");
    require(result.detector_rows == kDetectorRows &&
                result.detector_columns == kDetectorColumns &&
                result.source_dtype == QGPU_SOURCE_UINT16,
            "selected detector identity");
    require(result.destination_value_count == kDetectorValues,
            "selected destination count");
    require(std::memcmp(result.source_identity_sha256,
                        open.dataset.source_identity_sha256,
                        QGPU_SHA256_BYTES) == 0,
            "selected source identity");
    require(result.metrics.source_frame_count == 1,
            "exactly one source frame read");
    require(result.metrics.source_block_count == kBlocksPerFrame,
            "exactly one frame's blocks read");
    require(result.metrics.source_bytes_read == fixture.one_frame_source_bytes,
            "exact selected frame compressed bytes");
    require(result.metrics.storage_read_milliseconds >= 0.0 &&
                result.metrics.source_decode_milliseconds >= 0.0 &&
                result.metrics.total_milliseconds >= 0.0,
            "selected read timings");
    require(result.metrics.full_product_source_frames_read == 0 &&
                result.metrics.vulkan_queue_submit_count == 0,
            "no full product traversal or Vulkan submission");
    require(executor_creation_count == 0,
            "selected reads do not initialize Vulkan");
  }

  request.scan_row = kScanRows;
  request.scan_column = 0;
  std::fill(destination.begin(), destination.end(), 0x5a5aU);
  qgpu_vulkan_selected_diffraction_result invalid_result{};
  invalid_result.struct_size = sizeof(invalid_result);
  require(qgpu_vulkan_read_selected_diffraction_v1(session, &request,
                                                   &invalid_result, &error) ==
              QGPU_STATUS_INVALID_ARGUMENT,
          "reject invalid scan row");
  require(
      std::all_of(destination.begin(), destination.end(),
                  [](const std::uint16_t value) { return value == 0x5a5aU; }),
      "invalid coordinate leaves destination untouched");

  request.scan_row = 0;
  request.scan_column = kScanColumns;
  invalid_result.struct_size = sizeof(invalid_result);
  require(qgpu_vulkan_read_selected_diffraction_v1(session, &request,
                                                   &invalid_result, &error) ==
              QGPU_STATUS_INVALID_ARGUMENT,
          "reject invalid scan column");
  require(
      std::all_of(destination.begin(), destination.end(),
                  [](const std::uint16_t value) { return value == 0x5a5aU; }),
      "invalid column leaves destination untouched");

  request.scan_row = 1;
  request.scan_column = 1;
  request.destination_value_capacity = kDetectorValues - 1U;
  invalid_result.struct_size = sizeof(invalid_result);
  require(qgpu_vulkan_read_selected_diffraction_v1(session, &request,
                                                   &invalid_result, &error) ==
              QGPU_STATUS_INVALID_ARGUMENT,
          "reject undersized destination");
  require(
      std::all_of(destination.begin(), destination.end(),
                  [](const std::uint16_t value) { return value == 0x5a5aU; }),
      "invalid capacity leaves destination untouched");

  request.destination_value_capacity = destination.size();
  request.generation = 10;
  invalid_result.struct_size = sizeof(invalid_result);
  require(qgpu_vulkan_read_selected_diffraction_v1(session, &request,
                                                   &invalid_result, &error) ==
              QGPU_STATUS_STALE_GENERATION,
          "reject stale selected generation");

  request.generation = 11;
  request.scan_row = 4;
  request.scan_column = 4;
  invalid_result.struct_size = sizeof(invalid_result);
  require(qgpu_vulkan_read_selected_diffraction_v1(
              session, &request, &invalid_result, &error) == QGPU_STATUS_OK,
          "repeat selected read with reusable caller storage");
  require_exact_frame(destination, 4, 4);
  require(executor_creation_count == 0,
          "repeated selected reads remain Vulkan-free");

  std::vector<double> total_samples;
  std::vector<double> storage_samples;
  std::vector<double> decode_samples;
  total_samples.reserve(120);
  storage_samples.reserve(120);
  decode_samples.reserve(120);
  for (std::uint32_t sample = 0; sample < 120U; ++sample) {
    request.scan_row = (sample * 5U) % kScanRows;
    request.scan_column = (sample * 3U + 1U) % kScanColumns;
    invalid_result.struct_size = sizeof(invalid_result);
    require(qgpu_vulkan_read_selected_diffraction_v1(
                session, &request, &invalid_result, &error) == QGPU_STATUS_OK,
            "repeated selected timing read");
    require_exact_frame(destination, request.scan_row, request.scan_column);
    total_samples.push_back(invalid_result.metrics.total_milliseconds);
    storage_samples.push_back(invalid_result.metrics.storage_read_milliseconds);
    decode_samples.push_back(invalid_result.metrics.source_decode_milliseconds);
  }
  const auto percentile = [](std::vector<double> values,
                             const std::size_t numerator) {
    std::sort(values.begin(), values.end());
    return values[((values.size() - 1U) * numerator) / 100U];
  };
  const double total_p50 = percentile(total_samples, 50U);
  const double total_p95 = percentile(total_samples, 95U);
  const double storage_p50 = percentile(storage_samples, 50U);
  const double decode_p50 = percentile(decode_samples, 50U);
  event.struct_size = sizeof(event);
  require(qgpu_vulkan_poll_event_v1(session, &event, &error) ==
              QGPU_STATUS_NO_EVENT,
          "selected reads do not enqueue full-product events");

  qgpu_vulkan_close_v1(&session);
  require(session == nullptr, "session close");
  std::cout << "PASS: quantem.gpu selected diffraction C ABI tests; "
            << "host warm synthetic total p50=" << total_p50
            << " ms p95=" << total_p95 << " ms, storage p50=" << storage_p50
            << " ms, decode p50=" << decode_p50 << " ms\n";
  return 0;
}
