#include "quantem/gpu/vulkan/c_api.h"
#include "quantem/gpu/vulkan/contract.hpp"
#include "quantem/gpu/vulkan/exact_products.hpp"
#include "quantem/gpu/vulkan/packed_detector.hpp"
#include "quantem/gpu/vulkan/packed_detector_session.hpp"
#include "quantem/gpu/vulkan/qh5_indexed_source.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

using quantem::gpu::vulkan::BenchmarkMetrics;
using quantem::gpu::vulkan::BenchmarkOptions;
using quantem::gpu::vulkan::ExactProductExecutor;
using quantem::gpu::vulkan::ExactProducts;
using quantem::gpu::vulkan::Shape4D;
using quantem::gpu::vulkan::VulkanCapabilities;

std::string escape_json(const std::string &input) {
  std::ostringstream output;
  for (const unsigned char character : input) {
    switch (character) {
    case '\\':
      output << "\\\\";
      break;
    case '"':
      output << "\\\"";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20) {
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<int>(character) << std::dec;
      } else {
        output << character;
      }
    }
  }
  return output.str();
}

std::uint32_t parse_uint(const std::vector<std::string> &arguments,
                         const std::string &option,
                         const std::uint32_t fallback) {
  for (std::size_t index = 0; index + 1 < arguments.size(); ++index) {
    if (arguments[index] == option) {
      const unsigned long value = std::stoul(arguments[index + 1]);
      if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(option + " exceeds uint32 range");
      }
      return static_cast<std::uint32_t>(value);
    }
  }
  return fallback;
}

bool has_flag(const std::vector<std::string> &arguments,
              const std::string &option) {
  return std::find(arguments.begin(), arguments.end(), option) !=
         arguments.end();
}

std::string parse_string(const std::vector<std::string> &arguments,
                         const std::string &option,
                         const std::string &fallback = {}) {
  for (std::size_t index = 0; index + 1 < arguments.size(); ++index) {
    if (arguments[index] == option)
      return arguments[index + 1];
  }
  return fallback;
}

std::vector<std::uint8_t>
parse_mask_pixels(const std::vector<std::string> &arguments,
                  const Shape4D shape) {
  std::vector<std::uint8_t> result(
      static_cast<std::size_t>(shape.detector_pixel_count()), 0U);
  for (std::size_t index = 0; index + 1 < arguments.size(); ++index) {
    if (arguments[index] != "--mask-pixel")
      continue;
    const std::string &coordinate = arguments[index + 1];
    const std::size_t comma = coordinate.find(',');
    if (comma == std::string::npos)
      throw std::invalid_argument("--mask-pixel requires row,column");
    const unsigned long row = std::stoul(coordinate.substr(0, comma));
    const unsigned long column = std::stoul(coordinate.substr(comma + 1));
    if (row >= shape.detector_rows || column >= shape.detector_columns)
      throw std::invalid_argument("--mask-pixel is outside the detector");
    result[row * shape.detector_columns + column] = 1U;
  }
  return result;
}

std::vector<std::uint8_t> detector_bands(const Shape4D shape) {
  std::vector<std::uint8_t> membership(
      static_cast<std::size_t>(shape.detector_pixel_count()), 0);
  const double center_row = (shape.detector_rows - 1) / 2.0;
  const double center_column = (shape.detector_columns - 1) / 2.0;
  for (std::uint32_t row = 0; row < shape.detector_rows; ++row) {
    for (std::uint32_t column = 0; column < shape.detector_columns; ++column) {
      const double delta_row = row - center_row;
      const double delta_column = column - center_column;
      const double radius =
          std::sqrt(delta_row * delta_row + delta_column * delta_column);
      std::uint8_t member = 0;
      if (radius <= 30.5)
        member |= 1;
      if (radius > 20.5 && radius <= 50.5)
        member |= 2;
      if (radius > 60.5 && radius <= 90.5)
        member |= 4;
      membership[static_cast<std::size_t>(row) * shape.detector_columns +
                 column] = member;
    }
  }
  return membership;
}

std::string capabilities_json(const VulkanCapabilities &capabilities) {
  std::ostringstream output;
  output << std::boolalpha << '{'
         << "\"schema\":\"quantem.gpu.android-vulkan-capabilities/v1\","
         << "\"device_name\":\"" << escape_json(capabilities.device_name)
         << "\","
         << "\"api_version\":\"" << VK_VERSION_MAJOR(capabilities.api_version)
         << '.' << VK_VERSION_MINOR(capabilities.api_version) << '.'
         << VK_VERSION_PATCH(capabilities.api_version) << "\","
         << "\"driver_version_raw\":" << capabilities.driver_version << ','
         << "\"vendor_id\":" << capabilities.vendor_id << ','
         << "\"device_id\":" << capabilities.device_id << ','
         << "\"max_storage_buffer_range_bytes\":"
         << capabilities.max_storage_buffer_range_bytes << ','
         << "\"max_memory_allocation_bytes\":"
         << capabilities.max_memory_allocation_bytes << ','
         << "\"subgroup_size\":" << capabilities.subgroup_size << ','
         << "\"timestamp_valid_bits\":" << capabilities.timestamp_valid_bits
         << ',' << "\"timestamp_period_nanoseconds\":"
         << capabilities.timestamp_period_nanoseconds << ','
         << "\"shader_int16\":" << capabilities.shader_int16 << ','
         << "\"shader_int64\":" << capabilities.shader_int64 << ','
         << "\"storage_buffer_8bit_access\":"
         << capabilities.storage_buffer_8bit_access << ','
         << "\"host_visible_device_local_memory\":"
         << capabilities.host_visible_device_local_memory << '}';
  return output.str();
}

std::string trial_json(const std::uint32_t trial,
                       const BenchmarkOptions &options,
                       const BenchmarkMetrics &metrics,
                       const std::uint64_t global_total,
                       const quantem::gpu::vulkan::DpcProducts &dpc) {
  const double wall_seconds =
      metrics.full_exact_completion_milliseconds / 1000.0;
  const double source_gbps =
      wall_seconds > 0 ? static_cast<double>(metrics.source_bytes_staged) /
                             wall_seconds / 1.0e9
                       : 0.0;
  std::ostringstream output;
  output << std::boolalpha << std::fixed << std::setprecision(4) << '{'
         << "\"schema\":\"quantem.gpu.android-vulkan-benchmark-trial/v1\","
         << "\"status\":\"" << (metrics.mismatch_count == 0 ? "PASS" : "FAIL")
         << "\",\"trial\":" << trial << ','
         << "\"fixture\":\"synthetic-u8-shifted-detector-ramp\","
         << "\"generator\":\"value(scan,pixel)=(scan+pixel)&255\","
         << "\"real_data\":false,"
         << "\"source_shape\":[" << options.source_shape.scan_rows << ','
         << options.source_shape.scan_columns << ','
         << options.source_shape.detector_rows << ','
         << options.source_shape.detector_columns << "],"
         << "\"source_dtype\":\"uint8\",\"scan_bin\":1,"
         << "\"detector_bin\":1,\"crop\":null,"
         << "\"selected_scan_row\":" << options.selected_scan_row << ','
         << "\"selected_scan_column\":" << options.selected_scan_column << ','
         << "\"shard_scan_rows\":" << options.shard_scan_rows << ','
         << "\"staging_ring_depth\":" << options.staging_ring_depth << ','
         << "\"wait_after_each_shard\":" << options.wait_after_each_shard << ','
         << "\"products\":[\"selected_diffraction\",\"BF\",\"ABF\","
            "\"ADF\",\"mean_diffraction\",\"total_intensity\","
            "\"center_of_mass_row\",\"center_of_mass_column\","
            "\"DPC_row\",\"DPC_column\",\"iDPC\"],"
         << "\"accumulation_dtypes\":{\"per_scan_sums\":\"uint32\","
            "\"detector_moments\":\"uint32\","
            "\"diffraction_sum\":\"uint32\","
            "\"global_total\":\"uint64\"},"
         << "\"setup_ms\":" << metrics.setup_milliseconds << ','
         << "\"source_staging_ms\":" << metrics.source_staging_milliseconds
         << ',' << "\"storage_read_ms\":" << metrics.storage_read_milliseconds
         << ',' << "\"source_decode_ms\":" << metrics.source_decode_milliseconds
         << ',' << "\"vulkan_visibility_ms\":"
         << metrics.vulkan_visibility_milliseconds << ','
         << "\"first_correct_product_ms\":"
         << metrics.first_correct_product_milliseconds << ','
         << "\"full_exact_completion_ms\":"
         << metrics.full_exact_completion_milliseconds << ','
         << "\"dpc_and_idpc_ms\":" << metrics.dpc_and_idpc_milliseconds << ','
         << "\"package_ready_ms\":" << metrics.package_ready_milliseconds << ','
         << "\"gpu_scan_products_ms\":"
         << metrics.gpu_scan_products_milliseconds << ','
         << "\"gpu_mean_diffraction_ms\":"
         << metrics.gpu_mean_diffraction_milliseconds << ','
         << "\"source_bytes_staged\":" << metrics.source_bytes_staged << ','
         << "\"source_bytes_read\":" << metrics.source_bytes_read << ','
         << "\"explicit_copy_bytes\":" << metrics.explicit_copy_bytes << ','
         << "\"shader_source_bytes_read\":" << metrics.shader_source_bytes_read
         << ',' << "\"effective_source_gbps\":" << source_gbps << ','
         << "\"vulkan_committed_bytes\":" << metrics.vulkan_committed_bytes
         << ',' << "\"queue_submit_count\":" << metrics.queue_submit_count
         << ',' << "\"fence_wait_count\":" << metrics.fence_wait_count << ','
         << "\"device_wide_wait_count\":" << metrics.device_wide_wait_count
         << ',' << "\"user_cpu_ms\":" << metrics.user_cpu_milliseconds << ','
         << "\"system_cpu_ms\":" << metrics.system_cpu_milliseconds << ','
         << "\"maximum_resident_set_kibibytes\":"
         << metrics.maximum_resident_set_kibibytes << ','
         << "\"minor_page_fault_count\":" << metrics.minor_page_fault_count
         << ','
         << "\"major_page_fault_count\":" << metrics.major_page_fault_count
         << ',' << "\"parity_checked\":" << metrics.parity_checked << ','
         << "\"mismatch_count\":" << metrics.mismatch_count << ','
         << "\"global_total_intensity\":" << global_total << ','
         << "\"dpc_rotation_degrees\":" << dpc.rotation_degrees << ','
         << "\"dpc_component_order_exchanged\":"
         << dpc.component_order_exchanged << '}';
  return output.str();
}

int abi_smoke() {
  constexpr std::uint32_t scan_rows = 8;
  constexpr std::uint32_t scan_columns = 8;
  constexpr std::uint32_t detector_rows = 16;
  constexpr std::uint32_t detector_columns = 16;
  constexpr std::uint32_t scan_count = scan_rows * scan_columns;
  constexpr std::uint32_t detector_pixels = detector_rows * detector_columns;
  std::vector<std::uint8_t> source(scan_count * detector_pixels);
  for (std::uint32_t scan = 0; scan < scan_count; ++scan) {
    for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
      source[static_cast<std::size_t>(scan) * detector_pixels + pixel] =
          static_cast<std::uint8_t>((scan + pixel) & 255U);
    }
  }

  char temporary_name[] = "/data/local/tmp/quantem-gpu-abi-smoke-XXXXXX";
  const int descriptor = mkstemp(temporary_name);
  if (descriptor < 0)
    throw std::runtime_error("mkstemp failed for ABI smoke");
  unlink(temporary_name);
  std::size_t written = 0;
  while (written < source.size()) {
    const ssize_t count =
        write(descriptor, source.data() + written, source.size() - written);
    if (count <= 0) {
      close(descriptor);
      throw std::runtime_error("write failed for ABI smoke");
    }
    written += static_cast<std::size_t>(count);
  }

  qgpu_source_segment segment{};
  segment.struct_size = sizeof(segment);
  segment.borrowed_file_descriptor = descriptor;
  segment.length_bytes = source.size();
  segment.source_ordinal = 0;
  segment.sha256[0] = 1;
  constexpr char selector[] = "/entry/data/data";
  constexpr char identity[] = "quantem.gpu-abi-smoke";
  qgpu_vulkan_open_request open{};
  open.struct_size = sizeof(open);
  open.abi_version = QGPU_VULKAN_ABI_VERSION;
  open.generation = 41;
  open.container = QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT8;
  open.dataset.struct_size = sizeof(open.dataset);
  open.dataset.scan_rows = scan_rows;
  open.dataset.scan_columns = scan_columns;
  open.dataset.detector_rows = detector_rows;
  open.dataset.detector_columns = detector_columns;
  open.dataset.source_dtype = QGPU_SOURCE_UINT8;
  open.dataset.scan_bin = 1;
  open.dataset.detector_bin = 1;
  open.dataset.crop_is_none = true;
  open.dataset.dataset_selector = {selector, sizeof(selector) - 1};
  open.dataset.source_identity_sha256[0] = 1;
  open.ordered_source_segments = &segment;
  open.source_segment_count = 1;
  open.borrowed_cache_directory_file_descriptor = -1;
  open.uri_grant_identity = {identity, sizeof(identity) - 1};

  qgpu_error error{};
  error.struct_size = sizeof(error);
  qgpu_vulkan_session *session = nullptr;
  qgpu_status status = qgpu_vulkan_open_v1(&open, &session, &error);
  close(descriptor);
  if (status != QGPU_STATUS_OK) {
    throw std::runtime_error(std::string("ABI open failed: ") + error.message);
  }

  qgpu_vulkan_product_request request{};
  request.struct_size = sizeof(request);
  request.generation = 41;
  request.selected_scan_row = 3;
  request.selected_scan_column = 4;
  request.bright_field = {7.5F, 7.5F, 0.0F, 4.0F};
  request.annular_bright_field = {7.5F, 7.5F, 2.0F, 6.0F};
  request.annular_dark_field = {7.5F, 7.5F, 6.0F, 10.0F};
  request.shard_scan_rows = 2;
  request.staging_ring_depth = 2;
  qgpu_vulkan_result *result = nullptr;
  status = qgpu_vulkan_request_products_v1(session, &request, &result, &error);
  if (status != QGPU_STATUS_OK) {
    qgpu_vulkan_close_v1(&session);
    throw std::runtime_error(std::string("ABI request failed: ") +
                             error.message);
  }
  qgpu_vulkan_result_view view{};
  view.struct_size = sizeof(view);
  status = qgpu_vulkan_result_view_v1(result, &view, &error);
  if (status != QGPU_STATUS_OK) {
    qgpu_vulkan_result_release_v1(&result);
    qgpu_vulkan_close_v1(&session);
    throw std::runtime_error(std::string("ABI view failed: ") + error.message);
  }

  std::uint64_t mismatches = 0;
  const std::uint32_t selected_scan =
      request.selected_scan_row * scan_columns + request.selected_scan_column;
  std::uint64_t expected_global = 0;
  for (std::uint32_t scan = 0; scan < scan_count; ++scan) {
    std::uint32_t expected_total = 0;
    for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
      expected_total += (scan + pixel) & 255U;
    }
    expected_global += expected_total;
    mismatches += view.total_intensity_uint64[scan] != expected_total;
  }
  for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
    mismatches += view.selected_diffraction_uint8[pixel] !=
                  static_cast<std::uint8_t>((selected_scan + pixel) & 255U);
  }
  mismatches += view.global_total_intensity_uint64 != expected_global;
  mismatches += view.scan_product_count != scan_count;
  mismatches += view.detector_product_count != detector_pixels;
  mismatches += view.generation != 41;
  const qgpu_vulkan_metrics result_metrics = view.metrics;

  bool saw_opened = false;
  bool saw_plan = false;
  bool saw_first = false;
  bool saw_progress = false;
  bool saw_ready = false;
  bool event_order_ok = true;
  std::uint64_t last_event_sequence = 0;
  std::uint32_t event_count = 0;
  while (true) {
    qgpu_vulkan_event event{};
    event.struct_size = sizeof(event);
    status = qgpu_vulkan_poll_event_v1(session, &event, &error);
    if (status == QGPU_STATUS_NO_EVENT)
      break;
    if (status != QGPU_STATUS_OK) {
      qgpu_vulkan_result_release_v1(&result);
      qgpu_vulkan_close_v1(&session);
      throw std::runtime_error(std::string("ABI event poll failed: ") +
                               error.message);
    }
    ++event_count;
    event_order_ok = event_order_ok && event.sequence > last_event_sequence;
    last_event_sequence = event.sequence;
    mismatches += event.generation != 41;
    mismatches += event.source_identity_sha256[0] != 1;
    saw_opened = saw_opened || event.type == QGPU_EVENT_OPENED;
    saw_plan = saw_plan || event.type == QGPU_EVENT_SOURCE_PLAN_READY;
    saw_first = saw_first || event.type == QGPU_EVENT_FIRST_PRODUCT_READY;
    saw_progress = saw_progress || event.type == QGPU_EVENT_PROGRESS;
    saw_ready = saw_ready || event.type == QGPU_EVENT_PRODUCTS_READY;
  }
  const bool event_contract_checked = saw_opened && saw_plan && saw_first &&
                                      saw_progress && saw_ready &&
                                      event_order_ok;
  mismatches += !event_contract_checked;

  qgpu_vulkan_result_release_v1(&result);
  qgpu_vulkan_result_release_v1(&result);
  qgpu_vulkan_close_v1(&session);
  qgpu_vulkan_close_v1(&session);
  std::cout << std::boolalpha << std::fixed << std::setprecision(4) << '{'
            << "\"schema\":\"quantem.gpu.android-vulkan-abi-smoke/v1\","
            << "\"status\":\"" << (mismatches == 0 ? "PASS" : "FAIL") << "\","
            << "\"abi_version\":" << qgpu_vulkan_abi_version() << ','
            << "\"source_shape\":[8,8,16,16],\"source_dtype\":\"uint8\","
            << "\"descriptor_duplicated_before_source_close\":true,"
            << "\"idempotent_result_release\":true,"
            << "\"idempotent_session_close\":true,"
            << "\"event_contract_checked\":" << event_contract_checked << ','
            << "\"event_count\":" << event_count << ','
            << "\"benchmark_parity_checked\":true,"
            << "\"mismatch_count\":" << mismatches << ','
            << "\"storage_read_ms\":"
            << result_metrics.storage_read_milliseconds << ','
            << "\"full_exact_ms\":" << result_metrics.full_exact_milliseconds
            << ',' << "\"package_ready_ms\":"
            << result_metrics.package_ready_milliseconds << "}\n";
  return mismatches == 0 ? 0 : 2;
}

template <typename Value>
void write_values(const std::filesystem::path &path,
                  const std::vector<Value> &values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error("could not create " + path.string());
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(Value)));
  if (!stream)
    throw std::runtime_error("could not write " + path.string());
}

template <typename Value>
std::vector<Value> read_values(const std::filesystem::path &path) {
  const std::uintmax_t bytes = std::filesystem::file_size(path);
  if (bytes % sizeof(Value) != 0U ||
      bytes / sizeof(Value) > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("binary value file has an invalid size: " +
                                path.string());
  }
  std::vector<Value> values(static_cast<std::size_t>(bytes / sizeof(Value)));
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("could not open " + path.string());
  stream.read(reinterpret_cast<char *>(values.data()),
              static_cast<std::streamsize>(bytes));
  if (!stream)
    throw std::runtime_error("could not read " + path.string());
  return values;
}

std::uint32_t read_u32(const std::uint8_t *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t read_u64(const std::uint8_t *bytes) {
  return static_cast<std::uint64_t>(read_u32(bytes)) |
         (static_cast<std::uint64_t>(read_u32(bytes + 4)) << 32U);
}

std::vector<std::uint8_t> pread_bytes(const int descriptor,
                                      const std::uint64_t file_offset,
                                      const std::uint64_t byte_count) {
  if (byte_count > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("HDF5 byte range exceeds this process");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(byte_count));
  std::size_t completed = 0;
  while (completed < result.size()) {
    const auto count = pread(descriptor, result.data() + completed,
                             result.size() - completed,
                             static_cast<off_t>(file_offset + completed));
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("direct HDF5 range read failed");
    completed += static_cast<std::size_t>(count);
  }
  return result;
}

struct CacheResidency {
  bool known = false;
  std::uint64_t resident_bytes = 0;
  int error = 0;
};

CacheResidency cache_residency(const int descriptor,
                               const std::uint64_t file_bytes) {
  CacheResidency result;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || file_bytes > std::numeric_limits<std::size_t>::max()) {
    result.error = EOVERFLOW;
    return result;
  }
  void *mapping = mmap(nullptr, static_cast<std::size_t>(file_bytes),
                       PROT_READ, MAP_SHARED, descriptor, 0);
  if (mapping == MAP_FAILED) {
    result.error = errno;
    return result;
  }
  const std::size_t pages = static_cast<std::size_t>(
      (file_bytes + static_cast<std::uint64_t>(page_size) - 1U) /
      static_cast<std::uint64_t>(page_size));
  std::vector<unsigned char> status(pages);
  if (mincore(mapping, static_cast<std::size_t>(file_bytes), status.data()) !=
      0) {
    result.error = errno;
  } else {
    result.known = true;
    const auto resident_pages =
        std::count_if(status.begin(), status.end(),
                      [](const unsigned char value) {
                        return (value & 1U) != 0U;
                      });
    result.resident_bytes = std::min<std::uint64_t>(
        file_bytes, static_cast<std::uint64_t>(resident_pages) * page_size);
  }
  munmap(mapping, static_cast<std::size_t>(file_bytes));
  return result;
}

struct PackedH5Shard {
  std::uint64_t payload_offset = 0, payload_bytes = 0;
  std::uint64_t lengths_offset = 0, lengths_bytes = 0;
  std::uint64_t widths_offset = 0, widths_bytes = 0;
  std::uint64_t decoded_bytes = 0;
  std::uint32_t descriptor_count = 0, chunk_count = 0;
  std::array<std::uint8_t, 32> decoded_sha256{};
};

struct PackedH5Index {
  Shape4D shape{};
  std::uint32_t scans_per_shard = 0, chunk_bytes = 0;
  std::array<std::uint8_t, 32> source_identity{};
  std::vector<std::uint8_t> exclusions;
  std::vector<PackedH5Shard> shards;
};

PackedH5Index read_packed_h5_index(const int descriptor,
                                   const std::uint64_t file_bytes) {
  const auto prelude = pread_bytes(descriptor, 0U, 24U);
  const std::array<std::uint8_t, 8> container_magic{
      'Q', 'G', 'P', 'U', 'H', '5', 0, 1};
  if (!std::equal(container_magic.begin(), container_magic.end(),
                  prelude.begin()))
    throw std::invalid_argument("HDF5 file has no QuantEM GPU user-block index");
  const std::uint32_t binary_offset = read_u32(prelude.data() + 16U);
  const std::uint32_t binary_bytes = read_u32(prelude.data() + 20U);
  if (binary_offset < 24U || binary_bytes < 72U ||
      binary_offset > file_bytes || binary_bytes > file_bytes - binary_offset)
    throw std::invalid_argument("QuantEM GPU HDF5 index range is invalid");
  const auto bytes = pread_bytes(descriptor, binary_offset, binary_bytes);
  const std::array<std::uint8_t, 8> index_magic{
      'Q', 'G', 'I', 'X', 0, 0, 0, 1};
  if (!std::equal(index_magic.begin(), index_magic.end(), bytes.begin()))
    throw std::invalid_argument("QuantEM GPU HDF5 binary index version is invalid");
  std::size_t cursor = 8U;
  const std::uint32_t shard_count = read_u32(bytes.data() + cursor);
  cursor += 4U;
  PackedH5Index result;
  result.chunk_bytes = read_u32(bytes.data() + cursor);
  cursor += 4U;
  result.shape.scan_rows = read_u32(bytes.data() + cursor);
  cursor += 4U;
  result.shape.scan_columns = read_u32(bytes.data() + cursor);
  cursor += 4U;
  result.shape.detector_rows = read_u32(bytes.data() + cursor);
  cursor += 4U;
  result.shape.detector_columns = read_u32(bytes.data() + cursor);
  cursor += 4U;
  result.scans_per_shard = read_u32(bytes.data() + cursor);
  cursor += 4U;
  const std::uint32_t mask_count = read_u32(bytes.data() + cursor);
  cursor += 4U;
  if (shard_count != 64U || result.chunk_bytes != 128U ||
      result.shape.scan_rows != 512U || result.shape.scan_columns != 512U ||
      result.shape.detector_rows != 192U ||
      result.shape.detector_columns != 192U ||
      result.scans_per_shard != 4096U || mask_count > 192U * 192U ||
      cursor + static_cast<std::size_t>(mask_count) * 4U + 32U > bytes.size())
    throw std::invalid_argument("QuantEM GPU HDF5 shape or shard contract is invalid");
  result.exclusions.assign(192U * 192U, 0U);
  for (std::uint32_t index = 0; index < mask_count; ++index) {
    const std::uint32_t pixel = read_u32(bytes.data() + cursor);
    cursor += 4U;
    if (pixel >= result.exclusions.size() || result.exclusions[pixel] != 0U)
      throw std::invalid_argument("QuantEM GPU HDF5 detector mask is invalid");
    result.exclusions[pixel] = 1U;
  }
  std::copy_n(bytes.data() + cursor, result.source_identity.size(),
              result.source_identity.begin());
  cursor += result.source_identity.size();
  result.shards.resize(shard_count);
  for (auto &shard : result.shards) {
    if (cursor + 96U > bytes.size())
      throw std::invalid_argument("QuantEM GPU HDF5 shard index is truncated");
    shard.payload_offset = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.payload_bytes = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.lengths_offset = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.lengths_bytes = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.widths_offset = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.widths_bytes = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.decoded_bytes = read_u64(bytes.data() + cursor);
    cursor += 8U;
    shard.descriptor_count = read_u32(bytes.data() + cursor);
    cursor += 4U;
    shard.chunk_count = read_u32(bytes.data() + cursor);
    cursor += 4U;
    std::copy_n(bytes.data() + cursor, shard.decoded_sha256.size(),
                shard.decoded_sha256.begin());
    cursor += shard.decoded_sha256.size();
    for (const auto range : {
             std::pair{shard.payload_offset, shard.payload_bytes},
             std::pair{shard.lengths_offset, shard.lengths_bytes},
             std::pair{shard.widths_offset, shard.widths_bytes}}) {
      if (range.first > file_bytes || range.second > file_bytes - range.first)
        throw std::invalid_argument("QuantEM GPU HDF5 dataset range is invalid");
    }
    if (!shard.payload_bytes || shard.lengths_bytes != shard.chunk_count ||
        shard.widths_bytes != shard.descriptor_count ||
        shard.descriptor_count != 192U * 192U * 32U ||
        !shard.decoded_bytes || (shard.decoded_bytes & 3U) != 0U ||
        (shard.decoded_bytes + result.chunk_bytes - 1U) /
                result.chunk_bytes !=
            shard.chunk_count)
      throw std::invalid_argument("QuantEM GPU HDF5 shard metadata is invalid");
  }
  if (cursor != bytes.size())
    throw std::invalid_argument("QuantEM GPU HDF5 binary index has trailing bytes");
  return result;
}

std::vector<std::uint32_t>
reconstruct_descriptors(const std::vector<std::uint8_t> &widths,
                        const std::uint64_t decoded_bytes) {
  std::vector<std::uint32_t> descriptors(widths.size());
  std::uint32_t offset_words = 0U;
  for (std::size_t index = 0; index < widths.size(); ++index) {
    const std::uint32_t width = widths[index];
    if (width > 31U || offset_words >= (1U << 27U))
      throw std::invalid_argument("packed detector width or offset is invalid");
    descriptors[index] = (offset_words << 5U) | width;
    offset_words += width * 4U;
  }
  if (static_cast<std::uint64_t>(offset_words) * 4U != decoded_bytes)
    throw std::invalid_argument("packed detector widths do not cover the payload");
  return descriptors;
}

std::vector<std::uint32_t>
reconstruct_chunk_metadata(const std::vector<std::uint8_t> &lengths,
                           const std::uint32_t chunk_bytes,
                           const std::uint64_t decoded_bytes,
                           const std::uint64_t compressed_bytes) {
  std::vector<std::uint32_t> metadata(lengths.size() * 4U);
  std::uint64_t input_offset = 0U;
  std::uint64_t output_offset = 0U;
  for (std::size_t index = 0; index < lengths.size(); ++index) {
    const std::uint32_t input_bytes = lengths[index] + 1U;
    const std::uint32_t output_bytes = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(chunk_bytes, decoded_bytes - output_offset));
    if (input_offset > std::numeric_limits<std::uint32_t>::max() ||
        output_offset / 4U > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("packed HDF5 chunk offset exceeds uint32");
    metadata[index * 4U] = static_cast<std::uint32_t>(input_offset);
    metadata[index * 4U + 1U] = input_bytes;
    metadata[index * 4U + 2U] = static_cast<std::uint32_t>(output_offset / 4U);
    metadata[index * 4U + 3U] = output_bytes;
    input_offset += input_bytes;
    output_offset += output_bytes;
  }
  if (input_offset != compressed_bytes || output_offset != decoded_bytes)
    throw std::invalid_argument("packed HDF5 chunk lengths do not cover the datasets");
  return metadata;
}

int qh5_pack_shard(const std::vector<std::string> &arguments,
                   ExactProductExecutor &executor) {
  namespace fs = std::filesystem;
  const fs::path directory = parse_string(arguments, "--directory");
  const fs::path descriptor_path =
      parse_string(arguments, "--descriptors");
  const fs::path expected_path =
      parse_string(arguments, "--expected-payload");
  if (directory.empty() || !fs::is_directory(directory) ||
      !fs::is_regular_file(descriptor_path) ||
      !fs::is_regular_file(expected_path)) {
    throw std::invalid_argument(
        "qh5-pack-shard requires --directory, --descriptors, and "
        "--expected-payload files");
  }
  std::vector<fs::path> indexes;
  for (const auto &entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".qh5idx")
      indexes.push_back(entry.path());
  }
  std::sort(indexes.begin(), indexes.end());
  if (indexes.empty())
    throw std::invalid_argument("QH5 directory has no index files");
  struct OpenPair {
    int source = -1;
    int index = -1;
  };
  std::vector<OpenPair> open_files;
  std::vector<quantem::gpu::vulkan::Qh5IndexedSegment> segments;
  try {
    for (const fs::path &index_path : indexes) {
      fs::path source_path = index_path;
      source_path.replace_extension(".h5");
      if (!fs::is_regular_file(source_path))
        throw std::invalid_argument("missing source for " +
                                    index_path.string());
      OpenPair pair;
      pair.source = open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
      pair.index = open(index_path.c_str(), O_RDONLY | O_CLOEXEC);
      if (pair.source < 0 || pair.index < 0) {
        if (pair.source >= 0)
          close(pair.source);
        if (pair.index >= 0)
          close(pair.index);
        throw std::runtime_error("could not open QH5 source/index pair");
      }
      const auto source_bytes = fs::file_size(source_path);
      const auto index_bytes = fs::file_size(index_path);
      open_files.push_back(pair);
      segments.push_back(
          {pair.source, 0, source_bytes, pair.index, 0, index_bytes});
    }
    auto source = quantem::gpu::vulkan::Qh5IndexedSource::open(
        segments, Shape4D{512, 512, 192, 192});
    for (auto &pair : open_files) {
      close(pair.source);
      close(pair.index);
      pair.source = -1;
      pair.index = -1;
    }

    constexpr std::uint32_t shard_scans = 4096U;
    constexpr std::uint32_t detector_pixels = 192U * 192U;
    constexpr std::uint32_t tile_count =
        shard_scans / quantem::gpu::vulkan::PackedDetectorShard::scan_tile;
    const std::uint32_t shard_index =
        parse_uint(arguments, "--shard-index", 0U);
    const std::uint64_t first_frame =
        static_cast<std::uint64_t>(shard_index) * shard_scans;
    if (first_frame > source->frame_count() ||
        shard_scans > source->frame_count() - first_frame) {
      throw std::invalid_argument("packed shard index is outside the scan");
    }
    const auto descriptors = read_values<std::uint32_t>(descriptor_path);
    auto expected = read_values<std::uint32_t>(expected_path);
    const std::vector<std::uint8_t> exclusions =
        parse_mask_pixels(arguments, Shape4D{512, 512, 192, 192});
    if (descriptors.size() !=
        static_cast<std::size_t>(detector_pixels) * tile_count) {
      throw std::invalid_argument(
          "packed descriptor file does not describe one 4096-frame shard");
    }
    for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
      if (exclusions[pixel] == 0U)
        continue;
      for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        const std::uint32_t descriptor =
            descriptors[static_cast<std::size_t>(pixel) * tile_count + tile];
        const std::uint32_t offset = descriptor >> 5U;
        const std::uint32_t words = (descriptor & 31U) * 4U;
        if (offset > expected.size() || words > expected.size() - offset) {
          throw std::invalid_argument(
              "excluded packed descriptor escapes the expected payload");
        }
        std::fill(expected.begin() + offset,
                  expected.begin() + offset + words, 0U);
      }
    }
    quantem::gpu::vulkan::validate_packed_detector_shard(
        shard_scans, detector_pixels, descriptors, expected);
    std::vector<std::uint32_t> actual;
    const auto metrics = executor.pack_indexed_qh5_audited_low8_shard(
        *source, first_frame, shard_scans, descriptors,
        static_cast<std::uint32_t>(expected.size()), exclusions, &actual);
    std::uint64_t mismatch_count = 0U;
    for (std::size_t index = 0; index < expected.size(); ++index) {
      mismatch_count += expected[index] != actual[index];
    }
    const fs::path output = parse_string(arguments, "--output");
    if (!output.empty())
      write_values(output, actual);
    std::cout << std::boolalpha << std::fixed << std::setprecision(4) << '{'
              << "\"schema\":\"quantem.gpu.android-vulkan-qh5-pack-shard/v1\","
              << "\"status\":\"" << (mismatch_count == 0U ? "PASS" : "FAIL")
              << "\",\"real_data\":true,\"source_shape\":[512,512,192,192],"
              << "\"source_dtype\":\"uint16\",\"working_dtype\":\"uint8\","
              << "\"shard_index\":" << shard_index << ','
              << "\"frame_count\":" << shard_scans << ','
              << "\"descriptor_bytes\":" << descriptors.size() * 4ULL << ','
              << "\"payload_bytes\":" << actual.size() * 4ULL << ','
              << "\"storage_read_ms\":" << metrics.storage_read_milliseconds
              << ',' << "\"source_staging_ms\":"
              << metrics.source_staging_milliseconds << ','
              << "\"vulkan_visibility_ms\":"
              << metrics.vulkan_visibility_milliseconds << ','
              << "\"gpu_decode_ms\":" << metrics.gpu_decode_milliseconds
              << ',' << "\"gpu_pack_ms\":"
              << metrics.gpu_pack_milliseconds << ','
              << "\"gpu_decode_and_pack_ms\":"
              << metrics.gpu_decode_and_pack_milliseconds << ','
              << "\"ready_ms\":" << metrics.ready_milliseconds << ','
              << "\"source_bytes_read\":" << metrics.source_bytes_read << ','
              << "\"compressed_bytes_staged\":"
              << metrics.compressed_bytes_staged << ','
              << "\"vulkan_committed_bytes\":"
              << metrics.vulkan_committed_bytes << ','
              << "\"mismatch_word_count\":" << mismatch_count << ','
              << "\"output_written\":" << !output.empty() << "}\n";
    return mismatch_count == 0U ? 0 : 2;
  } catch (...) {
    for (auto &pair : open_files) {
      if (pair.source >= 0)
        close(pair.source);
      if (pair.index >= 0)
        close(pair.index);
    }
    throw;
  }
}

int packed_lz4_shard(const std::vector<std::string> &arguments,
                     ExactProductExecutor &executor) {
  namespace fs = std::filesystem;
  const fs::path compressed_path = parse_string(arguments, "--compressed");
  const fs::path index_path = parse_string(arguments, "--index");
  const fs::path descriptor_path = parse_string(arguments, "--descriptors");
  const fs::path expected_path =
      parse_string(arguments, "--expected-payload");
  for (const auto &path : {compressed_path, index_path, descriptor_path,
                           expected_path}) {
    if (!fs::is_regular_file(path))
      throw std::invalid_argument(
          "packed-lz4-shard requires all compressed/index/oracle files");
  }
  const auto source_started = std::chrono::steady_clock::now();
  const auto compressed = read_values<std::uint8_t>(compressed_path);
  const auto metadata = read_values<std::uint32_t>(index_path);
  const double source_read_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - source_started)
          .count();
  const auto oracle_started = std::chrono::steady_clock::now();
  const auto descriptors = read_values<std::uint32_t>(descriptor_path);
  auto expected = read_values<std::uint32_t>(expected_path);
  const double oracle_read_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - oracle_started)
          .count();
  const auto exclusions =
      parse_mask_pixels(arguments, Shape4D{512, 512, 192, 192});
  constexpr std::uint32_t detector_pixels = 192U * 192U;
  constexpr std::uint32_t tile_count = 32U;
  if (descriptors.size() !=
      static_cast<std::size_t>(detector_pixels) * tile_count) {
    throw std::invalid_argument("packed LZ4 oracle descriptors are invalid");
  }
  for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
    if (exclusions[pixel] == 0U)
      continue;
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
      const std::uint32_t descriptor =
          descriptors[static_cast<std::size_t>(pixel) * tile_count + tile];
      const std::uint32_t offset = descriptor >> 5U;
      const std::uint32_t words = (descriptor & 31U) * 4U;
      std::fill(expected.begin() + offset,
                expected.begin() + offset + words, 0U);
    }
  }
  std::vector<std::uint32_t> actual;
  const auto metrics = executor.decode_packed_lz4(
      compressed, metadata, static_cast<std::uint32_t>(expected.size()),
      &actual);
  std::uint64_t mismatch_words = 0U;
  for (std::size_t index = 0; index < expected.size(); ++index)
    mismatch_words += expected[index] != actual[index];
  const fs::path output = parse_string(arguments, "--output");
  if (!output.empty())
    write_values(output, actual);
  std::cout << std::boolalpha << std::fixed << std::setprecision(4) << '{'
            << "\"schema\":\"quantem.gpu.android-vulkan-packed-lz4-shard/v1\","
            << "\"status\":\"" << (mismatch_words == 0U ? "PASS" : "FAIL")
            << "\",\"real_data\":true,\"working_dtype\":\"uint8\","
            << "\"source_read_ms\":" << source_read_milliseconds << ','
            << "\"oracle_read_ms\":" << oracle_read_milliseconds << ','
            << "\"compressed_bytes\":" << compressed.size() << ','
            << "\"decoded_bytes\":" << metrics.decoded_bytes << ','
            << "\"chunk_count\":" << metadata.size() / 4U << ','
            << "\"vulkan_visibility_ms\":"
            << metrics.vulkan_visibility_milliseconds << ','
            << "\"gpu_decode_ms\":" << metrics.gpu_decode_milliseconds << ','
            << "\"ready_after_read_ms\":" << metrics.ready_milliseconds
            << ',' << "\"source_to_ready_ms\":"
            << source_read_milliseconds + metrics.ready_milliseconds << ','
            << "\"vulkan_committed_bytes\":"
            << metrics.vulkan_committed_bytes << ','
            << "\"mismatch_word_count\":" << mismatch_words << ','
            << "\"output_written\":" << !output.empty() << "}\n";
  return mismatch_words == 0U ? 0 : 2;
}

std::string packed_shard_name(std::uint32_t shard,
                              const std::string &suffix);

int packed_h5_resident(const std::vector<std::string> &arguments,
                       ExactProductExecutor &executor) {
  (void)executor;
  namespace fs = std::filesystem;
  const fs::path source_path = parse_string(arguments, "--source");
  const fs::path oracle_directory =
      parse_string(arguments, "--oracle-directory");
  const bool asynchronous = has_flag(arguments, "--async");
  const bool audit_oracle = !has_flag(arguments, "--skip-oracle");
  if (!fs::is_regular_file(source_path) ||
      (audit_oracle && !fs::is_directory(oracle_directory)))
    throw std::invalid_argument(
        "packed-h5-resident requires --source H5 and --oracle-directory");
  const std::uint64_t file_bytes = fs::file_size(source_path);
  const int descriptor = open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    throw std::runtime_error("could not open GPU-native HDF5 source");
  try {
    const CacheResidency cache_before =
        cache_residency(descriptor, file_bytes);
    const bool drop_cache = has_flag(arguments, "--drop-cache");
    const int drop_cache_result = drop_cache
        ? posix_fadvise(descriptor, 0, 0, POSIX_FADV_DONTNEED)
        : 0;
    const CacheResidency cache_after_advice =
        cache_residency(descriptor, file_bytes);
    const auto index_started = std::chrono::steady_clock::now();
    const PackedH5Index index = read_packed_h5_index(descriptor, file_bytes);
    const double index_read_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - index_started)
            .count();
    std::vector<quantem::gpu::vulkan::PackedDetectorShardSize> plan;
    plan.reserve(index.shards.size());
    for (const auto &shard : index.shards) {
      if (shard.decoded_bytes / 4U >
          std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("packed HDF5 payload exceeds uint32 words");
      plan.push_back(
          {index.scans_per_shard,
           static_cast<std::uint32_t>(shard.decoded_bytes / 4U),
           static_cast<std::uint32_t>(shard.payload_bytes),
           shard.chunk_count});
    }
    double source_read_milliseconds = 0.0;
    double metadata_reconstruction_milliseconds = 0.0;
    double decode_staging_milliseconds = 0.0;
    double decode_visibility_milliseconds = 0.0;
    double gpu_decode_milliseconds = 0.0;
    double decode_ready_milliseconds = 0.0;
    double oracle_read_milliseconds = 0.0;
    double oracle_compare_milliseconds = 0.0;
    double session_copy_milliseconds = 0.0;
    std::uint64_t source_bytes_read = 0U;
    std::uint64_t mismatch_words = 0U;
    std::uint64_t mismatch_descriptors = 0U;
    const auto resident_started = std::chrono::steady_clock::now();
    quantem::gpu::vulkan::PackedDetectorSession session(
        index.shape, plan,
        [](const std::size_t, const std::uint64_t) {},
        [&](const std::size_t shard_index,
            const quantem::gpu::vulkan::PackedDetectorShardDestination
                destination) {
          const auto &shard = index.shards.at(shard_index);
          const auto source_started = std::chrono::steady_clock::now();
          const auto compressed = pread_bytes(
              descriptor, shard.payload_offset, shard.payload_bytes);
          const auto lengths = pread_bytes(
              descriptor, shard.lengths_offset, shard.lengths_bytes);
          const auto widths = pread_bytes(
              descriptor, shard.widths_offset, shard.widths_bytes);
          source_read_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - source_started)
                  .count();
          source_bytes_read += compressed.size() + lengths.size() +
                               widths.size();
          const auto metadata_started = std::chrono::steady_clock::now();
          const auto descriptors =
              reconstruct_descriptors(widths, shard.decoded_bytes);
          const auto metadata = reconstruct_chunk_metadata(
              lengths, index.chunk_bytes, shard.decoded_bytes,
              shard.payload_bytes);
          metadata_reconstruction_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - metadata_started)
                  .count();
          if (!destination.decode_lz4_to_words ||
              !destination.enqueue_lz4_to_words)
            throw std::logic_error(
                "packed HDF5 plan did not admit direct GPU decode");
          const auto metrics = asynchronous
              ? destination.enqueue_lz4_to_words(compressed, metadata)
              : destination.decode_lz4_to_words(compressed, metadata);
          decode_staging_milliseconds += metrics.staging_milliseconds;
          decode_visibility_milliseconds +=
              metrics.vulkan_visibility_milliseconds;
          gpu_decode_milliseconds += metrics.gpu_decode_milliseconds;
          decode_ready_milliseconds += metrics.ready_milliseconds;

          if (audit_oracle) {
            if (asynchronous)
              throw std::invalid_argument(
                  "Use synchronous decode for the full payload oracle audit");
            const auto oracle_started = std::chrono::steady_clock::now();
            const fs::path oracle_descriptors =
                oracle_directory /
                packed_shard_name(static_cast<std::uint32_t>(shard_index),
                                  "_descriptors_u32_le.bin");
            const fs::path oracle_payload =
                oracle_directory /
                packed_shard_name(static_cast<std::uint32_t>(shard_index),
                                  "_payload_u32_le.bin");
            const auto expected_descriptors =
                read_values<std::uint32_t>(oracle_descriptors);
            auto expected = read_values<std::uint32_t>(oracle_payload);
            oracle_read_milliseconds +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - oracle_started)
                    .count();
            const auto compare_started = std::chrono::steady_clock::now();
            if (expected_descriptors.size() != descriptors.size() ||
                expected.size() != destination.words.size())
              throw std::runtime_error("packed HDF5 oracle size differs");
            for (std::size_t offset = 0; offset < descriptors.size(); ++offset)
              mismatch_descriptors +=
                  descriptors[offset] != expected_descriptors[offset];
            for (std::uint32_t pixel = 0; pixel < 192U * 192U; ++pixel) {
              if (index.exclusions[pixel] == 0U)
                continue;
              for (std::uint32_t tile = 0; tile < 32U; ++tile) {
                const std::uint32_t packed = descriptors[
                    static_cast<std::size_t>(pixel) * 32U + tile];
                const std::uint32_t offset = packed >> 5U;
                const std::uint32_t words = (packed & 31U) * 4U;
                std::fill(expected.begin() + offset,
                          expected.begin() + offset + words, 0U);
              }
            }
            for (std::size_t offset = 0; offset < expected.size(); ++offset)
              mismatch_words += expected[offset] != destination.words[offset];
            oracle_compare_milliseconds +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - compare_started)
                    .count();
            if (mismatch_words != 0U || mismatch_descriptors != 0U)
              throw std::runtime_error(
                  "packed HDF5 parity failed at shard " +
                  std::to_string(shard_index));
          }
          if (destination.descriptors.size() != descriptors.size())
            throw std::logic_error("packed HDF5 resident destination differs");
          const auto copy_started = std::chrono::steady_clock::now();
          std::memcpy(destination.descriptors.data(), descriptors.data(),
                      descriptors.size() * sizeof(std::uint32_t));
          session_copy_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - copy_started)
                  .count();
        },
        index.exclusions, 4ULL * 1024ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL);
    const double resident_wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - resident_started)
            .count();
    std::vector<std::uint32_t> image(index.shape.scan_count());
    const auto interaction = session.request(
        {95.5F, 95.5F, 0.0F, 30.5F}, 1U, image);
    const auto &admission = session.admission();
    const auto &load = admission.load_timing;
    gpu_decode_milliseconds = load.compressed_gpu_decode_milliseconds;
    const std::uint64_t image_sum =
        std::accumulate(image.begin(), image.end(), std::uint64_t{0});
    const double ready_excluding_oracle =
        resident_wall_milliseconds - oracle_read_milliseconds -
        oracle_compare_milliseconds;
    const std::string expected_bf_text =
        parse_string(arguments, "--expected-bf-sum");
    const std::uint64_t expected_bf_sum = expected_bf_text.empty()
        ? 0U
        : std::stoull(expected_bf_text);
    if (asynchronous && expected_bf_sum == 0U)
      throw std::invalid_argument(
          "asynchronous resident load requires --expected-bf-sum");
    const bool first_bf_matches =
        expected_bf_sum == 0U || image_sum == expected_bf_sum;
    close(descriptor);
    std::cout << std::boolalpha << std::fixed << std::setprecision(4) << '{'
              << "\"schema\":\"quantem.gpu.android-vulkan-packed-h5-resident/v1\","
              << "\"status\":\"" << (first_bf_matches ? "PASS" : "FAIL")
              << "\",\"real_data\":true,"
              << "\"asynchronous_decode\":" << asynchronous << ','
              << "\"full_oracle_checked\":" << audit_oracle << ','
              << "\"source_shape\":[512,512,192,192],"
              << "\"source_dtype\":\"uint16\",\"working_dtype\":\"uint8\","
              << "\"hdf5_file_bytes\":" << file_bytes << ','
              << "\"drop_cache_requested\":" << drop_cache << ','
              << "\"drop_cache_result\":" << drop_cache_result << ','
              << "\"cache_residency_known_before\":"
              << cache_before.known << ','
              << "\"cache_resident_bytes_before\":"
              << cache_before.resident_bytes << ','
              << "\"cache_residency_error_before\":"
              << cache_before.error << ','
              << "\"cache_residency_known_after_advice\":"
              << cache_after_advice.known << ','
              << "\"cache_resident_bytes_after_advice\":"
              << cache_after_advice.resident_bytes << ','
              << "\"cache_residency_error_after_advice\":"
              << cache_after_advice.error << ','
              << "\"shard_count\":" << index.shards.size() << ','
              << "\"index_read_ms\":" << index_read_milliseconds << ','
              << "\"source_read_ms\":" << source_read_milliseconds << ','
              << "\"source_bytes_read\":" << source_bytes_read << ','
              << "\"metadata_reconstruction_ms\":"
              << metadata_reconstruction_milliseconds << ','
              << "\"decode_staging_ms\":" << decode_staging_milliseconds
              << ',' << "\"decode_visibility_ms\":"
              << decode_visibility_milliseconds << ','
              << "\"gpu_decode_ms\":" << gpu_decode_milliseconds << ','
              << "\"decode_ready_sum_ms\":" << decode_ready_milliseconds
              << ',' << "\"decode_enqueue_sum_ms\":"
              << load.compressed_enqueue_milliseconds
              << ',' << "\"decode_wait_sum_ms\":"
              << load.compressed_wait_milliseconds
              << ',' << "\"oracle_read_ms\":" << oracle_read_milliseconds
              << ',' << "\"oracle_compare_ms\":"
              << oracle_compare_milliseconds << ','
              << "\"session_copy_ms\":" << session_copy_milliseconds << ','
              << "\"resident_wall_ms_with_oracle\":"
              << resident_wall_milliseconds << ','
              << "\"resident_ready_excluding_oracle_ms\":"
              << ready_excluding_oracle << ','
              << "\"mismatch_descriptor_count\":" << mismatch_descriptors
              << ',' << "\"mismatch_word_count\":" << mismatch_words << ','
              << "\"resident_source_upload_bytes\":"
              << admission.source_upload_bytes << ','
              << "\"resident_committed_bytes\":" << admission.committed_bytes
              << ',' << "\"device_creation_ms\":"
              << load.device_creation_milliseconds
              << ',' << "\"plan_admission_ms\":"
              << load.plan_admission_milliseconds
              << ',' << "\"work_allocation_ms\":"
              << load.work_allocation_milliseconds
              << ',' << "\"source_allocation_mapping_ms\":"
              << load.source_allocation_mapping_milliseconds
              << ',' << "\"shard_validation_ms\":"
              << load.shard_validation_milliseconds
              << ',' << "\"source_flush_ms\":"
              << load.source_flush_milliseconds
              << ',' << "\"descriptor_cost_scan_ms\":"
              << load.descriptor_cost_scan_milliseconds
              << ',' << "\"interaction_pipeline_setup_ms\":"
              << load.pipeline_descriptor_creation_milliseconds +
                     load.command_creation_milliseconds
              << ',' << "\"resident_budget_check_ms\":"
              << load.resident_budget_check_milliseconds
              << ',' << "\"load_other_ms\":" << load.other_milliseconds
              << ',' << "\"first_bf_wall_ms\":"
              << interaction.timing.wall_milliseconds << ','
              << "\"first_bf_gpu_ms\":"
              << interaction.timing.gpu_total_milliseconds << ','
              << "\"first_bf_image_sum\":" << image_sum << "}\n";
    return first_bf_matches ? 0 : 2;
  } catch (...) {
    close(descriptor);
    throw;
  }
}

std::string packed_shard_name(const std::uint32_t shard,
                              const std::string &suffix) {
  std::ostringstream name;
  name << "shard_" << std::setw(3) << std::setfill('0') << shard << suffix;
  return name.str();
}

int qh5_resident(const std::vector<std::string> &arguments,
                 ExactProductExecutor &executor) {
  namespace fs = std::filesystem;
  const fs::path directory = parse_string(arguments, "--directory");
  const fs::path packed_directory =
      parse_string(arguments, "--packed-directory", directory.string());
  if (directory.empty() || !fs::is_directory(directory) ||
      !fs::is_directory(packed_directory)) {
    throw std::invalid_argument(
        "qh5-resident requires HDF5 --directory and --packed-directory");
  }
  std::vector<fs::path> indexes;
  for (const auto &entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".qh5idx")
      indexes.push_back(entry.path());
  }
  std::sort(indexes.begin(), indexes.end());
  if (indexes.empty())
    throw std::invalid_argument("QH5 directory has no index files");
  struct OpenPair {
    int source = -1;
    int index = -1;
  };
  std::vector<OpenPair> open_files;
  std::vector<quantem::gpu::vulkan::Qh5IndexedSegment> segments;
  try {
    for (const fs::path &index_path : indexes) {
      fs::path source_path = index_path;
      source_path.replace_extension(".h5");
      OpenPair pair;
      pair.source = open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
      pair.index = open(index_path.c_str(), O_RDONLY | O_CLOEXEC);
      if (pair.source < 0 || pair.index < 0) {
        if (pair.source >= 0)
          close(pair.source);
        if (pair.index >= 0)
          close(pair.index);
        throw std::runtime_error("could not open QH5 source/index pair");
      }
      const auto source_bytes = fs::file_size(source_path);
      const auto index_bytes = fs::file_size(index_path);
      open_files.push_back(pair);
      segments.push_back(
          {pair.source, 0, source_bytes, pair.index, 0, index_bytes});
    }
    const Shape4D shape{512, 512, 192, 192};
    auto source = quantem::gpu::vulkan::Qh5IndexedSource::open(segments,
                                                               shape);
    for (auto &pair : open_files) {
      close(pair.source);
      close(pair.index);
      pair.source = -1;
      pair.index = -1;
    }

    constexpr std::uint32_t shard_count = 64U;
    constexpr std::uint32_t shard_scans = 4096U;
    constexpr std::uint32_t detector_pixels = 192U * 192U;
    constexpr std::uint32_t tile_count = 32U;
    std::vector<quantem::gpu::vulkan::PackedDetectorShardSize> plan;
    plan.reserve(shard_count);
    for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
      const fs::path descriptor_path = packed_directory /
          packed_shard_name(shard, "_descriptors_u32_le.bin");
      const fs::path payload_path = packed_directory /
          packed_shard_name(shard, "_payload_u32_le.bin");
      const auto descriptor_bytes = fs::file_size(descriptor_path);
      const auto payload_bytes = fs::file_size(payload_path);
      if (descriptor_bytes !=
              static_cast<std::uintmax_t>(detector_pixels) * tile_count * 4U ||
          payload_bytes % 4U != 0U ||
          payload_bytes / 4U > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "packed resident shard files do not match the 4096-frame plan");
      }
      plan.push_back({shard_scans,
                      static_cast<std::uint32_t>(payload_bytes / 4U)});
    }
    const std::vector<std::uint8_t> exclusions =
        parse_mask_pixels(arguments, shape);
    double oracle_read_milliseconds = 0.0;
    double oracle_compare_milliseconds = 0.0;
    double session_copy_milliseconds = 0.0;
    double qh5_storage_milliseconds = 0.0;
    double qh5_staging_milliseconds = 0.0;
    double qh5_gpu_milliseconds = 0.0;
    double qh5_transcode_ready_milliseconds = 0.0;
    std::uint64_t qh5_source_bytes_read = 0U;
    std::uint64_t qh5_compressed_bytes_staged = 0U;
    std::uint64_t mismatch_words = 0U;
    const auto resident_started = std::chrono::steady_clock::now();
    quantem::gpu::vulkan::PackedDetectorSession session(
        shape, plan,
        [](const std::size_t, const std::uint64_t) {},
        [&](const std::size_t shard,
            const quantem::gpu::vulkan::PackedDetectorShardDestination
                destination) {
          const fs::path descriptor_path = packed_directory /
              packed_shard_name(static_cast<std::uint32_t>(shard),
                                "_descriptors_u32_le.bin");
          const fs::path payload_path = packed_directory /
              packed_shard_name(static_cast<std::uint32_t>(shard),
                                "_payload_u32_le.bin");
          const auto oracle_started = std::chrono::steady_clock::now();
          const auto descriptors = read_values<std::uint32_t>(descriptor_path);
          auto expected = read_values<std::uint32_t>(payload_path);
          oracle_read_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - oracle_started)
                  .count();
          for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
            if (exclusions[pixel] == 0U)
              continue;
            for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
              const std::uint32_t descriptor =
                  descriptors[static_cast<std::size_t>(pixel) * tile_count +
                              tile];
              const std::uint32_t offset = descriptor >> 5U;
              const std::uint32_t words = (descriptor & 31U) * 4U;
              std::fill(expected.begin() + offset,
                        expected.begin() + offset + words, 0U);
            }
          }
          std::vector<std::uint32_t> actual;
          const auto metrics = executor.pack_indexed_qh5_audited_low8_shard(
              *source, shard * shard_scans, shard_scans, descriptors,
              static_cast<std::uint32_t>(expected.size()), exclusions,
              &actual);
          qh5_storage_milliseconds += metrics.storage_read_milliseconds;
          qh5_staging_milliseconds += metrics.source_staging_milliseconds;
          qh5_gpu_milliseconds += metrics.gpu_decode_and_pack_milliseconds;
          qh5_transcode_ready_milliseconds += metrics.ready_milliseconds;
          qh5_source_bytes_read += metrics.source_bytes_read;
          qh5_compressed_bytes_staged += metrics.compressed_bytes_staged;
          const auto compare_started = std::chrono::steady_clock::now();
          for (std::size_t index = 0; index < expected.size(); ++index)
            mismatch_words += expected[index] != actual[index];
          oracle_compare_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - compare_started)
                  .count();
          if (mismatch_words != 0U)
            throw std::runtime_error(
                "full resident QH5 payload parity failed at shard " +
                std::to_string(shard));
          if (destination.descriptors.size() != descriptors.size() ||
              destination.words.size() != actual.size()) {
            throw std::logic_error(
                "resident session destination does not match the shard plan");
          }
          const auto copy_started = std::chrono::steady_clock::now();
          std::memcpy(destination.descriptors.data(), descriptors.data(),
                      descriptors.size() * sizeof(std::uint32_t));
          std::memcpy(destination.words.data(), actual.data(),
                      actual.size() * sizeof(std::uint32_t));
          session_copy_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - copy_started)
                  .count();
        },
        exclusions, 4ULL * 1024ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL);
    const double resident_wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - resident_started)
            .count();
    std::vector<std::uint32_t> image(shape.scan_count());
    const auto interaction = session.request(
        {95.5F, 95.5F, 0.0F, 30.5F}, 1U, image);
    const std::uint64_t image_sum =
        std::accumulate(image.begin(), image.end(), std::uint64_t{0});
    const auto &admission = session.admission();
    const double ready_excluding_oracle =
        resident_wall_milliseconds - oracle_read_milliseconds -
        oracle_compare_milliseconds;
    std::cout << std::boolalpha << std::fixed << std::setprecision(4) << '{'
              << "\"schema\":\"quantem.gpu.android-vulkan-qh5-resident/v1\","
              << "\"status\":\"PASS\",\"real_data\":true,"
              << "\"source_shape\":[512,512,192,192],"
              << "\"source_dtype\":\"uint16\",\"working_dtype\":\"uint8\","
              << "\"shard_count\":" << shard_count << ','
              << "\"mismatch_word_count\":" << mismatch_words << ','
              << "\"resident_wall_ms_with_oracle\":"
              << resident_wall_milliseconds << ','
              << "\"resident_ready_excluding_oracle_ms\":"
              << ready_excluding_oracle << ','
              << "\"oracle_read_ms\":" << oracle_read_milliseconds << ','
              << "\"oracle_compare_ms\":" << oracle_compare_milliseconds
              << ',' << "\"session_copy_ms\":"
              << session_copy_milliseconds << ','
              << "\"qh5_storage_read_ms\":" << qh5_storage_milliseconds
              << ',' << "\"qh5_source_staging_ms\":"
              << qh5_staging_milliseconds << ','
              << "\"qh5_gpu_decode_pack_ms\":" << qh5_gpu_milliseconds
              << ',' << "\"qh5_transcode_ready_sum_ms\":"
              << qh5_transcode_ready_milliseconds << ','
              << "\"qh5_source_bytes_read\":" << qh5_source_bytes_read
              << ',' << "\"qh5_compressed_bytes_staged\":"
              << qh5_compressed_bytes_staged << ','
              << "\"resident_source_upload_bytes\":"
              << admission.source_upload_bytes << ','
              << "\"resident_committed_bytes\":"
              << admission.committed_bytes << ','
              << "\"first_bf_wall_ms\":"
              << interaction.timing.wall_milliseconds << ','
              << "\"first_bf_gpu_ms\":"
              << interaction.timing.gpu_total_milliseconds << ','
              << "\"first_bf_image_sum\":" << image_sum << "}\n";
    return 0;
  } catch (...) {
    for (auto &pair : open_files) {
      if (pair.source >= 0)
        close(pair.source);
      if (pair.index >= 0)
        close(pair.index);
    }
    throw;
  }
}

int qh5_real(const std::vector<std::string> &arguments,
             ExactProductExecutor &executor) {
  namespace fs = std::filesystem;
  const fs::path directory = parse_string(arguments, "--directory");
  if (directory.empty() || !fs::is_directory(directory))
    throw std::invalid_argument(
        "qh5-real requires an existing --directory with paired .h5/.qh5idx files");
  std::vector<fs::path> indexes;
  for (const auto &entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".qh5idx")
      indexes.push_back(entry.path());
  }
  std::sort(indexes.begin(), indexes.end());
  if (indexes.empty())
    throw std::invalid_argument("qh5-real directory has no QH5 indexes");

  struct OpenPair {
    int source = -1;
    int index = -1;
    std::uint64_t source_bytes = 0;
    std::uint64_t index_bytes = 0;
  };
  std::vector<OpenPair> open_files;
  std::vector<quantem::gpu::vulkan::Qh5IndexedSegment> segments;
  try {
    for (const fs::path &index_path : indexes) {
      fs::path source_path = index_path;
      source_path.replace_extension(".h5");
      if (!fs::is_regular_file(source_path))
        throw std::invalid_argument("missing source for " + index_path.string());
      OpenPair pair;
      pair.source = open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
      pair.index = open(index_path.c_str(), O_RDONLY | O_CLOEXEC);
      if (pair.source < 0 || pair.index < 0)
        throw std::runtime_error("could not open QH5 source/index pair");
      pair.source_bytes = fs::file_size(source_path);
      pair.index_bytes = fs::file_size(index_path);
      open_files.push_back(pair);
      segments.push_back({pair.source, 0, pair.source_bytes,
                          pair.index, 0, pair.index_bytes});
    }
    auto source = quantem::gpu::vulkan::Qh5IndexedSource::open(
        segments, Shape4D{512, 512, 192, 192});
    for (auto &pair : open_files) {
      close(pair.source);
      close(pair.index);
      pair.source = -1;
      pair.index = -1;
    }

    BenchmarkOptions options;
    options.source_shape = {512, 512, 192, 192};
    options.shard_scan_rows = parse_uint(arguments, "--shard-rows", 2);
    options.staging_ring_depth = parse_uint(arguments, "--ring-depth", 2);
    options.wait_after_each_shard = has_flag(arguments, "--wait-each");
    options.selected_scan_row = parse_uint(arguments, "--selected-row", 256);
    options.selected_scan_column =
        parse_uint(arguments, "--selected-column", 256);
    const bool audited_low8 = has_flag(arguments, "--audited-low8");
    const std::vector<std::uint8_t> exclusions =
        parse_mask_pixels(arguments, options.source_shape);
    ExactProducts products;
    const auto package_started = std::chrono::steady_clock::now();
    BenchmarkMetrics metrics = audited_low8
        ? executor.run_indexed_qh5_audited_low8(
              options, *source, detector_bands(options.source_shape),
              exclusions, &products)
        : executor.run_indexed_qh5_u16(
              options, *source, detector_bands(options.source_shape),
              &products);
    const auto derived = quantem::gpu::vulkan::derive_products(products);
    const auto dpc_started = std::chrono::steady_clock::now();
    const auto dpc = quantem::gpu::vulkan::derive_dpc(derived);
    metrics.dpc_and_idpc_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - dpc_started).count();
    metrics.package_ready_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - package_started).count();

    const fs::path output = parse_string(arguments, "--output-dir");
    if (!output.empty()) {
      fs::create_directories(output);
      write_values(output / "total_intensity_u64_le.bin",
                   products.total_intensity);
      write_values(output / "band1_u64_le.bin", products.band1);
      write_values(output / "band2_u64_le.bin", products.band2);
      write_values(output / "band4_u64_le.bin", products.band4);
      write_values(output / "detector_row_moment_u64_le.bin",
                   products.detector_row_moment);
      write_values(output / "detector_column_moment_u64_le.bin",
                   products.detector_column_moment);
      write_values(output / "diffraction_sum_u64_le.bin",
                   products.diffraction_sum);
      if (audited_low8) {
        write_values(output / "selected_masked_u8.bin",
                     products.selected_diffraction_uint8);
      } else {
        write_values(output / "selected_raw_u16_le.bin",
                     products.selected_diffraction_uint16);
      }
    }
    std::uint64_t source_file_bytes = 0;
    for (const auto &pair : open_files)
      source_file_bytes += pair.source_bytes;
    std::cout << std::boolalpha << std::fixed << std::setprecision(4) << '{'
              << "\"schema\":\"quantem.gpu.android-vulkan-qh5-real/v1\","
              << "\"status\":\"PASS\",\"real_data\":true,"
              << "\"source_shape\":[512,512,192,192],"
              << "\"source_dtype\":\"uint16\","
              << "\"working_dtype\":\""
              << (audited_low8 ? "uint8" : "uint16") << "\","
              << "\"audited_low8\":" << audited_low8 << ','
              << "\"source_file_count\":" << indexes.size() << ','
              << "\"source_file_bytes\":" << source_file_bytes << ','
              << "\"storage_read_ms\":" << metrics.storage_read_milliseconds << ','
              << "\"source_decode_gpu_ms\":" << metrics.source_decode_milliseconds << ','
              << "\"source_lz4_gpu_ms\":" << metrics.gpu_source_lz4_milliseconds << ','
              << "\"source_bitunshuffle_gpu_ms\":" << metrics.gpu_source_bitunshuffle_milliseconds << ','
              << "\"source_staging_ms\":" << metrics.source_staging_milliseconds << ','
              << "\"vulkan_visibility_ms\":" << metrics.vulkan_visibility_milliseconds << ','
              << "\"first_correct_product_ms\":" << metrics.first_correct_product_milliseconds << ','
              << "\"full_exact_ms\":" << metrics.full_exact_completion_milliseconds << ','
              << "\"package_ready_ms\":" << metrics.package_ready_milliseconds << ','
              << "\"source_bytes_read\":" << metrics.source_bytes_read << ','
              << "\"compressed_bytes_staged\":" << metrics.source_bytes_staged << ','
              << "\"vulkan_committed_bytes\":" << metrics.vulkan_committed_bytes << ','
              << "\"global_total_intensity\":" << derived.global_total_intensity << ','
              << "\"output_written\":" << !output.empty() << "}\n";
    return 0;
  } catch (...) {
    for (auto &pair : open_files) {
      if (pair.source >= 0)
        close(pair.source);
      if (pair.index >= 0)
        close(pair.index);
    }
    throw;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::vector<std::string> arguments(argv + 1, argv + argc);
    if (!arguments.empty() && arguments.front() == "abi-smoke") {
      return abi_smoke();
    }
    auto executor = ExactProductExecutor::create();
    if (arguments.empty() || arguments.front() == "capabilities") {
      std::cout << capabilities_json(executor->capabilities()) << '\n';
      return 0;
    }
    if (arguments.front() == "qh5-real")
      return qh5_real(arguments, *executor);
    if (arguments.front() == "qh5-pack-shard")
      return qh5_pack_shard(arguments, *executor);
    if (arguments.front() == "qh5-resident")
      return qh5_resident(arguments, *executor);
    if (arguments.front() == "packed-lz4-shard")
      return packed_lz4_shard(arguments, *executor);
    if (arguments.front() == "packed-h5-resident")
      return packed_h5_resident(arguments, *executor);
    if (arguments.front() != "synthetic") {
      throw std::invalid_argument("usage: quantem-gpu-vulkan-benchmark "
                                  "[capabilities|abi-smoke|synthetic options|"
                                  "qh5-real --directory PATH options|"
                                  "qh5-pack-shard --directory PATH "
                                  "--descriptors FILE --expected-payload FILE]");
    }

    BenchmarkOptions options;
    options.shard_scan_rows = parse_uint(arguments, "--shard-rows", 8);
    options.staging_ring_depth = parse_uint(arguments, "--ring-depth", 3);
    options.wait_after_each_shard = has_flag(arguments, "--wait-each");
    options.selected_scan_row = parse_uint(arguments, "--selected-row", 256);
    options.selected_scan_column =
        parse_uint(arguments, "--selected-column", 256);
    const std::uint32_t trials = parse_uint(arguments, "--trials", 1);
    const auto membership = detector_bands(options.source_shape);
    for (std::uint32_t trial = 1; trial <= trials; ++trial) {
      ExactProducts products;
      const auto package_started = std::chrono::steady_clock::now();
      BenchmarkMetrics metrics =
          executor->run_synthetic(options, membership, &products);
      const auto derived = quantem::gpu::vulkan::derive_products(products);
      const auto dpc_started = std::chrono::steady_clock::now();
      const auto dpc = quantem::gpu::vulkan::derive_dpc(derived);
      metrics.dpc_and_idpc_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - dpc_started)
              .count();
      metrics.package_ready_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - package_started)
              .count();
      std::cout << trial_json(trial, options, metrics,
                              derived.global_total_intensity, dpc)
                << '\n';
      if (metrics.mismatch_count != 0)
        return 2;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "{\"schema\":\"quantem.gpu.android-vulkan-error/v1\","
              << "\"status\":\"ERROR\",\"message\":\""
              << escape_json(error.what()) << "\"}\n";
    return 1;
  }
}
