#pragma once

#include "quantem/gpu/vulkan/contract.hpp"
#include "quantem/gpu/vulkan/qh5_indexed_source.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace quantem::gpu::vulkan {

struct VulkanCapabilities {
  std::string device_name;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint64_t max_storage_buffer_range_bytes = 0;
  std::uint64_t max_memory_allocation_bytes = 0;
  std::uint32_t subgroup_size = 0;
  std::uint32_t timestamp_valid_bits = 0;
  float timestamp_period_nanoseconds = 0.0F;
  bool shader_int16 = false;
  bool shader_int64 = false;
  bool storage_buffer_8bit_access = false;
  bool host_visible_device_local_memory = false;
};

struct BenchmarkOptions {
  Shape4D source_shape{512, 512, 192, 192};
  std::uint32_t selected_scan_row = 256;
  std::uint32_t selected_scan_column = 256;
  std::uint32_t shard_scan_rows = 8;
  std::uint32_t staging_ring_depth = 3;
  bool wait_after_each_shard = false;
  const std::atomic<bool> *cancellation_flag = nullptr;
  void (*progress_callback)(void *context, std::uint32_t completed_scans,
                            std::uint32_t total_scans,
                            bool first_product_ready) = nullptr;
  void *progress_context = nullptr;
};

class OperationCancelled final : public std::exception {
public:
  [[nodiscard]] const char *what() const noexcept override {
    return "quantem.gpu Vulkan operation cancelled";
  }
};

class SourceIoError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct BenchmarkMetrics {
  double setup_milliseconds = 0.0;
  double source_staging_milliseconds = 0.0;
  double storage_read_milliseconds = 0.0;
  double source_decode_milliseconds = 0.0;
  double vulkan_visibility_milliseconds = 0.0;
  double first_correct_product_milliseconds = 0.0;
  double full_exact_completion_milliseconds = 0.0;
  double dpc_and_idpc_milliseconds = 0.0;
  double package_ready_milliseconds = 0.0;
  double gpu_source_lz4_milliseconds = 0.0;
  double gpu_source_bitunshuffle_milliseconds = 0.0;
  double gpu_scan_products_milliseconds = 0.0;
  double gpu_mean_diffraction_milliseconds = 0.0;
  std::uint64_t source_bytes_staged = 0;
  std::uint64_t source_bytes_read = 0;
  std::uint64_t explicit_copy_bytes = 0;
  std::uint64_t shader_source_bytes_read = 0;
  std::uint64_t vulkan_committed_bytes = 0;
  std::uint32_t queue_submit_count = 0;
  std::uint32_t fence_wait_count = 0;
  std::uint32_t device_wide_wait_count = 0;
  double user_cpu_milliseconds = 0.0;
  double system_cpu_milliseconds = 0.0;
  std::uint64_t maximum_resident_set_kibibytes = 0;
  std::uint64_t minor_page_fault_count = 0;
  std::uint64_t major_page_fault_count = 0;
  std::uint64_t mismatch_count = 0;
  bool parity_checked = false;
};

struct Qh5PackedShardMetrics {
  double storage_read_milliseconds = 0.0;
  double source_staging_milliseconds = 0.0;
  double vulkan_visibility_milliseconds = 0.0;
  double gpu_decode_milliseconds = 0.0;
  double gpu_pack_milliseconds = 0.0;
  double gpu_decode_and_pack_milliseconds = 0.0;
  double ready_milliseconds = 0.0;
  std::uint64_t source_bytes_read = 0;
  std::uint64_t compressed_bytes_staged = 0;
  std::uint64_t vulkan_committed_bytes = 0;
};

struct Qh5SelectedFrameMetrics {
  double total_milliseconds = 0.0;
  double storage_read_milliseconds = 0.0;
  double source_staging_milliseconds = 0.0;
  double vulkan_visibility_milliseconds = 0.0;
  double gpu_decode_milliseconds = 0.0;
  std::uint64_t source_bytes_read = 0;
  std::uint64_t source_frame_count = 0;
  std::uint64_t source_block_count = 0;
  std::uint64_t vulkan_committed_bytes = 0;
  std::uint32_t queue_submit_count = 0;
};

struct PackedLz4Metrics {
  double vulkan_visibility_milliseconds = 0.0;
  double gpu_decode_milliseconds = 0.0;
  double ready_milliseconds = 0.0;
  std::uint64_t compressed_bytes_staged = 0;
  std::uint64_t decoded_bytes = 0;
  std::uint64_t vulkan_committed_bytes = 0;
};

struct PreparedSourceSegment {
  int borrowed_file_descriptor = -1;
  std::uint64_t file_offset_bytes = 0;
  std::uint64_t length_bytes = 0;
};

class ExactProductExecutor {
public:
  static std::unique_ptr<ExactProductExecutor> create();

  virtual ~ExactProductExecutor() = default;
  [[nodiscard]] virtual const VulkanCapabilities &capabilities() const = 0;
  [[nodiscard]] virtual BenchmarkMetrics
  run_synthetic(const BenchmarkOptions &options,
                const std::vector<std::uint8_t> &detector_band_membership,
                ExactProducts *products) = 0;
  [[nodiscard]] virtual BenchmarkMetrics run_prepared_contiguous_u8(
      const BenchmarkOptions &options,
      const std::vector<PreparedSourceSegment> &ordered_segments,
      const std::vector<std::uint8_t> &detector_band_membership,
      ExactProducts *products) = 0;
  [[nodiscard]] virtual BenchmarkMetrics run_prepared_contiguous_u16(
      const BenchmarkOptions &options,
      const std::vector<PreparedSourceSegment> &ordered_segments,
      const std::vector<std::uint8_t> &detector_band_membership,
      ExactProducts *products) = 0;
  [[nodiscard]] virtual BenchmarkMetrics
  run_indexed_qh5_u16(const BenchmarkOptions &options,
                      const Qh5IndexedSource &source,
                      const std::vector<std::uint8_t> &detector_band_membership,
                      ExactProducts *products) = 0;
  [[nodiscard]] virtual BenchmarkMetrics run_indexed_qh5_audited_low8(
      const BenchmarkOptions &options, const Qh5IndexedSource &source,
      const std::vector<std::uint8_t> &detector_band_membership,
      const std::vector<std::uint8_t> &excluded_detector_pixels,
      ExactProducts *products) = 0;
  [[nodiscard]] virtual Qh5PackedShardMetrics
  pack_indexed_qh5_audited_low8_shard(
      const Qh5IndexedSource &source, std::uint64_t first_frame,
      std::uint32_t frame_count,
      const std::vector<std::uint32_t> &packed_descriptors,
      std::uint32_t packed_payload_word_count,
      const std::vector<std::uint8_t> &excluded_detector_pixels,
      std::vector<std::uint32_t> *packed_payload) = 0;
  [[nodiscard]] virtual Qh5SelectedFrameMetrics
  read_indexed_qh5_frame_u16(const Qh5IndexedSource &source,
                             std::uint64_t frame,
                             std::uint16_t *destination,
                             std::size_t destination_value_capacity) = 0;
  [[nodiscard]] virtual PackedLz4Metrics decode_packed_lz4(
      const std::vector<std::uint8_t> &compressed,
      const std::vector<std::uint32_t> &chunk_metadata,
      std::uint32_t decoded_word_count,
      std::vector<std::uint32_t> *decoded) = 0;
};

} // namespace quantem::gpu::vulkan
