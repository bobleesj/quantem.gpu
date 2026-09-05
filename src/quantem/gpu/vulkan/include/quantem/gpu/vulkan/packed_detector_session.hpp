#pragma once

#include "quantem/gpu/vulkan/packed_detector.hpp"
#include "quantem/gpu/vulkan/radial_fft.hpp"

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace quantem::gpu::vulkan {

struct PackedDetectorShardSize {
  std::uint32_t scan_count = 0;
  std::uint32_t payload_words = 0;
  std::uint32_t compressed_bytes = 0;
  std::uint32_t compressed_chunk_count = 0;
  // Zero preserves the v1 expanded-u32 descriptor count derived from scan_count.
  std::uint32_t header_words = 0;
  std::uint32_t scan_tile = 128;
  // 0 = expanded u32 {word offset,width}; 1 = compact 32-scan nibble widths.
  std::uint32_t header_encoding = 0;
};

struct PackedDetectorHeapAdmission {
  std::uint32_t index = 0;
  std::uint64_t size = 0, planned = 0, initial_budget = 0, initial_usage = 0;
  std::uint64_t minimum_headroom = 0;
};

/** Actual source-buffer allocation properties, without exposing Vulkan handles.
 * property_flags contains the Vulkan memory property bit values. Host visibility
 * does not imply host caching or compatibility with every operating-system IO path.
 */
struct PackedDetectorSourceMemory {
  std::uint32_t memory_type_index = 0, property_flags = 0, heap_index = 0;
  std::uint32_t allocation_count = 0;
  std::uint64_t allocated_bytes = 0;
};

struct PackedDetectorLz4Metrics {
  double staging_milliseconds = 0;
  double vulkan_visibility_milliseconds = 0;
  double gpu_decode_milliseconds = 0;
  double ready_milliseconds = 0;
  std::uint64_t compressed_bytes = 0;
  std::uint32_t chunk_count = 0;
};

/** Exact logical destinations borrowed only during the synchronous loader call.
 * Each memory entry describes one actual allocation, including alignment padding.
 * An empty words span excludes the private four-byte all-zero payload binding.
 * Do not retain the spans or their pointers beyond the loader call. Any internal
 * worker accessing them must be joined before that call returns or unwinds.
 */
struct PackedDetectorShardDestination {
  std::span<std::uint32_t> descriptors, words;
  PackedDetectorSourceMemory descriptor_memory, payload_memory;
  std::function<PackedDetectorLz4Metrics(
      std::span<const std::uint8_t>, std::span<const std::uint32_t>)>
      decode_lz4_to_words;
  std::function<PackedDetectorLz4Metrics(
      std::span<const std::uint8_t>, std::span<const std::uint32_t>)>
      enqueue_lz4_to_words;
};

/** Non-overlapping steady-clock phases within initialization_milliseconds.
 *
 * Shard phases are aggregate durations, not per-shard records. shard_loading
 * includes the complete loader callback (IO and authentication); the pre-allocation
 * memory guard has its own disjoint phase.
 * These are host wall times, not GPU timestamps or first-presentation evidence.
 */
struct PackedDetectorLoadTimings {
  double device_creation_milliseconds = 0;
  double plan_admission_milliseconds = 0;
  double work_allocation_milliseconds = 0;
  double shard_guard_milliseconds = 0;
  double shard_loading_milliseconds = 0;
  double shard_validation_milliseconds = 0;
  double source_allocation_mapping_milliseconds = 0;
  double source_copy_milliseconds = 0;
  double source_flush_milliseconds = 0;
  double compressed_staging_milliseconds = 0;
  double compressed_enqueue_milliseconds = 0;
  double compressed_wait_milliseconds = 0;
  double compressed_gpu_decode_milliseconds = 0;
  double descriptor_cost_scan_milliseconds = 0;
  double pipeline_descriptor_creation_milliseconds = 0;
  double command_creation_milliseconds = 0;
  double resident_budget_check_milliseconds = 0;
  double other_milliseconds = 0;
};

struct PackedDetectorSessionAdmission {
  std::string device_name;
  std::uint32_t driver_version = 0;
  std::uint32_t shard_count = 0;
  std::uint64_t source_upload_bytes = 0;
  std::uint64_t committed_bytes = 0;
  std::uint64_t reserved_staging_bytes = 0;
  double initialization_milliseconds = 0;
  std::uint64_t prepared_dpc_bytes = 0;
  double prepared_dpc_prime_milliseconds = 0;
  double prepared_dpc_gpu_milliseconds =
      std::numeric_limits<double>::quiet_NaN();
  bool prepared_dpc_ready = false;
  std::uint64_t prepared_detector_product_bytes = 0;
  bool prepared_detector_products_ready = false;
  bool timestamps_available = false;
  bool memory_budget_supported = false;
  std::vector<PackedDetectorHeapAdmission> heaps;
  std::vector<PackedDetectorSourceMemory> source_memory;
  PackedDetectorLoadTimings load_timing;
  DataRepresentation representation = DataRepresentation::lossless_packed;
};

struct PackedPreparedDpcMetrics {
  std::uint64_t output_copy_bytes = 0;
  std::uint64_t source_upload_bytes = 0, storage_read_bytes = 0;
  std::uint32_t queue_submit_count = 0, fence_wait_count = 0,
                dispatch_count = 0, device_wide_wait_count = 0;
  double wall_milliseconds = 0, output_copy_milliseconds = 0;
};

struct PackedDetectorSessionMetrics {
  RadialFftMetrics timing;
  double mask_plan_milliseconds = 0;
  bool rebase = false;
  bool source_changed = false;
  bool prepared_detector_product = false;
  std::uint32_t prepared_detector_product_index = UINT32_MAX;
  std::uint32_t changed_pixel_entries = 0;
  std::uint64_t logical_source_bytes = 0;
  std::uint64_t committed_generation = 0;
};

/** One authenticated canonical detector mask and its exact uint32 scan map. */
struct PackedPreparedDetectorProduct {
  CircularDetector detector;
  std::span<const std::uint8_t> mask;
  std::span<const std::uint32_t> values;
};

/** One selected DP from this session's authenticated resident source.
 * GPU timing is NaN when timestamps are unavailable. Wall time includes mutex
 * waiting and the measured small GPU-to-caller copy; it is not presentation time.
 * Source identity belongs to the caller that authenticated this session's plan.
 */
struct PackedSelectedDiffractionMetrics {
  std::uint64_t generation = 0;
  std::uint64_t output_copy_bytes = 0, source_upload_bytes = 0, storage_read_bytes = 0;
  std::uint32_t row = 0, column = 0, detector_rows = 0, detector_columns = 0;
  std::uint32_t shard_index = 0, shard_local_scan = 0;
  std::uint32_t queue_submit_count = 0, fence_wait_count = 0, dispatch_count = 0;
  std::uint32_t device_wide_wait_count = 0;
  double wall_milliseconds = 0, mutex_wait_milliseconds = 0, output_copy_milliseconds = 0;
  double gpu_decode_milliseconds = std::numeric_limits<double>::quiet_NaN();
};

/** Local, resident, exact uint16 detector sums and optional existing 512² FFT.
 *
 * Constructor loads ONE bounded shard at a time. Before each shard allocation,
 * the mandatory guard receives its index and previously admitted logical source
 * bytes. The loader fills the exact borrowed final-buffer spans and authenticates
 * all bytes against the source-bound manifest before returning. Neither callback
 * may retain borrowed storage or leave asynchronous work running. Bounded internal
 * IO/authentication overlap is allowed only when every worker is joined before
 * the callback returns or unwinds, with no concurrent accesses to the same bytes
 * and no GPU/session access from those workers. Any exception aborts
 * construction and releases all allocations without publishing a partial session.
 * Loader scratch (including IO/decode buffers) must fit reserved_staging_bytes;
 * the unchanged conservative reservation still covers one entire shard. The complete
 * source plan is checked before source allocations; construction returns only
 * when every shard has been admitted. No partial source counts as resident.
 *
 * Detector requests serialize and keep a private committed mask/image base. Every
 * successful detector compute advances that base, regardless of whether UI displays
 * the result. Selected diffraction shares that serialization but never changes the
 * detector base or FFT cache. Device/fence failure makes the session unusable, rather than guessing
 * which base survived. Output copies occur only after the combined GPU detector
 * and FFT submission; FFT never reads a CPU intermediate image.
 *
 * The runtime accepts square 512 or 1024 scans without an implicit crop/resize.
 * FFT remains the qualified 512-square path; a 1024-square request must omit it.
 * Histogram/presentation integration and physical FPS acceptance are separate.
 */
class PackedDetectorSession final {
public:
  using ShardLoadGuard = std::function<void(std::size_t, std::uint64_t)>;
  using AuthenticatedShardLoader =
      std::function<void(std::size_t, PackedDetectorShardDestination)>;

  PackedDetectorSession(Shape4D shape,
      std::span<const PackedDetectorShardSize> source_plan,
      ShardLoadGuard before_shard,
      AuthenticatedShardLoader load_shard,
      std::span<const std::uint8_t> excluded_detector_pixels,
      std::uint64_t reserved_process_bytes,
      std::uint64_t reserved_staging_bytes);
  PackedDetectorSession(Shape4D shape,
      std::span<const PackedDetectorShardSize> source_plan,
      ShardLoadGuard before_shard,
      AuthenticatedShardLoader load_shard,
      std::span<const std::uint8_t> excluded_detector_pixels,
      std::span<const std::uint32_t> authenticated_prepared_dpc_words,
      std::uint64_t reserved_process_bytes,
      std::uint64_t reserved_staging_bytes);
  PackedDetectorSession(Shape4D shape,
      std::span<const PackedDetectorShardSize> source_plan,
      ShardLoadGuard before_shard,
      AuthenticatedShardLoader load_shard,
      std::span<const std::uint8_t> excluded_detector_pixels,
      std::span<const std::uint32_t> authenticated_prepared_dpc_words,
      std::span<const PackedPreparedDetectorProduct>
          authenticated_prepared_detector_products,
      std::uint64_t reserved_process_bytes,
      std::uint64_t reserved_staging_bytes);
  ~PackedDetectorSession();
  PackedDetectorSession(const PackedDetectorSession &) = delete;
  PackedDetectorSession &operator=(const PackedDetectorSession &) = delete;

  [[nodiscard]] const PackedDetectorSessionAdmission &admission() const noexcept;
  [[nodiscard]] PackedDetectorSessionMetrics request(
      CircularDetector detector, std::uint64_t generation,
      std::span<std::uint32_t> image, std::span<float> fft_magnitude = {});

  /** Decode one complete detector frame at the requested (row, column) on GPU.
   * diffraction must contain exactly detector_rows*detector_columns uint32 values.
   * Values retain the admitted uint16 counts exactly; nonzero admitted exclusions
   * return zero. The source itself is never altered. There are no source reads,
   * source uploads, CPU unpacking, or full-product dispatches per call.
   *
   * The caller owns the output, which is borrowed only until return/unwind and
   * copied after successful GPU completion. An exception makes the result invalid.
   * No output alias survives, and later calls cannot change a returned frame.
   * generation is echoed without ordering/cancellation policy: repeated or older
   * generations are valid and do not alter the committed detector generation,
   * image, mask, or FFT-current state. The caller decides which result to display.
   * Calls serialize with request(); validation errors leave the session usable,
   * while GPU/fence/readback errors retain the failed-session reopen requirement.
   */
  [[nodiscard]] PackedSelectedDiffractionMetrics selected_diffraction(
      std::uint32_t row, std::uint32_t column, std::uint64_t generation,
      std::span<std::uint32_t> diffraction);

  /** Copy already-primed full-nonexcluded DPC maps; no source read or dispatch. */
  [[nodiscard]] PackedPreparedDpcMetrics prepared_dpc(
      std::span<float> row, std::span<float> column);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace quantem::gpu::vulkan
