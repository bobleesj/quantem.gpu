#include "quantem/gpu/vulkan/packed_detector_session.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace quantem::gpu::vulkan {
namespace {
[[maybe_unused]] const std::uint32_t kDetector[] =
#include "packed_detector_update_spirv.inc"
    ;
const std::uint32_t kAtomicDetector[] =
#include "packed_detector_update_atomic_spirv.inc"
    ;
const std::uint32_t kTileDetector[] =
#include "packed_detector_update_tile_spirv.inc"
    ;
const std::uint32_t kScalarDetector[] =
#include "packed_detector_update_scalar_spirv.inc"
    ;
const std::uint32_t kSelectedDiffraction[] =
#include "packed_selected_diffraction_spirv.inc"
    ;
const std::uint32_t kPreparedDpcCom[] =
#include "prepared_dpc_com_spirv.inc"
    ;
const std::uint32_t kPreparedDpcMean[] =
#include "prepared_dpc_mean_spirv.inc"
    ;
const std::uint32_t kPreparedDpcCenter[] =
#include "prepared_dpc_center_spirv.inc"
    ;
const std::uint32_t kPackedLz4[] =
#include "packed_lz4_decode_spirv.inc"
    ;
const std::uint32_t kWorkgroupPackedLz4[] =
#include "packed_lz4_decode_workgroup_spirv.inc"
    ;
[[maybe_unused]] const std::uint32_t kLeaderPackedLz4[] =
#include "packed_lz4_decode_leader_spirv.inc"
    ;
const std::uint32_t kScalarPackedLz4[] =
#include "packed_lz4_decode_scalar_spirv.inc"
    ;
const std::uint32_t kCompactHeaderValidate[] =
#include "packed_compact_header_validate_spirv.inc"
    ;
// Reuse the already-qualified FFT shaders verbatim, with their existing bindings.
const std::uint32_t kRows[] =
#include "radial_fft_rows_spirv.inc"
    ;
const std::uint32_t kColumns[] =
#include "radial_fft_columns_spirv.inc"
    ;
const std::uint32_t kMagnitude[] =
#include "radial_magnitude_spirv.inc"
    ;
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t lz4_ring_depth = 8U;
constexpr std::uint32_t compact_header_encoding = 1U;
constexpr std::uint32_t compact_checkpoint_tiles = 32U;

template <class T> T structure(VkStructureType type) {
  T value{};
  value.sType = type;
  return value;
}
void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string("Packed detector ") + operation +
                             " failed: Vulkan " + std::to_string(result));
}
double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
  if (b > UINT64_MAX - a) throw std::invalid_argument("Packed source byte count overflow");
  return a + b;
}
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize bytes = 0;
  VkDeviceSize allocated_bytes = 0;
  std::uint32_t memory_type_index = 0;
  void *mapped = nullptr;
  bool coherent = false;
};
struct Lz4Slot {
  Buffer compressed, metadata, status;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  Buffer *output = nullptr;
  std::uint32_t chunk_count = 0;
  Clock::time_point enqueued{};
  PackedDetectorLz4Metrics metrics;
  bool active = false;
};
struct ResidentShard {
  Buffer payload, headers;
  std::uint32_t scans = 0, first_scan = 0, tiles = 0, scan_tile = 128;
  std::uint32_t header_encoding = 0, header_words_per_pixel = 0;
  std::uint32_t payload_words = 0;
};
struct Parameters {
  std::uint32_t scan_count, tile_count, entry_count, output_offset, mode;
  std::uint32_t scan_tile, header_encoding, header_words_per_pixel;
};
static_assert(sizeof(Parameters) == 32);
struct SelectedDiffractionParameters {
  std::uint32_t scan, tile_count, pixel_count, scan_tile;
  std::uint32_t header_encoding, header_words_per_pixel;
};
static_assert(sizeof(SelectedDiffractionParameters) == 24);
struct PackedLz4Parameters {
  std::uint32_t chunk_count, compressed_byte_count, dispatch_width;
};
static_assert(sizeof(PackedLz4Parameters) == 12);
struct CompactHeaderValidationParameters {
  std::uint32_t pixel_count, tile_count, header_words_per_pixel, payload_words;
};
static_assert(sizeof(CompactHeaderValidationParameters) == 16);
} // namespace

struct PackedDetectorSession::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t family = 0, timestamp_bits = 0;
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  std::array<std::uint64_t, VK_MAX_MEMORY_HEAPS> planned_heap_bytes{}, committed_heap_bytes{};
  VkDeviceSize maximum_allocation = 0;
  VkDescriptorSetLayout detector_layout = VK_NULL_HANDLE, fft_layout = VK_NULL_HANDLE;
  VkPipelineLayout detector_pipeline_layout = VK_NULL_HANDLE, fft_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout diffraction_layout = VK_NULL_HANDLE;
  VkPipelineLayout diffraction_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout prepared_dpc_layout = VK_NULL_HANDLE;
  VkPipelineLayout prepared_dpc_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  std::array<std::vector<VkDescriptorSet>, 2> detector_sets;
  std::array<VkDescriptorSet, 2> fft_sets{};
  std::vector<VkDescriptorSet> diffraction_sets;
  std::array<VkPipeline, 6> pipelines{};
  std::array<VkDescriptorSet, 3> prepared_dpc_sets{};
  std::array<VkPipeline, 3> prepared_dpc_pipelines{};
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  std::array<std::array<VkCommandBuffer, 3>, 2> detector_commands{};
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  VkDescriptorSetLayout lz4_layout = VK_NULL_HANDLE;
  VkPipelineLayout lz4_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorPool lz4_descriptor_pool = VK_NULL_HANDLE;
  VkPipeline lz4_pipeline = VK_NULL_HANDLE;
  VkPipeline lz4_scalar_pipeline = VK_NULL_HANDLE;
  VkCommandPool lz4_command_pool = VK_NULL_HANDLE;
  std::vector<Lz4Slot> lz4_slots;
  std::size_t lz4_next_slot = 0;
  std::vector<ResidentShard> shards;
  std::array<Buffer, 2> images;
  Buffer entries, request_parameters, rows, columns, magnitude, twiddles;
  Buffer diffraction, diffraction_exclusions;
  Buffer prepared_dpc_moments, prepared_dpc_row, prepared_dpc_column,
      prepared_dpc_mean;
  std::array<Buffer, 3> prepared_detector_products;
  std::array<std::vector<std::uint8_t>, 3> prepared_detector_masks;
  std::array<CircularDetector, 3> prepared_detector_geometries{};
  Shape4D shape;
  VkDeviceSize plane_bytes = 0;
  std::vector<std::uint8_t> excluded, committed_mask;
  std::vector<std::uint64_t> column_bytes;
  std::uint32_t committed_image = 0;
  std::uint64_t committed_generation = 0, process_budget = 0;
  bool fft_current = false, failed = false, atomic_parallel_detector = false;
  bool has_prepared_dpc = false;
  bool has_prepared_detector_products = false;
  std::uint64_t prepared_dpc_input_bytes = 0;
  PackedDetectorSessionAdmission admission;
  std::mutex mutex;

  static std::uint32_t header_width(
      const ResidentShard &shard, std::span<const std::uint32_t> headers,
      std::uint32_t pixel, std::uint32_t tile) {
    if (shard.header_encoding == 0U)
      return headers[std::size_t{pixel} * shard.tiles + tile] & 31U;
    const auto checkpoint_words =
        (shard.tiles + compact_checkpoint_tiles - 1U) /
        compact_checkpoint_tiles;
    const auto width_word = tile / 8U;
    const auto shift = (tile % 8U) * 4U;
    const auto base = std::size_t{pixel} * shard.header_words_per_pixel;
    return (headers[base + checkpoint_words + width_word] >> shift) & 15U;
  }

  Impl(Shape4D source_shape, std::span<const PackedDetectorShardSize> plan,
       const ShardLoadGuard &before_shard, const AuthenticatedShardLoader &loader,
       std::span<const std::uint8_t> exclusions,
       std::span<const std::uint32_t> prepared_dpc_words,
       std::span<const PackedPreparedDetectorProduct> prepared_products,
       std::uint64_t budget, std::uint64_t staging)
      : shape(source_shape),
        plane_bytes(std::uint64_t{source_shape.scan_rows} *
                    source_shape.scan_columns * 4U),
        excluded(exclusions.begin(), exclusions.end()), process_budget(budget) {
    const auto start = Clock::now();
    if ((shape.scan_rows != 512 && shape.scan_rows != 1024) ||
        shape.scan_columns != shape.scan_rows ||
        !shape.detector_rows || !shape.detector_columns ||
        std::uint64_t{shape.detector_rows} * shape.detector_columns > 65537 ||
        plan.empty() || plan.size() > 2048 || !before_shard || !loader || !budget)
      throw std::invalid_argument("Use a complete 512x512 or 1024x1024 scan, bounded detector plan and memory budget");
    const auto pixels = shape.detector_rows * shape.detector_columns;
    if (!excluded.empty() && excluded.size() != pixels)
      throw std::invalid_argument("Authoritative detector mask size differs from source shape");
    has_prepared_dpc = !prepared_dpc_words.empty();
    if (has_prepared_dpc) {
      (void)validate_prepared_dpc_moments(shape, excluded, prepared_dpc_words);
      prepared_dpc_input_bytes = prepared_dpc_words.size_bytes();
      if (prepared_dpc_input_bytes > staging)
        throw std::invalid_argument(
            "Prepared DPC range exceeds the declared loader staging budget");
    }
    has_prepared_detector_products = !prepared_products.empty();
    if (has_prepared_detector_products) {
      if (prepared_products.size() != prepared_detector_products.size())
        throw std::invalid_argument(
            "Prepared detector products require exactly BF, ABF, and ADF");
      const auto scan_pixels =
          std::size_t{shape.scan_rows} * shape.scan_columns;
      for (std::size_t product = 0; product < prepared_products.size(); ++product) {
        const auto &input = prepared_products[product];
        prepared_detector_geometries[product] = input.detector;
        if (input.mask.size() != pixels || input.values.size() != scan_pixels)
          throw std::invalid_argument(
              "Prepared detector product shape differs from the admitted source");
        std::uint64_t selected = 0;
        prepared_detector_masks[product].assign(input.mask.begin(), input.mask.end());
        for (std::size_t pixel = 0; pixel < input.mask.size(); ++pixel) {
          if (input.mask[pixel] > 1U ||
              (!excluded.empty() && excluded[pixel] && input.mask[pixel]))
            throw std::invalid_argument(
                "Prepared detector product mask is not binary or includes an excluded pixel");
          selected += input.mask[pixel];
        }
        const auto bound = selected * 255U;
        if (std::any_of(input.values.begin(), input.values.end(),
                        [bound](std::uint32_t value) { return value > bound; }))
          throw std::invalid_argument(
              "Prepared detector product violates exact uint8 source bounds");
      }
      const auto prepared_bytes = 3U * plane_bytes;
      if (prepared_bytes > staging)
        throw std::invalid_argument(
            "Prepared detector products exceed the declared loader staging budget");
    }
    admission.reserved_staging_bytes = staging;
    column_bytes.resize(pixels);
    auto &timing = admission.load_timing;
    try {
      auto phase_start = Clock::now();
      create_device();
      timing.device_creation_milliseconds = milliseconds(Clock::now() - phase_start);
      phase_start = Clock::now();
      admit_plan(plan);
      timing.plan_admission_milliseconds = milliseconds(Clock::now() - phase_start);
      phase_start = Clock::now();
      create_work_buffers();
      timing.work_allocation_milliseconds = milliseconds(Clock::now() - phase_start);
      create_lz4_loader(plan);
      bool has_compact_headers = false;
      // Size all slots first so failure cleanup also owns partially loaded shards.
      shards.resize(plan.size());
      std::uint32_t first_scan = 0;
      for (std::size_t index = 0; index < plan.size(); ++index) {
        phase_start = Clock::now();
        // Must precede source allocation, not merely the IO into that allocation.
        before_shard(index, admission.source_upload_bytes);
        timing.shard_guard_milliseconds += milliseconds(Clock::now() - phase_start);
        auto &resident = shards[index];
        resident.scans = plan[index].scan_count;
        resident.first_scan = first_scan;
        resident.scan_tile = plan[index].scan_tile;
        resident.header_encoding = plan[index].header_encoding;
        resident.tiles = resident.scans / resident.scan_tile +
                         (resident.scans % resident.scan_tile != 0);
        const auto descriptor_count = plan[index].header_words != 0U
            ? std::size_t{plan[index].header_words}
            : std::size_t{pixels} * resident.tiles;
        resident.header_words_per_pixel = static_cast<std::uint32_t>(
            descriptor_count / pixels);
        const auto payload_count = plan[index].payload_words;
        resident.payload_words = static_cast<std::uint32_t>(payload_count);
        phase_start = Clock::now();
        resident.payload = allocate(std::max(4ULL, payload_count * 4ULL), true);
        resident.headers = allocate(descriptor_count * 4ULL, true);
        // An all-zero shard has no logical payload bytes, but Vulkan still needs
        // a valid binding. Never include this private word in authentication.
        if (!payload_count) *static_cast<std::uint32_t *>(resident.payload.mapped) = 0;
        PackedDetectorShardDestination destination{
            {static_cast<std::uint32_t *>(resident.headers.mapped), descriptor_count},
            {static_cast<std::uint32_t *>(resident.payload.mapped), payload_count},
            source_memory(resident.headers), source_memory(resident.payload),
            {}, {}};
        if (plan[index].compressed_bytes != 0U) {
          destination.decode_lz4_to_words =
              [this, &resident](std::span<const std::uint8_t> compressed,
                                std::span<const std::uint32_t> metadata) {
                return decode_lz4(resident.payload, compressed, metadata);
              };
          destination.enqueue_lz4_to_words =
              [this, &resident](std::span<const std::uint8_t> compressed,
                                std::span<const std::uint32_t> metadata) {
                return enqueue_lz4(resident.payload, compressed, metadata);
              };
        }
        record_source_memory(destination.descriptor_memory);
        record_source_memory(destination.payload_memory);
        timing.source_allocation_mapping_milliseconds += milliseconds(Clock::now() - phase_start);
        phase_start = Clock::now();
        loader(index, destination); // Authenticate once here, never during a drag.
        timing.shard_loading_milliseconds += milliseconds(Clock::now() - phase_start);
        phase_start = Clock::now();
        if (resident.header_encoding == 0U) {
          validate_packed_detector_shard(resident.scans, pixels,
                                         destination.descriptors,
                                         destination.words, resident.scan_tile);
        } else {
          has_compact_headers = true;
        }
        timing.shard_validation_milliseconds += milliseconds(Clock::now() - phase_start);
        const auto bytes = add(destination.words.size() * 4ULL, destination.descriptors.size() * 4ULL);
        if (bytes > staging)
          throw std::invalid_argument("One loaded shard exceeds reserved staging; increase honest peak budget");
        phase_start = Clock::now();
        if (plan[index].compressed_bytes == 0U)
          flush(resident.payload);
        flush(resident.headers);
        timing.source_flush_milliseconds += milliseconds(Clock::now() - phase_start);
        admission.source_upload_bytes = add(admission.source_upload_bytes, bytes);
        phase_start = Clock::now();
        if (resident.header_encoding == 0U) {
          for (std::uint32_t pixel = 0; pixel < pixels; ++pixel) {
            column_bytes[pixel] +=
                std::uint64_t{resident.header_words_per_pixel} * 4U;
            for (std::uint32_t tile = 0; tile < resident.tiles; ++tile) {
              const auto width = header_width(resident, destination.descriptors,
                                              pixel, tile);
              column_bytes[pixel] +=
                  4U * ((resident.scan_tile * width + 31U) / 32U);
            }
          }
        }
        timing.descriptor_cost_scan_milliseconds += milliseconds(Clock::now() - phase_start);
        first_scan += resident.scans;
      }
      finish_lz4_loader();
      if (has_prepared_dpc) {
        phase_start = Clock::now();
        before_shard(plan.size(), admission.source_upload_bytes);
        timing.shard_guard_milliseconds +=
            milliseconds(Clock::now() - phase_start);
        phase_start = Clock::now();
        create_prepared_dpc_buffers(prepared_dpc_words);
        timing.shard_loading_milliseconds +=
            milliseconds(Clock::now() - phase_start);
      }
      if (has_prepared_detector_products) {
        phase_start = Clock::now();
        before_shard(plan.size() + (has_prepared_dpc ? 1U : 0U),
                     admission.source_upload_bytes);
        timing.shard_guard_milliseconds +=
            milliseconds(Clock::now() - phase_start);
        phase_start = Clock::now();
        create_prepared_detector_product_buffers(prepared_products);
        timing.shard_loading_milliseconds +=
            milliseconds(Clock::now() - phase_start);
      }
      if (has_compact_headers) {
        phase_start = Clock::now();
        validate_compact_headers_gpu();
        timing.shard_validation_milliseconds +=
            milliseconds(Clock::now() - phase_start);
      }
      phase_start = Clock::now();
      create_pipelines_and_descriptors();
      timing.pipeline_descriptor_creation_milliseconds = milliseconds(Clock::now() - phase_start);
      phase_start = Clock::now();
      create_commands();
      timing.command_creation_milliseconds = milliseconds(Clock::now() - phase_start);
      if (has_prepared_dpc)
        prime_prepared_dpc();
      phase_start = Clock::now();
      record_resident_heap_budget();
      timing.resident_budget_check_milliseconds = milliseconds(Clock::now() - phase_start);
      admission.shard_count = static_cast<std::uint32_t>(shards.size());
      admission.initialization_milliseconds = milliseconds(Clock::now() - start);
      // Retain loop bookkeeping and timer overhead.
      timing.other_milliseconds = admission.initialization_milliseconds -
          (timing.device_creation_milliseconds + timing.plan_admission_milliseconds +
           timing.work_allocation_milliseconds + timing.shard_guard_milliseconds +
           timing.shard_loading_milliseconds +
           timing.shard_validation_milliseconds + timing.source_allocation_mapping_milliseconds +
           timing.source_copy_milliseconds + timing.source_flush_milliseconds +
           timing.descriptor_cost_scan_milliseconds + timing.pipeline_descriptor_creation_milliseconds +
           timing.command_creation_milliseconds + admission.prepared_dpc_prime_milliseconds +
           timing.resident_budget_check_milliseconds);
    } catch (...) { destroy(); throw; }
  }
  ~Impl() { destroy(); }

  void create_device() {
    auto app = structure<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "QuantEM exact movable detector";
    app.apiVersion = VK_API_VERSION_1_1;
    auto info = structure<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    info.pApplicationInfo = &app;
    check(vkCreateInstance(&info, nullptr, &instance), "create instance");
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "count devices");
    std::vector<VkPhysicalDevice> candidates(count);
    check(vkEnumeratePhysicalDevices(instance, &count, candidates.data()), "enumerate devices");
    for (const auto candidate : candidates) {
      std::uint32_t families_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families_count, nullptr);
      std::vector<VkQueueFamilyProperties> families(families_count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families_count, families.data());
      for (std::uint32_t index = 0; index < families_count; ++index) {
        if (families[index].queueCount && (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          physical = candidate; family = index; timestamp_bits = families[index].timestampValidBits;
          break;
        }
      }
      if (physical) break;
    }
    if (!physical) throw std::runtime_error("No Vulkan compute device is available");
    auto maintenance = structure<VkPhysicalDeviceMaintenance3Properties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
    auto props = structure<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    props.pNext = &maintenance;
    vkGetPhysicalDeviceProperties2(physical, &props);
    properties = props.properties;
    maximum_allocation = maintenance.maxMemoryAllocationSize;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    std::uint32_t extension_count = 0;
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr), "count extensions");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, extensions.data()), "read extensions");
    admission.memory_budget_supported = std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
      return std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    });
    const auto &limits = properties.limits;
    if (limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 256 ||
        limits.maxComputeSharedMemorySize < 4096 || limits.maxComputeWorkGroupCount[0] < 2048 ||
        limits.maxStorageBufferRange < 2 * plane_bytes || limits.maxPerStageDescriptorStorageBuffers < 7 ||
        limits.maxDescriptorSetStorageBuffers < 7 || limits.maxPushConstantsSize < sizeof(Parameters))
      throw std::runtime_error("Vulkan limits do not support the exact detector interaction contract");
    const float priority = 1;
    auto queue_info = structure<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue_info.queueFamilyIndex = family; queue_info.queueCount = 1; queue_info.pQueuePriorities = &priority;
    auto device_info = structure<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical, &device_info, nullptr, &device), "create device");
    vkGetDeviceQueue(device, family, 0, &queue);
    admission.device_name = properties.deviceName;
    admission.driver_version = properties.driverVersion;
    admission.timestamps_available = timestamp_bits != 0;
    atomic_parallel_detector =
        properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
        std::strstr(properties.deviceName, "Apple") != nullptr;
  }

  void admit_plan(std::span<const PackedDetectorShardSize> plan) {
    const auto pixels = shape.detector_rows * shape.detector_columns;
    std::uint64_t total_scans = 0;
    std::uint64_t maximum_compressed_bytes = 0;
    std::uint64_t maximum_chunk_count = 0;
    // Two image planes, two complex FFT planes, magnitude, twiddles and entries;
    // plus a dedicated uint32 selected DP and immutable uint32 exclusion mask.
    std::uint64_t required = 7 * plane_bytes + 2048 + std::uint64_t{pixels} * 16 + 8;
    if (has_prepared_dpc)
      required = add(required, add(prepared_dpc_input_bytes, add(2 * plane_bytes, 8U)));
    if (has_prepared_detector_products)
      required = add(required, 3U * plane_bytes);
    const bool has_compressed_source = std::any_of(
        plan.begin(), plan.end(), [](const auto &item) {
          return item.compressed_bytes != 0U ||
                 item.compressed_chunk_count != 0U;
        });
    if (plan.size() * 2 + 10 + (has_prepared_dpc ? 4U : 0U) +
            (has_prepared_detector_products ? 3U : 0U) +
            (has_compressed_source ? lz4_ring_depth * 3U : 0U) >
        properties.limits.maxMemoryAllocationCount)
      throw std::invalid_argument("Too many packed allocations for this device; use an admitted shard plan");
    for (const auto item : plan) {
      if (!item.scan_count || item.scan_count > 262144 ||
          item.payload_words > (1U << 27) ||
          (item.scan_tile != 128U && item.scan_tile != 32U) ||
          item.scan_count % item.scan_tile != 0U ||
          item.header_encoding > compact_header_encoding ||
          (item.header_encoding == compact_header_encoding &&
           item.scan_tile != 32U))
        throw std::invalid_argument("Packed shard size or payload word offset range is invalid");
      if ((item.scan_count + 31U) / 32U > properties.limits.maxComputeWorkGroupCount[0])
        throw std::invalid_argument("Packed rebase dispatch exceeds actual Vulkan workgroup limits; split before admission");
      if ((item.compressed_bytes == 0U) !=
              (item.compressed_chunk_count == 0U) ||
          (has_compressed_source && item.compressed_bytes == 0U))
        throw std::invalid_argument(
            "Compressed packed-source plans require bytes and chunks for every shard");
      maximum_compressed_bytes =
          std::max<std::uint64_t>(maximum_compressed_bytes,
                                  item.compressed_bytes);
      maximum_chunk_count =
          std::max<std::uint64_t>(maximum_chunk_count,
                                  item.compressed_chunk_count);
      total_scans = add(total_scans, item.scan_count);
      const auto tiles = item.scan_count / item.scan_tile;
      const auto payload = std::max(std::uint64_t{4}, std::uint64_t{item.payload_words} * 4);
      const auto expected_header_words = item.header_encoding == 0U
          ? std::uint64_t{pixels} * tiles
          : std::uint64_t{pixels} *
                ((tiles + compact_checkpoint_tiles - 1U) /
                     compact_checkpoint_tiles +
                 (tiles + 7U) / 8U);
      const auto header_words = item.header_words != 0U
          ? std::uint64_t{item.header_words}
          : std::uint64_t{pixels} * tiles;
      if (header_words != expected_header_words)
        throw std::invalid_argument(
            "Packed shard header count does not match its encoding");
      const auto headers = header_words * 4U;
      for (const auto bytes : {payload, headers}) {
        if (bytes > maximum_allocation || bytes > properties.limits.maxStorageBufferRange)
          throw std::invalid_argument("Packed shard exceeds actual Vulkan buffer limits; split before admission");
        required = add(required, bytes);
      }
      if (add(payload, headers) > admission.reserved_staging_bytes)
        throw std::invalid_argument("Shard plan exceeds declared loader staging budget");
    }
    const auto expected_scans =
        std::uint64_t{shape.scan_rows} * shape.scan_columns;
    if (total_scans != expected_scans ||
        add(required, admission.reserved_staging_bytes) > process_budget)
      throw std::invalid_argument("Full source coverage or reserved peak memory admission failed");

    // Query actual aligned requirements and selected heaps for the WHOLE plan
    // before loading a shard or allocating device memory. A process RAM budget
    // alone cannot admit a source larger than the device's accessible heap.
    std::uint64_t actual_required = 0;
    auto account = [&](VkDeviceSize bytes, bool host) {
      Buffer probe;
      try {
        probe = create_buffer(bytes);
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, probe.buffer, &requirements);
        if (requirements.size > maximum_allocation)
          throw std::invalid_argument("Packed buffer allocation exceeds the device allocation limit");
        const auto type = select_memory_type(requirements.memoryTypeBits, host);
        const auto heap = memory_properties.memoryTypes[type].heapIndex;
        planned_heap_bytes[heap] = add(planned_heap_bytes[heap], requirements.size);
        if (planned_heap_bytes[heap] > memory_properties.memoryHeaps[heap].size)
          throw std::invalid_argument("Complete packed source exceeds the selected Vulkan memory heap; no source was loaded");
        actual_required = add(actual_required, requirements.size);
        if (add(actual_required, admission.reserved_staging_bytes) > process_budget)
          throw std::invalid_argument("Aligned Vulkan allocations and staging exceed the reserved process budget");
        free_buffer(probe);
      } catch (...) { free_buffer(probe); throw; }
    };
    for (const auto bytes : {plane_bytes, plane_bytes, std::uint64_t{pixels} * 8,
                             plane_bytes, std::uint64_t{2048}}) account(bytes, true);
    account(8, true);
    account(2 * plane_bytes, false);
    account(2 * plane_bytes, false);
    account(std::uint64_t{pixels} * 4, true);
    account(std::uint64_t{pixels} * 4, true);
    if (has_prepared_dpc) {
      account(prepared_dpc_input_bytes, true);
      account(plane_bytes, true);
      account(plane_bytes, true);
      account(8U, true);
    }
    if (has_prepared_detector_products)
      for (std::uint32_t product = 0; product < 3U; ++product)
        account(plane_bytes, true);
    for (const auto item : plan) {
      account(std::max(std::uint64_t{4}, std::uint64_t{item.payload_words} * 4), true);
      const auto tiles = item.scan_count / item.scan_tile;
      const auto header_words = item.header_words != 0U
          ? std::uint64_t{item.header_words}
          : std::uint64_t{pixels} * tiles;
      account(header_words * 4U, true);
    }
    if (has_compressed_source) {
      const auto dispatch_width =
          std::min<std::uint64_t>(32768U,
                                  properties.limits.maxComputeWorkGroupCount[0]);
      if (!dispatch_width ||
          (maximum_chunk_count + dispatch_width - 1U) / dispatch_width >
              properties.limits.maxComputeWorkGroupCount[1])
        throw std::invalid_argument(
            "Compressed packed-source dispatch exceeds Vulkan workgroup limits");
      for (std::uint32_t slot = 0; slot < lz4_ring_depth; ++slot) {
        account((maximum_compressed_bytes + 3U) & ~std::uint64_t{3U}, true);
        account(maximum_chunk_count * 16U, true);
        account(maximum_chunk_count * 4U, true);
      }
    }
    const auto budget = current_heap_budget();
    for (std::uint32_t heap = 0; heap < memory_properties.memoryHeapCount; ++heap) {
      if (!planned_heap_bytes[heap]) continue;
      const auto headroom = budget.heapBudget[heap] > budget.heapUsage[heap]
          ? budget.heapBudget[heap] - budget.heapUsage[heap] : 0;
      admission.heaps.push_back({heap, memory_properties.memoryHeaps[heap].size,
          planned_heap_bytes[heap], budget.heapBudget[heap], budget.heapUsage[heap], headroom});
      if (admission.memory_budget_supported && planned_heap_bytes[heap] > headroom)
        throw std::invalid_argument("Complete packed source exceeds current Vulkan heap budget; no source was loaded");
    }
  }

  VkPhysicalDeviceMemoryBudgetPropertiesEXT current_heap_budget() {
    auto budget = structure<VkPhysicalDeviceMemoryBudgetPropertiesEXT>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
    if (admission.memory_budget_supported) {
      auto properties2 = structure<VkPhysicalDeviceMemoryProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2);
      properties2.pNext = &budget;
      vkGetPhysicalDeviceMemoryProperties2(physical, &properties2);
    }
    return budget;
  }

  void record_resident_heap_budget() {
    if (!admission.memory_budget_supported) return;
    const auto budget = current_heap_budget();
    for (auto& heap : admission.heaps) {
      const auto headroom = budget.heapBudget[heap.index] > budget.heapUsage[heap.index]
          ? budget.heapBudget[heap.index] - budget.heapUsage[heap.index] : 0;
      heap.minimum_headroom = std::min(heap.minimum_headroom, static_cast<std::uint64_t>(headroom));
      if (budget.heapUsage[heap.index] > budget.heapBudget[heap.index])
        throw std::runtime_error("Resident packed source exceeds the current Vulkan heap budget");
    }
  }

  Buffer create_buffer(VkDeviceSize bytes) {
    Buffer result;
    result.bytes = bytes;
    auto info = structure<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device, &info, nullptr, &result.buffer), "create buffer");
    return result;
  }

  std::uint32_t select_memory_type(std::uint32_t type_bits, bool host) const {
    std::uint32_t selected = UINT32_MAX;
    int best = -1;
    for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      const auto flags = memory_properties.memoryTypes[i].propertyFlags;
      if (!(type_bits & (1U << i)) || (host && !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) continue;
      const int score = ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 4 : 0) +
                        ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 2 : 0) +
                        ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
      if (score > best) { best = score; selected = i; }
    }
    if (selected == UINT32_MAX) throw std::runtime_error("No admitted Vulkan memory type for packed source");
    return selected;
  }

  Buffer allocate(VkDeviceSize bytes, bool host) {
    Buffer result;
    try {
      result = create_buffer(bytes);
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
      if (requirements.size > maximum_allocation ||
          add(add(admission.committed_bytes, requirements.size), admission.reserved_staging_bytes) > process_budget)
        throw std::invalid_argument("Actual Vulkan allocation/staging peak exceeds admitted budget");
      const auto selected = select_memory_type(requirements.memoryTypeBits, host);
      const auto heap = memory_properties.memoryTypes[selected].heapIndex;
      if (admission.memory_budget_supported) {
        const auto budget = current_heap_budget();
        const auto headroom = budget.heapBudget[heap] > budget.heapUsage[heap]
            ? budget.heapBudget[heap] - budget.heapUsage[heap] : 0;
        for (auto& entry : admission.heaps) if (entry.index == heap)
          entry.minimum_headroom = std::min(entry.minimum_headroom, static_cast<std::uint64_t>(headroom));
        if (planned_heap_bytes[heap] - committed_heap_bytes[heap] > headroom)
          throw std::invalid_argument("Current Vulkan heap budget eroded during complete source admission");
      }
      const auto heap_required = add(committed_heap_bytes[heap], requirements.size);
      if (heap_required > planned_heap_bytes[heap] ||
          heap_required > memory_properties.memoryHeaps[heap].size)
        throw std::invalid_argument("Actual Vulkan heap allocation exceeds the complete admitted plan");
      auto info_memory = structure<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
      info_memory.allocationSize = requirements.size; info_memory.memoryTypeIndex = selected;
      check(vkAllocateMemory(device, &info_memory, nullptr, &result.memory), "allocate memory");
      check(vkBindBufferMemory(device, result.buffer, result.memory, 0), "bind buffer");
      result.coherent = (memory_properties.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      if (host) check(vkMapMemory(device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "map buffer");
      result.memory_type_index = selected;
      result.allocated_bytes = requirements.size;
      admission.committed_bytes += requirements.size;
      committed_heap_bytes[heap] = heap_required;
      return result;
    } catch (...) { free_buffer(result); throw; }
  }

  PackedDetectorSourceMemory source_memory(const Buffer &buffer) const {
    const auto &type = memory_properties.memoryTypes[buffer.memory_type_index];
    return {buffer.memory_type_index, type.propertyFlags, type.heapIndex, 1, buffer.allocated_bytes};
  }

  void record_source_memory(const PackedDetectorSourceMemory &memory) {
    for (auto &entry : admission.source_memory) {
      if (entry.memory_type_index != memory.memory_type_index) continue;
      entry.allocation_count += memory.allocation_count;
      entry.allocated_bytes = add(entry.allocated_bytes, memory.allocated_bytes);
      return;
    }
    admission.source_memory.push_back(memory);
  }

  void flush(const Buffer &buffer) {
    if (buffer.coherent) return;
    auto range = structure<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory; range.size = VK_WHOLE_SIZE;
    check(vkFlushMappedMemoryRanges(device, 1, &range), "flush upload");
  }
  void invalidate(const Buffer &buffer) {
    if (buffer.coherent) return;
    auto range = structure<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory; range.size = VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidate readback");
  }
  void create_work_buffers() {
    for (auto &image : images) image = allocate(plane_bytes, true);
    entries = allocate(std::uint64_t{shape.detector_rows} * shape.detector_columns * 8, true);
    request_parameters = allocate(8, true);
    rows = allocate(2 * plane_bytes, false); columns = allocate(2 * plane_bytes, false);
    magnitude = allocate(plane_bytes, true); twiddles = allocate(2048, true);
    const auto pixels = shape.detector_rows * shape.detector_columns;
    diffraction = allocate(std::uint64_t{pixels} * 4, true);
    diffraction_exclusions = allocate(std::uint64_t{pixels} * 4, true);
    auto *mask = static_cast<std::uint32_t *>(diffraction_exclusions.mapped);
    for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
      mask[pixel] = excluded.empty() ? 0U : excluded[pixel];
    flush(diffraction_exclusions);
    auto *table = static_cast<float *>(twiddles.mapped);
    for (std::uint32_t i = 0; i < 256; ++i) {
      const auto angle = -2 * std::numbers::pi * i / 512;
      table[2*i] = static_cast<float>(std::cos(angle));
      table[2*i+1] = static_cast<float>(std::sin(angle));
    }
    table[0] = 1; table[1] = 0; table[256] = 0; table[257] = -1;
    flush(twiddles);
  }

  void create_prepared_dpc_buffers(
      std::span<const std::uint32_t> prepared_words) {
    if (!has_prepared_dpc || prepared_words.size_bytes() != prepared_dpc_input_bytes)
      throw std::invalid_argument(
          "Prepared DPC upload differs from its admitted exact range");
    prepared_dpc_moments = allocate(prepared_dpc_input_bytes, true);
    prepared_dpc_row = allocate(plane_bytes, true);
    prepared_dpc_column = allocate(plane_bytes, true);
    prepared_dpc_mean = allocate(8U, true);
    std::memcpy(prepared_dpc_moments.mapped, prepared_words.data(),
                prepared_words.size_bytes());
    flush(prepared_dpc_moments);
    record_source_memory(source_memory(prepared_dpc_moments));
    admission.source_upload_bytes =
        add(admission.source_upload_bytes, prepared_dpc_input_bytes);
    admission.prepared_dpc_bytes = prepared_dpc_input_bytes;
  }

  void create_prepared_detector_product_buffers(
      std::span<const PackedPreparedDetectorProduct> products) {
    if (!has_prepared_detector_products ||
        products.size() != prepared_detector_products.size())
      throw std::invalid_argument(
          "Prepared detector product upload differs from its admitted contract");
    for (std::size_t product = 0; product < products.size(); ++product) {
      auto &destination = prepared_detector_products[product];
      destination = allocate(plane_bytes, true);
      std::memcpy(destination.mapped, products[product].values.data(), plane_bytes);
      flush(destination);
      record_source_memory(source_memory(destination));
      admission.source_upload_bytes =
          add(admission.source_upload_bytes, plane_bytes);
    }
    admission.prepared_detector_product_bytes = 3U * plane_bytes;
    admission.prepared_detector_products_ready = true;
  }

  void validate_compact_headers_gpu() {
    const auto pixels = shape.detector_rows * shape.detector_columns;
    const auto cost_bytes = static_cast<VkDeviceSize>(pixels) * 4U;
    if (!entries.mapped || entries.bytes < cost_bytes ||
        !request_parameters.mapped || request_parameters.bytes < 4U)
      throw std::runtime_error(
          "Compact detector validation scratch is unavailable");
    std::fill_n(static_cast<std::uint32_t *>(entries.mapped), pixels, 0U);
    *static_cast<std::uint32_t *>(request_parameters.mapped) = 0U;
    flush(entries);
    flush(request_parameters);

    std::vector<std::size_t> compact_shards;
    for (std::size_t index = 0; index < shards.size(); ++index) {
      if (shards[index].header_encoding == compact_header_encoding)
        compact_shards.push_back(index);
    }
    if (compact_shards.empty())
      return;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool validation_command_pool = VK_NULL_HANDLE;
    VkFence validation_fence = VK_NULL_HANDLE;
    bool submitted = false;
    auto cleanup = [&] {
      if (submitted)
        vkDeviceWaitIdle(device);
      if (validation_fence)
        vkDestroyFence(device, validation_fence, nullptr);
      if (validation_command_pool)
        vkDestroyCommandPool(device, validation_command_pool, nullptr);
      if (pipeline)
        vkDestroyPipeline(device, pipeline, nullptr);
      if (pipeline_layout)
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (pool)
        vkDestroyDescriptorPool(device, pool, nullptr);
      if (layout)
        vkDestroyDescriptorSetLayout(device, layout, nullptr);
    };
    try {
      layout = make_layout(3U);
      const VkPushConstantRange push_range{
          VK_SHADER_STAGE_COMPUTE_BIT, 0U,
          sizeof(CompactHeaderValidationParameters)};
      auto pipeline_layout_info = structure<VkPipelineLayoutCreateInfo>(
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
      pipeline_layout_info.setLayoutCount = 1U;
      pipeline_layout_info.pSetLayouts = &layout;
      pipeline_layout_info.pushConstantRangeCount = 1U;
      pipeline_layout_info.pPushConstantRanges = &push_range;
      check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                   &pipeline_layout),
            "create compact header validation pipeline layout");
      pipeline = make_pipeline(kCompactHeaderValidate, pipeline_layout);

      const VkDescriptorPoolSize pool_size{
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          static_cast<std::uint32_t>(compact_shards.size() * 3U)};
      auto pool_info = structure<VkDescriptorPoolCreateInfo>(
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
      pool_info.maxSets = static_cast<std::uint32_t>(compact_shards.size());
      pool_info.poolSizeCount = 1U;
      pool_info.pPoolSizes = &pool_size;
      check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool),
            "create compact header validation descriptor pool");
      std::vector<VkDescriptorSetLayout> layouts(compact_shards.size(), layout);
      std::vector<VkDescriptorSet> sets(compact_shards.size());
      auto set_info = structure<VkDescriptorSetAllocateInfo>(
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
      set_info.descriptorPool = pool;
      set_info.descriptorSetCount =
          static_cast<std::uint32_t>(sets.size());
      set_info.pSetLayouts = layouts.data();
      check(vkAllocateDescriptorSets(device, &set_info, sets.data()),
            "allocate compact header validation descriptor sets");
      for (std::size_t ordinal = 0; ordinal < compact_shards.size();
           ++ordinal) {
        const auto &shard = shards[compact_shards[ordinal]];
        const std::array<VkDescriptorBufferInfo, 3> buffers{{
            {shard.headers.buffer, 0U, shard.headers.bytes},
            {entries.buffer, 0U, cost_bytes},
            {request_parameters.buffer, 0U, 4U},
        }};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
          writes[binding] = structure<VkWriteDescriptorSet>(
              VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
          writes[binding].dstSet = sets[ordinal];
          writes[binding].dstBinding = binding;
          writes[binding].descriptorCount = 1U;
          writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          writes[binding].pBufferInfo = &buffers[binding];
        }
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0U, nullptr);
      }

      auto command_pool_info = structure<VkCommandPoolCreateInfo>(
          VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
      command_pool_info.queueFamilyIndex = family;
      check(vkCreateCommandPool(device, &command_pool_info, nullptr,
                                &validation_command_pool),
            "create compact header validation command pool");
      VkCommandBuffer command_buffer = VK_NULL_HANDLE;
      auto command_info = structure<VkCommandBufferAllocateInfo>(
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
      command_info.commandPool = validation_command_pool;
      command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_info.commandBufferCount = 1U;
      check(vkAllocateCommandBuffers(device, &command_info, &command_buffer),
            "allocate compact header validation command");
      auto begin = structure<VkCommandBufferBeginInfo>(
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(command_buffer, &begin),
            "begin compact header validation");
      auto host_barrier = structure<VkMemoryBarrier>(
          VK_STRUCTURE_TYPE_MEMORY_BARRIER);
      host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      host_barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U,
                           &host_barrier, 0U, nullptr, 0U, nullptr);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline);
      for (std::size_t ordinal = 0; ordinal < compact_shards.size();
           ++ordinal) {
        const auto &shard = shards[compact_shards[ordinal]];
        const CompactHeaderValidationParameters parameters{
            pixels, shard.tiles, shard.header_words_per_pixel,
            shard.payload_words};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0U, 1U, &sets[ordinal], 0U,
                                nullptr);
        vkCmdPushConstants(command_buffer, pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(parameters), &parameters);
        vkCmdDispatch(command_buffer, (pixels + 255U) / 256U, 1U, 1U);
      }
      auto result_barrier = structure<VkMemoryBarrier>(
          VK_STRUCTURE_TYPE_MEMORY_BARRIER);
      result_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      result_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0U, 1U,
                           &result_barrier, 0U, nullptr, 0U, nullptr);
      check(vkEndCommandBuffer(command_buffer),
            "end compact header validation");
      auto fence_info = structure<VkFenceCreateInfo>(
          VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
      check(vkCreateFence(device, &fence_info, nullptr, &validation_fence),
            "create compact header validation fence");
      auto submit = structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
      submit.commandBufferCount = 1U;
      submit.pCommandBuffers = &command_buffer;
      check(vkQueueSubmit(queue, 1U, &submit, validation_fence),
            "submit compact header validation");
      submitted = true;
      check(vkWaitForFences(device, 1U, &validation_fence, VK_TRUE,
                            UINT64_MAX),
            "wait compact header validation");
      submitted = false;
      invalidate(entries);
      invalidate(request_parameters);
      const auto *results =
          static_cast<const std::uint32_t *>(entries.mapped);
      const auto status =
          *static_cast<const std::uint32_t *>(request_parameters.mapped);
      if (status != 0U)
        throw std::invalid_argument(
            "Compact detector GPU header validation failed with status " +
            std::to_string(status));
      for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
        column_bytes[pixel] += results[pixel];
      cleanup();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  VkDescriptorSetLayout make_layout(std::uint32_t count) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(count);
    for (std::uint32_t i = 0; i < count; ++i)
      bindings[i] = {i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    auto info = structure<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    info.bindingCount = count; info.pBindings = bindings.data();
    VkDescriptorSetLayout layout{};
    check(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout), "create descriptor layout");
    return layout;
  }
  VkPipelineLayout make_pipeline_layout(VkDescriptorSetLayout layout, bool push) {
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Parameters)};
    auto info = structure<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    info.setLayoutCount = 1; info.pSetLayouts = &layout;
    info.pushConstantRangeCount = push ? 1 : 0; info.pPushConstantRanges = push ? &range : nullptr;
    VkPipelineLayout result{};
    check(vkCreatePipelineLayout(device, &info, nullptr, &result), "create pipeline layout");
    return result;
  }
  VkPipeline make_pipeline(std::span<const std::uint32_t> code, VkPipelineLayout layout) {
    auto shader = structure<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    shader.codeSize = code.size_bytes(); shader.pCode = code.data();
    VkShaderModule module{};
    check(vkCreateShaderModule(device, &shader, nullptr, &module), "create shader");
    auto info = structure<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    info.stage = structure<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; info.stage.module = module; info.stage.pName = "main";
    info.layout = layout;
    VkPipeline result{};
    const auto status = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &result);
    vkDestroyShaderModule(device, module, nullptr);
    check(status, "create compute pipeline");
    return result;
  }
  void create_lz4_loader(std::span<const PackedDetectorShardSize> plan) {
    std::uint32_t maximum_compressed_bytes = 0U;
    std::uint32_t maximum_chunk_count = 0U;
    for (const auto &item : plan) {
      maximum_compressed_bytes =
          std::max(maximum_compressed_bytes, item.compressed_bytes);
      maximum_chunk_count =
          std::max(maximum_chunk_count, item.compressed_chunk_count);
    }
    if (maximum_compressed_bytes == 0U)
      return;
    lz4_slots.resize(lz4_ring_depth);
    for (auto &slot : lz4_slots) {
      slot.compressed = allocate(
          (static_cast<std::uint64_t>(maximum_compressed_bytes) + 3U) &
              ~std::uint64_t{3U},
          true);
      slot.metadata =
          allocate(static_cast<std::uint64_t>(maximum_chunk_count) * 16U,
                   true);
      slot.status =
          allocate(static_cast<std::uint64_t>(maximum_chunk_count) * 4U,
                   true);
    }
    lz4_layout = make_layout(4U);
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(PackedLz4Parameters)};
    auto layout_info = structure<VkPipelineLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    layout_info.setLayoutCount = 1U;
    layout_info.pSetLayouts = &lz4_layout;
    layout_info.pushConstantRangeCount = 1U;
    layout_info.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(device, &layout_info, nullptr,
                                 &lz4_pipeline_layout),
          "create packed LZ4 pipeline layout");
    lz4_pipeline = atomic_parallel_detector
        ? make_pipeline(kWorkgroupPackedLz4, lz4_pipeline_layout)
        : make_pipeline(kPackedLz4, lz4_pipeline_layout);
    lz4_scalar_pipeline = make_pipeline(kScalarPackedLz4,
                                        lz4_pipeline_layout);
    const VkDescriptorPoolSize pool_size{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, lz4_ring_depth * 4U};
    auto pool_info = structure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    pool_info.maxSets = lz4_ring_depth;
    pool_info.poolSizeCount = 1U;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr,
                                 &lz4_descriptor_pool),
          "create packed LZ4 descriptor pool");
    std::vector<VkDescriptorSetLayout> layouts(lz4_ring_depth, lz4_layout);
    std::vector<VkDescriptorSet> sets(lz4_ring_depth);
    auto set_info = structure<VkDescriptorSetAllocateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    set_info.descriptorPool = lz4_descriptor_pool;
    set_info.descriptorSetCount = lz4_ring_depth;
    set_info.pSetLayouts = layouts.data();
    check(vkAllocateDescriptorSets(device, &set_info, sets.data()),
          "allocate packed LZ4 descriptor sets");
    for (std::size_t index = 0; index < lz4_slots.size(); ++index)
      lz4_slots[index].set = sets[index];
    auto command_pool_info = structure<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    command_pool_info.queueFamilyIndex = family;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device, &command_pool_info, nullptr,
                              &lz4_command_pool),
          "create packed LZ4 command pool");
    std::vector<VkCommandBuffer> commands(lz4_ring_depth);
    auto command_info = structure<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = lz4_command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = lz4_ring_depth;
    check(vkAllocateCommandBuffers(device, &command_info, commands.data()),
          "allocate packed LZ4 command buffers");
    for (std::size_t index = 0; index < lz4_slots.size(); ++index)
      lz4_slots[index].command = commands[index];
    auto fence_info = structure<VkFenceCreateInfo>(
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    for (auto &slot : lz4_slots) {
      check(vkCreateFence(device, &fence_info, nullptr, &slot.fence),
            "create packed LZ4 fence");
      if (admission.timestamps_available) {
        auto query_info = structure<VkQueryPoolCreateInfo>(
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2U;
        check(vkCreateQueryPool(device, &query_info, nullptr, &slot.queries),
              "create packed LZ4 timestamps");
      }
    }
  }

  PackedDetectorLz4Metrics wait_lz4_slot(Lz4Slot &slot) {
    if (!slot.active)
      return slot.metrics;
    const auto wait_started = Clock::now();
    check(vkWaitForFences(device, 1U, &slot.fence, VK_TRUE, UINT64_MAX),
          "wait packed LZ4 decode");
    admission.load_timing.compressed_wait_milliseconds +=
        milliseconds(Clock::now() - wait_started);
    if (slot.queries) {
      std::array<std::uint64_t, 2> ticks{};
      check(vkGetQueryPoolResults(
                device, slot.queries, 0U, 2U, sizeof(ticks), ticks.data(),
                sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT),
            "read packed LZ4 timestamps");
      const auto mask = timestamp_bits == 64U
                            ? UINT64_MAX
                            : ((std::uint64_t{1} << timestamp_bits) - 1U);
      slot.metrics.gpu_decode_milliseconds =
          static_cast<double>((ticks[1] - ticks[0]) & mask) *
          properties.limits.timestampPeriod / 1.0e6;
      admission.load_timing.compressed_gpu_decode_milliseconds +=
          slot.metrics.gpu_decode_milliseconds;
    }
    invalidate(slot.status);
    invalidate(*slot.output);
    const auto *statuses =
        static_cast<const std::uint32_t *>(slot.status.mapped);
    for (std::uint32_t chunk = 0; chunk < slot.chunk_count; ++chunk) {
      if (statuses[chunk] != 0U)
        throw std::runtime_error(
            "Direct packed LZ4 GPU decode rejected chunk " +
            std::to_string(chunk) + " with status " +
            std::to_string(statuses[chunk]));
    }
    slot.metrics.ready_milliseconds =
        milliseconds(Clock::now() - slot.enqueued);
    slot.active = false;
    slot.output = nullptr;
    return slot.metrics;
  }

  Lz4Slot &enqueue_lz4_slot(
      Buffer &output, std::span<const std::uint8_t> compressed,
      std::span<const std::uint32_t> metadata) {
    if (!lz4_pipeline || lz4_slots.empty() || compressed.empty() || metadata.empty() ||
        metadata.size() % 4U != 0U)
      throw std::invalid_argument(
          "Direct packed LZ4 decode requires an admitted compressed shard");
    Lz4Slot &slot = lz4_slots[lz4_next_slot++ % lz4_slots.size()];
    wait_lz4_slot(slot);
    const auto enqueue_started = Clock::now();
    const std::uint32_t chunk_count =
        static_cast<std::uint32_t>(metadata.size() / 4U);
    const std::uint64_t padded_compressed =
        (compressed.size() + 3U) & ~std::uint64_t{3U};
    if (padded_compressed > slot.compressed.bytes ||
        metadata.size_bytes() > slot.metadata.bytes ||
        static_cast<std::uint64_t>(chunk_count) * 4U > slot.status.bytes ||
        compressed.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "Direct packed LZ4 shard exceeds its admitted staging plan");
    std::uint64_t expected_output = 0U;
    bool scalar_chunks = true;
    for (std::uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
      const std::uint32_t input_offset = metadata[chunk * 4U];
      const std::uint32_t input_bytes = metadata[chunk * 4U + 1U];
      const std::uint32_t output_word = metadata[chunk * 4U + 2U];
      const std::uint32_t output_bytes = metadata[chunk * 4U + 3U];
      if (!input_bytes || input_offset > compressed.size() ||
          input_bytes > compressed.size() - input_offset ||
          !output_bytes || (output_bytes & 3U) != 0U ||
          static_cast<std::uint64_t>(output_word) * 4U != expected_output)
        throw std::invalid_argument(
            "Direct packed LZ4 metadata is noncanonical");
      expected_output += output_bytes;
      scalar_chunks = scalar_chunks && output_bytes <= 4096U;
    }
    if (expected_output != output.bytes)
      throw std::invalid_argument(
          "Direct packed LZ4 chunks do not cover the resident payload");
    slot.metrics = {};
    slot.metrics.compressed_bytes = compressed.size();
    slot.metrics.chunk_count = chunk_count;
    const auto staging_started = Clock::now();
    std::memcpy(slot.compressed.mapped, compressed.data(), compressed.size());
    if (padded_compressed != compressed.size())
      std::memset(static_cast<std::uint8_t *>(slot.compressed.mapped) +
                      compressed.size(),
                  0, padded_compressed - compressed.size());
    std::memcpy(slot.metadata.mapped, metadata.data(), metadata.size_bytes());
    slot.metrics.staging_milliseconds =
        milliseconds(Clock::now() - staging_started);
    admission.load_timing.compressed_staging_milliseconds +=
        slot.metrics.staging_milliseconds;
    const auto visibility_started = Clock::now();
    flush(slot.compressed);
    flush(slot.metadata);
    slot.metrics.vulkan_visibility_milliseconds =
        milliseconds(Clock::now() - visibility_started);

    std::array<VkDescriptorBufferInfo, 4> buffers{{
        {slot.compressed.buffer, 0, padded_compressed},
        {slot.metadata.buffer, 0, metadata.size_bytes()},
        {output.buffer, 0, output.bytes},
        {slot.status.buffer, 0,
         static_cast<VkDeviceSize>(chunk_count) * 4U},
    }};
    std::array<VkWriteDescriptorSet, 4> writes{};
    for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
      writes[binding] = structure<VkWriteDescriptorSet>(
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
      writes[binding].dstSet = slot.set;
      writes[binding].dstBinding = binding;
      writes[binding].descriptorCount = 1U;
      writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[binding].pBufferInfo = &buffers[binding];
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0U, nullptr);
    check(vkResetFences(device, 1U, &slot.fence),
          "reset packed LZ4 fence");
    check(vkResetCommandBuffer(slot.command, 0U),
          "reset packed LZ4 command buffer");
    auto begin = structure<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(slot.command, &begin),
          "begin packed LZ4 command buffer");
    if (slot.queries) {
      vkCmdResetQueryPool(slot.command, slot.queries, 0U, 2U);
      vkCmdWriteTimestamp(slot.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          slot.queries, 0U);
    }
    const std::uint32_t dispatch_width = std::min<std::uint32_t>(
        32768U, properties.limits.maxComputeWorkGroupCount[0]);
    const PackedLz4Parameters parameters{
        chunk_count, static_cast<std::uint32_t>(compressed.size()),
        dispatch_width};
    vkCmdBindPipeline(slot.command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      scalar_chunks ? lz4_scalar_pipeline : lz4_pipeline);
    vkCmdBindDescriptorSets(slot.command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lz4_pipeline_layout, 0U, 1U, &slot.set, 0U,
                            nullptr);
    vkCmdPushConstants(slot.command, lz4_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(parameters),
                       &parameters);
    if (scalar_chunks) {
      const auto groups = (chunk_count + 63U) / 64U;
      if (groups > properties.limits.maxComputeWorkGroupCount[0])
        throw std::invalid_argument(
            "Scalar packed LZ4 chunk dispatch exceeds Vulkan limits");
      vkCmdDispatch(slot.command, groups, 1U, 1U);
    } else {
      vkCmdDispatch(slot.command, std::min(chunk_count, dispatch_width),
                    (chunk_count + dispatch_width - 1U) / dispatch_width, 1U);
    }
    if (slot.queries)
      vkCmdWriteTimestamp(slot.command,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, slot.queries,
                          1U);
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    const std::array<VkBuffer, 2> barrier_buffers{output.buffer,
                                                  slot.status.buffer};
    const std::array<VkDeviceSize, 2> barrier_sizes{
        output.bytes, static_cast<VkDeviceSize>(chunk_count) * 4U};
    for (std::size_t index = 0; index < barriers.size(); ++index) {
      barriers[index] = structure<VkBufferMemoryBarrier>(
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
      barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barriers[index].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].buffer = barrier_buffers[index];
      barriers[index].size = barrier_sizes[index];
    }
    vkCmdPipelineBarrier(slot.command,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0U, 0U, nullptr,
                         static_cast<std::uint32_t>(barriers.size()),
                         barriers.data(), 0U, nullptr);
    check(vkEndCommandBuffer(slot.command),
          "end packed LZ4 command buffer");
    auto submit =
        structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &slot.command;
    check(vkQueueSubmit(queue, 1U, &submit, slot.fence),
          "submit packed LZ4 decode");
    slot.output = &output;
    slot.chunk_count = chunk_count;
    slot.enqueued = Clock::now();
    slot.active = true;
    admission.load_timing.compressed_enqueue_milliseconds +=
        milliseconds(Clock::now() - enqueue_started);
    return slot;
  }

  PackedDetectorLz4Metrics enqueue_lz4(
      Buffer &output, std::span<const std::uint8_t> compressed,
      std::span<const std::uint32_t> metadata) {
    return enqueue_lz4_slot(output, compressed, metadata).metrics;
  }

  PackedDetectorLz4Metrics decode_lz4(
      Buffer &output, std::span<const std::uint8_t> compressed,
      std::span<const std::uint32_t> metadata) {
    return wait_lz4_slot(enqueue_lz4_slot(output, compressed, metadata));
  }

  void finish_lz4_loader() {
    for (auto &slot : lz4_slots)
      wait_lz4_slot(slot);
  }
  VkDescriptorSet make_set(VkDescriptorSetLayout layout, std::span<const Buffer *const> buffers) {
    auto info = structure<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    info.descriptorPool = descriptor_pool; info.descriptorSetCount = 1; info.pSetLayouts = &layout;
    VkDescriptorSet result{};
    check(vkAllocateDescriptorSets(device, &info, &result), "allocate descriptor set");
    std::vector<VkDescriptorBufferInfo> descriptions(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
      descriptions[i] = {buffers[i]->buffer,0,buffers[i]->bytes};
      writes[i] = structure<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
      writes[i].dstSet = result; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &descriptions[i];
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return result;
  }

  std::uint32_t detector_workgroups(
      const ResidentShard &shard, std::uint32_t mode) const {
    if (mode == 0U)
      return (shard.scans + 127U) / 128U;
    if (mode != 0U && !atomic_parallel_detector) {
      const auto tiles_per_group = 128U / shard.scan_tile;
      return (shard.tiles + tiles_per_group - 1U) / tiles_per_group;
    }
    return (shard.scans + 31U) / 32U;
  }

  void create_pipelines_and_descriptors() {
    detector_layout = make_layout(6); fft_layout = make_layout(7);
    diffraction_layout = make_layout(4);
    if (has_prepared_dpc) prepared_dpc_layout = make_layout(3);
    detector_pipeline_layout = make_pipeline_layout(detector_layout, true);
    fft_pipeline_layout = make_pipeline_layout(fft_layout, false);
    diffraction_pipeline_layout = make_pipeline_layout(diffraction_layout, true);
    if (has_prepared_dpc)
      prepared_dpc_pipeline_layout =
          make_pipeline_layout(prepared_dpc_layout, true);
    // Real Android GPUs keep the physically profiled shared-memory kernel.
    // The emulator virtual GPU uses the barrier-free variant because the
    // bundled host MoltenVK compiler rejects workgroup barriers on current macOS.
    pipelines[0] = atomic_parallel_detector
        ? make_pipeline(kAtomicDetector, detector_pipeline_layout)
        : make_pipeline(kTileDetector, detector_pipeline_layout);
    pipelines[1] = make_pipeline(kRows, fft_pipeline_layout);
    pipelines[2] = make_pipeline(kColumns, fft_pipeline_layout);
    pipelines[3] = make_pipeline(kMagnitude, fft_pipeline_layout);
    pipelines[4] = make_pipeline(kSelectedDiffraction, diffraction_pipeline_layout);
    pipelines[5] = make_pipeline(kScalarDetector, detector_pipeline_layout);
    if (has_prepared_dpc) {
      prepared_dpc_pipelines[0] =
          make_pipeline(kPreparedDpcCom, prepared_dpc_pipeline_layout);
      prepared_dpc_pipelines[1] =
          make_pipeline(kPreparedDpcMean, prepared_dpc_pipeline_layout);
      prepared_dpc_pipelines[2] =
          make_pipeline(kPreparedDpcCenter, prepared_dpc_pipeline_layout);
    }
    const auto count = static_cast<std::uint32_t>(shards.size());
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        2*count*6+14+count*4+(has_prepared_dpc ? 9U : 0U)};
    auto pool = structure<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    pool.maxSets = 3*count+2+(has_prepared_dpc ? 3U : 0U);
    pool.poolSizeCount = 1; pool.pPoolSizes = &size;
    check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool), "create descriptor pool");
    for (std::uint32_t bank = 0; bank < 2; ++bank) {
      detector_sets[bank].reserve(shards.size());
      for (const auto &shard : shards) {
        // The first request seeds image 1 from image 0. Once seeded, exact
        // deltas are serialized by the session fence and may safely update
        // each scan value in image 1 in place. This avoids the physical Fold8
        // failure observed whenever the swapped bank wrote image 0.
        const auto output_bank = 1U;
        const std::array<const Buffer *,6> buffers{
            &shard.payload,&shard.headers,&entries,&images[bank],&images[output_bank],
            &request_parameters};
        detector_sets[bank].push_back(make_set(detector_layout,buffers));
      }
      const std::array<const Buffer *,7> buffers{&images[bank],&images[bank],&images[bank],&rows,&columns,&magnitude,&twiddles};
      fft_sets[bank] = make_set(fft_layout,buffers);
    }
    diffraction_sets.reserve(shards.size());
    for (const auto &shard : shards) {
      const std::array<const Buffer *,4> buffers{
          &shard.payload,&shard.headers,&diffraction_exclusions,&diffraction};
      diffraction_sets.push_back(make_set(diffraction_layout,buffers));
    }
    if (has_prepared_dpc) {
      const std::array<const Buffer *, 3> buffers{
          &prepared_dpc_moments, &prepared_dpc_row, &prepared_dpc_column};
      prepared_dpc_sets[0] = make_set(prepared_dpc_layout, buffers);
      const std::array<const Buffer *, 3> derived{
          &prepared_dpc_row, &prepared_dpc_column, &prepared_dpc_mean};
      prepared_dpc_sets[1] = make_set(prepared_dpc_layout, derived);
      prepared_dpc_sets[2] = make_set(prepared_dpc_layout, derived);
    }
  }
  void create_commands() {
    auto pool = structure<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool.queueFamilyIndex = family; pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device,&pool,nullptr,&command_pool), "create command pool");
    auto allocate_info = structure<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    allocate_info.commandPool = command_pool; allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 7;
    std::array<VkCommandBuffer, 7> allocated{};
    check(vkAllocateCommandBuffers(device,&allocate_info,allocated.data()), "allocate command buffers");
    command = allocated[0];
    for (std::uint32_t bank = 0; bank < detector_commands.size(); ++bank)
      for (std::uint32_t mode = 0; mode < detector_commands[bank].size(); ++mode)
        detector_commands[bank][mode] = allocated[1U + bank * 3U + mode];
    auto fence_info = structure<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    check(vkCreateFence(device,&fence_info,nullptr,&fence), "create fence");
    if (admission.timestamps_available) {
      auto info = structure<VkQueryPoolCreateInfo>(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
      info.queryType = VK_QUERY_TYPE_TIMESTAMP; info.queryCount = 8;
      check(vkCreateQueryPool(device,&info,nullptr,&queries), "create timestamps");
    }
    for (std::uint32_t bank = 0; bank < detector_commands.size(); ++bank) {
      for (std::uint32_t command_mode = 0;
           command_mode < detector_commands[bank].size(); ++command_mode) {
      const auto recorded = detector_commands[bank][command_mode];
      auto begin = structure<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
      check(vkBeginCommandBuffer(recorded,&begin), "begin reusable detector request");
      if (queries) vkCmdResetQueryPool(recorded,queries,0,2);
      if (atomic_parallel_detector && command_mode == 1U)
        vkCmdFillBuffer(recorded, images[1].buffer, 0, plane_bytes, 0U);
      auto barrier = structure<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
      barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT |
                              VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(recorded,
          VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
              VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
      if (queries) vkCmdWriteTimestamp(recorded,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,0);
      vkCmdBindPipeline(recorded,VK_PIPELINE_BIND_POINT_COMPUTE,
          pipelines[command_mode == 0U ? 5U : 0U]);
      for (std::size_t i = 0; i < shards.size(); ++i) {
        const auto &shard = shards[i];
        const Parameters parameters{
            shard.scans, shard.tiles, 0U, shard.first_scan, 0U,
            shard.scan_tile, shard.header_encoding, shard.header_words_per_pixel};
        vkCmdBindDescriptorSets(recorded,VK_PIPELINE_BIND_POINT_COMPUTE,
            detector_pipeline_layout,0,1,&detector_sets[bank][i],0,nullptr);
        vkCmdPushConstants(recorded,detector_pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,
            0,sizeof(parameters),&parameters);
        vkCmdDispatch(recorded,detector_workgroups(shard, command_mode),1,1);
      }
      if (queries) vkCmdWriteTimestamp(recorded,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,1);
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(recorded,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
      check(vkEndCommandBuffer(recorded), "end reusable detector request");
      }
    }
  }

  void prime_prepared_dpc() {
    const auto started = Clock::now();
    const std::uint32_t scans = shape.scan_rows * shape.scan_columns;
    check(vkResetFences(device, 1U, &fence), "reset prepared DPC fence");
    check(vkResetCommandBuffer(command, 0U), "reset prepared DPC command");
    auto begin = structure<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "begin prepared DPC prime");
    if (queries) {
      vkCmdResetQueryPool(command, queries, 0U, 2U);
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queries, 0U);
    }
    auto barrier = structure<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U,
                         &barrier, 0U, nullptr, 0U, nullptr);
    for (std::uint32_t stage = 0; stage < prepared_dpc_pipelines.size(); ++stage) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        prepared_dpc_pipelines[stage]);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              prepared_dpc_pipeline_layout, 0U, 1U,
                              &prepared_dpc_sets[stage], 0U, nullptr);
      vkCmdPushConstants(command, prepared_dpc_pipeline_layout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(scans),
                         &scans);
      vkCmdDispatch(command, stage == 1U ? 1U : (scans + 255U) / 256U,
                    1U, 1U);
      if (stage + 1U != prepared_dpc_pipelines.size()) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U,
                             &barrier, 0U, nullptr, 0U, nullptr);
      }
    }
    if (queries)
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queries, 1U);
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0U, 1U, &barrier, 0U,
                         nullptr, 0U, nullptr);
    check(vkEndCommandBuffer(command), "end prepared DPC prime");
    auto submit = structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1U, &submit, fence),
          "submit prepared DPC prime");
    check(vkWaitForFences(device, 1U, &fence, VK_TRUE, UINT64_MAX),
          "wait prepared DPC prime");
    if (queries) {
      std::array<std::uint64_t, 2> ticks{};
      check(vkGetQueryPoolResults(device, queries, 0U, 2U, sizeof(ticks),
                                  ticks.data(), sizeof(std::uint64_t),
                                  VK_QUERY_RESULT_64_BIT),
            "read prepared DPC timestamps");
      const auto mask = timestamp_bits == 64U
          ? UINT64_MAX
          : ((std::uint64_t{1} << timestamp_bits) - 1U);
      admission.prepared_dpc_gpu_milliseconds =
          static_cast<double>((ticks[1] - ticks[0]) & mask) *
          properties.limits.timestampPeriod / 1.0e6;
    }
    admission.prepared_dpc_prime_milliseconds =
        milliseconds(Clock::now() - started);
    admission.prepared_dpc_ready = true;
  }

  PackedDetectorSessionMetrics request(CircularDetector geometry, std::uint64_t generation,
      std::span<std::uint32_t> output, std::span<float> fft) {
    const auto scan_pixels =
        std::size_t{shape.scan_rows} * shape.scan_columns;
    if (output.size() != scan_pixels ||
        (!fft.empty() && (fft.size() != scan_pixels || shape.scan_rows != 512)))
      throw std::invalid_argument(
          "Exact image destination must match the scan; FFT is qualified only for 512x512");
    std::lock_guard lock(mutex);
    if (failed) throw std::runtime_error("Packed Vulkan session failed; reopen authenticated source");
    const auto start = Clock::now();
    auto next_mask = circular_detector_mask(shape.detector_rows,shape.detector_columns,geometry,excluded);
    std::uint32_t prepared_product = UINT32_MAX;
    if (has_prepared_detector_products) {
      for (std::uint32_t product = 0;
           product < prepared_detector_geometries.size(); ++product) {
        const auto &prepared_geometry = prepared_detector_geometries[product];
        if (prepared_geometry.center_row == geometry.center_row &&
            prepared_geometry.center_column == geometry.center_column &&
            prepared_geometry.inner_radius == geometry.inner_radius &&
            prepared_geometry.outer_radius == geometry.outer_radius) {
          prepared_product = product;
          next_mask = prepared_detector_masks[product];
          break;
        }
      }
    }
    const auto update = plan_packed_detector_update(committed_mask,next_mask,column_bytes);
    const auto mask_plan_time = milliseconds(Clock::now()-start);
    const bool source_work = update.rebase || !update.entries.empty();
    const bool prepared_source_work = source_work && prepared_product != UINT32_MAX;
    const bool fft_work = !fft.empty() && (source_work || !fft_current);
    // Image 1 is the single serialized committed image after the initial seed.
    const auto target = source_work ? 1U : committed_image;
    PackedDetectorSessionMetrics result;
    result.mask_plan_milliseconds = mask_plan_time;
    auto &metrics = result.timing;
    metrics.generation = generation; result.rebase = update.rebase;
    result.source_changed = source_work;
    // Report authenticated prepared-product provenance even when this exact
    // product is already committed and therefore needs no redundant GPU copy.
    result.prepared_detector_product = prepared_product != UINT32_MAX;
    result.prepared_detector_product_index = prepared_product;
    result.changed_pixel_entries = static_cast<std::uint32_t>(update.entries.size());
    result.logical_source_bytes = update.logical_source_bytes;
    try {
      if (source_work || fft_work) {
        if (!update.entries.empty()) {
          std::memcpy(entries.mapped,update.entries.data(),update.entries.size()*sizeof(DetectorPixelChange));
          flush(entries);
        }
        auto *request_words = static_cast<std::uint32_t *>(request_parameters.mapped);
        request_words[0] = result.changed_pixel_entries;
        // Tiny deltas keep one lane per scan. On a physical integrated GPU,
        // rebases and wider deltas map one invocation to each scan in a packed
        // tile and cooperatively cache each detector column's descriptor and
        // payload. The emulator retains the barrier-free four-lane path.
        constexpr std::uint32_t parallel_delta_entries = 64U;
        const auto request_mode = update.rebase ? 1U :
            (result.changed_pixel_entries >= parallel_delta_entries ? 2U : 0U);
        request_words[1] = request_mode;
        flush(request_parameters);
        check(vkResetFences(device,1,&fence), "reset fence");
        const bool reusable_detector_only =
            source_work && !fft_work && !prepared_source_work;
        const auto command_mode = request_mode;
        auto submitted_command = reusable_detector_only
            ? detector_commands[committed_image][command_mode] : command;
        if (reusable_detector_only) {
          metrics.dispatch_count = static_cast<std::uint32_t>(shards.size());
        } else {
          check(vkResetCommandBuffer(command,0), "reset command buffer");
          auto begin = structure<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
          begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
          check(vkBeginCommandBuffer(command,&begin), "begin request");
          if (queries) vkCmdResetQueryPool(command,queries,0,8);
          if (atomic_parallel_detector && request_mode == 1U)
            vkCmdFillBuffer(command, images[1].buffer, 0, plane_bytes, 0U);
          auto barrier = structure<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
          barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                                  VK_ACCESS_SHADER_WRITE_BIT |
                                  VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT |
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
          auto stamp = [&](std::uint32_t query, VkPipelineStageFlagBits stage =
                                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
            if (queries) vkCmdWriteTimestamp(command,stage,queries,query);
          };
          stamp(0, prepared_source_work ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
          if (prepared_source_work) {
            VkBufferCopy copy{0U, 0U, plane_bytes};
            vkCmdCopyBuffer(command,
                            prepared_detector_products[prepared_product].buffer,
                            images[target].buffer, 1U, &copy);
          } else if (source_work) {
            vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelines[request_mode == 0U ? 5U : 0U]);
            for (std::size_t i = 0; i < shards.size(); ++i) {
              const auto &shard = shards[i];
              const Parameters parameters{
                  shard.scans, shard.tiles, 0U, shard.first_scan, 0U,
                  shard.scan_tile, shard.header_encoding, shard.header_words_per_pixel};
              vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,detector_pipeline_layout,
                  0,1,&detector_sets[committed_image][i],0,nullptr);
              vkCmdPushConstants(command,detector_pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(parameters),&parameters);
              vkCmdDispatch(command,detector_workgroups(shard, request_mode),1,1);
              ++metrics.dispatch_count;
            }
          }
          stamp(1, prepared_source_work ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
          barrier.srcAccessMask = prepared_source_work
              ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          if (fft_work) {
            vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,fft_pipeline_layout,0,1,&fft_sets[target],0,nullptr);
            for (std::uint32_t stage = 1; stage <= 3; ++stage) {
              vkCmdPipelineBarrier(command,
                  prepared_source_work && stage == 1U ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                                      : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  0,1,&barrier,0,nullptr,0,nullptr);
              stamp(stage*2);
              vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[stage]);
              vkCmdDispatch(command,stage == 3 ? 1024 : 512,1,1);
              stamp(stage*2+1);
              ++metrics.dispatch_count;
            }
          }
          barrier.srcAccessMask = fft_work
              ? VK_ACCESS_SHADER_WRITE_BIT
              : (prepared_source_work ? VK_ACCESS_TRANSFER_WRITE_BIT
                                      : VK_ACCESS_SHADER_WRITE_BIT);
          barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
          vkCmdPipelineBarrier(command,
              fft_work ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                  : (prepared_source_work ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                          : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
              VK_PIPELINE_STAGE_HOST_BIT,
              0,1,&barrier,0,nullptr,0,nullptr);
          check(vkEndCommandBuffer(command), "end request");
        }
        auto submit = structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit.commandBufferCount = 1; submit.pCommandBuffers = &submitted_command;
        check(vkQueueSubmit(queue,1,&submit,fence), "submit detector and FFT");
        metrics.queue_submit_count = 1;
        check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX), "wait exact result");
        metrics.fence_wait_count = 1;
        // This commit is independent of output/UI publication or newer input.
        if (source_work) {
          committed_image = target; committed_mask = std::move(next_mask); fft_current = false;
        }
        if (fft_work) fft_current = true;
        if (queries) {
          std::array<std::uint64_t,8> ticks{};
          const auto query_count = fft_work ? 8U : 2U;
          check(vkGetQueryPoolResults(device,queries,0,query_count,query_count*sizeof(std::uint64_t),ticks.data(),
              sizeof(std::uint64_t),VK_QUERY_RESULT_64_BIT), "read timestamps");
          const auto mask = timestamp_bits == 64 ? UINT64_MAX : ((std::uint64_t{1} << timestamp_bits)-1);
          auto elapsed = [&](std::uint32_t a, std::uint32_t b) {
            return double((ticks[b]-ticks[a]) & mask) * properties.limits.timestampPeriod / 1e6;
          };
          if (source_work) metrics.gpu_annulus_milliseconds = elapsed(0,1);
          metrics.gpu_total_milliseconds = elapsed(0,query_count-1);
          if (fft_work) {
            metrics.gpu_fft_rows_milliseconds = elapsed(2,3);
            metrics.gpu_fft_columns_milliseconds = elapsed(4,5);
            metrics.gpu_magnitude_milliseconds = elapsed(6,7);
          }
        }
      }
      committed_generation = generation;
      result.committed_generation = committed_generation;
      const auto copy_start = Clock::now();
      invalidate(images[committed_image]);
      std::memcpy(output.data(),images[committed_image].mapped,plane_bytes);
      if (!fft.empty()) {
        invalidate(magnitude);
        std::memcpy(fft.data(),magnitude.mapped,plane_bytes);
      }
      metrics.output_copy_bytes = fft.empty() ? plane_bytes : 2*plane_bytes;
      metrics.output_copy_milliseconds = milliseconds(Clock::now()-copy_start);
      // Residency is fixed after admission: pointer-rate detector requests do
      // not allocate, upload, or grow any Vulkan memory. The admission check
      // above remains authoritative; querying the driver budget every frame
      // adds a device call without making an in-flight allocation safer.
      metrics.wall_milliseconds = milliseconds(Clock::now()-start);
      return result;
    } catch (...) { failed = true; throw; }
  }

  PackedSelectedDiffractionMetrics selected_diffraction(std::uint32_t row,
      std::uint32_t column, std::uint64_t generation, std::span<std::uint32_t> output) {
    const auto pixels = shape.detector_rows * shape.detector_columns;
    if (row >= shape.scan_rows || column >= shape.scan_columns)
      throw std::invalid_argument("Selected diffraction (row, column) is outside the admitted scan shape");
    if (output.size() != pixels)
      throw std::invalid_argument("Selected diffraction destination must contain exactly detector_rows*detector_columns uint32 values");
    const auto start = Clock::now();
    std::lock_guard lock(mutex);
    const auto locked = Clock::now();
    if (failed) throw std::runtime_error("Packed Vulkan session failed; reopen authenticated source");
    const auto scan = row * shape.scan_columns + column;
    // The complete admitted plan starts at zero, has positive shard lengths and
    // covers every scan exactly once, including non-tile-aligned shard tails.
    const auto after = std::upper_bound(shards.begin(),shards.end(),scan,
        [](std::uint32_t value, const ResidentShard &shard) { return value < shard.first_scan; });
    const auto index = static_cast<std::size_t>(after-shards.begin()-1);
    const auto &shard = shards[index];
    PackedSelectedDiffractionMetrics result;
    result.generation = generation; result.row = row; result.column = column;
    result.detector_rows = shape.detector_rows; result.detector_columns = shape.detector_columns;
    result.shard_index = static_cast<std::uint32_t>(index);
    result.shard_local_scan = scan-shard.first_scan;
    result.mutex_wait_milliseconds = milliseconds(locked-start);
    try {
      check(vkResetFences(device,1,&fence), "reset selected diffraction fence");
      check(vkResetCommandBuffer(command,0), "reset selected diffraction command buffer");
      auto begin = structure<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(command,&begin), "begin selected diffraction");
      if (queries) vkCmdResetQueryPool(command,queries,0,2);
      auto barrier = structure<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
      barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
      if (queries) vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,0);
      const SelectedDiffractionParameters parameters{
          result.shard_local_scan, shard.tiles, pixels, shard.scan_tile,
          shard.header_encoding, shard.header_words_per_pixel};
      vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[4]);
      vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,diffraction_pipeline_layout,
          0,1,&diffraction_sets[index],0,nullptr);
      vkCmdPushConstants(command,diffraction_pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,
          0,sizeof(parameters),&parameters);
      vkCmdDispatch(command,(pixels+127)/128,1,1);
      result.dispatch_count = 1;
      if (queries) vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,1);
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
          0,1,&barrier,0,nullptr,0,nullptr);
      check(vkEndCommandBuffer(command), "end selected diffraction");
      auto submit = structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
      submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
      check(vkQueueSubmit(queue,1,&submit,fence), "submit selected diffraction");
      result.queue_submit_count = 1;
      check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX), "wait selected diffraction");
      result.fence_wait_count = 1;
      if (queries) {
        std::array<std::uint64_t,2> ticks{};
        check(vkGetQueryPoolResults(device,queries,0,2,sizeof(ticks),ticks.data(),sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT), "read selected diffraction timestamps");
        const auto mask = timestamp_bits == 64 ? UINT64_MAX : ((std::uint64_t{1} << timestamp_bits)-1);
        result.gpu_decode_milliseconds = double((ticks[1]-ticks[0]) & mask) * properties.limits.timestampPeriod / 1e6;
      }
      // Same strict resident-memory check as detector requests, before exposing
      // this output. No detector mask/image/generation or FFT state is changed.
      record_resident_heap_budget();
      const auto copy_start = Clock::now();
      invalidate(diffraction);
      std::memcpy(output.data(),diffraction.mapped,diffraction.bytes);
      result.output_copy_bytes = diffraction.bytes;
      result.output_copy_milliseconds = milliseconds(Clock::now()-copy_start);
      result.wall_milliseconds = milliseconds(Clock::now()-start);
      return result;
    } catch (...) { failed = true; throw; }
  }

  PackedPreparedDpcMetrics prepared_dpc(std::span<float> row,
                                        std::span<float> column) {
    const auto scans = std::size_t{shape.scan_rows} * shape.scan_columns;
    if (!admission.prepared_dpc_ready)
      throw std::invalid_argument(
          "This source has no authenticated prepared DPC moments");
    if (row.size() != scans || column.size() != scans)
      throw std::invalid_argument(
          "Prepared DPC destinations must match the admitted scan shape");
    const auto started = Clock::now();
    std::lock_guard lock(mutex);
    if (failed)
      throw std::runtime_error(
          "Packed Vulkan session failed; reopen authenticated source");
    PackedPreparedDpcMetrics result;
    try {
      const auto copy_started = Clock::now();
      invalidate(prepared_dpc_row);
      invalidate(prepared_dpc_column);
      std::memcpy(row.data(), prepared_dpc_row.mapped, plane_bytes);
      std::memcpy(column.data(), prepared_dpc_column.mapped, plane_bytes);
      result.output_copy_bytes = 2U * plane_bytes;
      result.output_copy_milliseconds =
          milliseconds(Clock::now() - copy_started);
      result.wall_milliseconds = milliseconds(Clock::now() - started);
      return result;
    } catch (...) {
      failed = true;
      throw;
    }
  }

  void free_buffer(Buffer &buffer) noexcept {
    if (buffer.mapped) vkUnmapMemory(device,buffer.memory);
    if (buffer.buffer) vkDestroyBuffer(device,buffer.buffer,nullptr);
    if (buffer.memory) vkFreeMemory(device,buffer.memory,nullptr);
    buffer = {};
  }
  void destroy() noexcept {
    if (device) {
      vkDeviceWaitIdle(device); // Teardown only, never per interaction.
      for (auto &slot : lz4_slots) {
        if (slot.queries) vkDestroyQueryPool(device,slot.queries,nullptr);
        if (slot.fence) vkDestroyFence(device,slot.fence,nullptr);
      }
      if (lz4_command_pool) vkDestroyCommandPool(device,lz4_command_pool,nullptr);
      if (lz4_pipeline) vkDestroyPipeline(device,lz4_pipeline,nullptr);
      if (lz4_scalar_pipeline) vkDestroyPipeline(device,lz4_scalar_pipeline,nullptr);
      if (lz4_pipeline_layout) vkDestroyPipelineLayout(device,lz4_pipeline_layout,nullptr);
      if (lz4_descriptor_pool) vkDestroyDescriptorPool(device,lz4_descriptor_pool,nullptr);
      if (lz4_layout) vkDestroyDescriptorSetLayout(device,lz4_layout,nullptr);
      if (queries) vkDestroyQueryPool(device,queries,nullptr);
      if (fence) vkDestroyFence(device,fence,nullptr);
      if (command_pool) vkDestroyCommandPool(device,command_pool,nullptr);
      for (auto pipeline : pipelines) if (pipeline) vkDestroyPipeline(device,pipeline,nullptr);
      for (auto pipeline : prepared_dpc_pipelines)
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
      if (detector_pipeline_layout) vkDestroyPipelineLayout(device,detector_pipeline_layout,nullptr);
      if (fft_pipeline_layout) vkDestroyPipelineLayout(device,fft_pipeline_layout,nullptr);
      if (diffraction_pipeline_layout) vkDestroyPipelineLayout(device,diffraction_pipeline_layout,nullptr);
      if (prepared_dpc_pipeline_layout)
        vkDestroyPipelineLayout(device, prepared_dpc_pipeline_layout, nullptr);
      if (descriptor_pool) vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
      if (detector_layout) vkDestroyDescriptorSetLayout(device,detector_layout,nullptr);
      if (fft_layout) vkDestroyDescriptorSetLayout(device,fft_layout,nullptr);
      if (diffraction_layout) vkDestroyDescriptorSetLayout(device,diffraction_layout,nullptr);
      if (prepared_dpc_layout)
        vkDestroyDescriptorSetLayout(device, prepared_dpc_layout, nullptr);
      for (auto &shard : shards) { free_buffer(shard.payload); free_buffer(shard.headers); }
      for (auto &image : images) free_buffer(image);
      free_buffer(entries); free_buffer(request_parameters); free_buffer(rows); free_buffer(columns);
      free_buffer(magnitude); free_buffer(twiddles);
      free_buffer(diffraction); free_buffer(diffraction_exclusions);
      free_buffer(prepared_dpc_moments); free_buffer(prepared_dpc_row);
      free_buffer(prepared_dpc_column); free_buffer(prepared_dpc_mean);
      for (auto &product : prepared_detector_products) free_buffer(product);
      for (auto &slot : lz4_slots) {
        free_buffer(slot.compressed); free_buffer(slot.metadata);
        free_buffer(slot.status);
      }
      vkDestroyDevice(device,nullptr);
    }
    if (instance) vkDestroyInstance(instance,nullptr);
    device = VK_NULL_HANDLE; instance = VK_NULL_HANDLE;
  }
};

PackedDetectorSession::PackedDetectorSession(Shape4D shape,
    std::span<const PackedDetectorShardSize> plan, ShardLoadGuard before_shard,
    AuthenticatedShardLoader loader,
    std::span<const std::uint8_t> excluded, std::uint64_t budget, std::uint64_t staging)
    : PackedDetectorSession(shape, plan, std::move(before_shard),
          std::move(loader), excluded, {}, budget, staging) {}
PackedDetectorSession::PackedDetectorSession(Shape4D shape,
    std::span<const PackedDetectorShardSize> plan, ShardLoadGuard before_shard,
    AuthenticatedShardLoader loader,
    std::span<const std::uint8_t> excluded,
    std::span<const std::uint32_t> prepared_dpc_words,
    std::uint64_t budget, std::uint64_t staging)
    : PackedDetectorSession(shape, plan, std::move(before_shard),
          std::move(loader), excluded, prepared_dpc_words, {}, budget, staging) {}
PackedDetectorSession::PackedDetectorSession(Shape4D shape,
    std::span<const PackedDetectorShardSize> plan, ShardLoadGuard before_shard,
    AuthenticatedShardLoader loader,
    std::span<const std::uint8_t> excluded,
    std::span<const std::uint32_t> prepared_dpc_words,
    std::span<const PackedPreparedDetectorProduct> prepared_products,
    std::uint64_t budget, std::uint64_t staging)
    : impl_(std::make_unique<Impl>(shape,plan,before_shard,loader,excluded,
          prepared_dpc_words,prepared_products,budget,staging)) {}
PackedDetectorSession::~PackedDetectorSession() = default;
const PackedDetectorSessionAdmission &PackedDetectorSession::admission() const noexcept { return impl_->admission; }
PackedDetectorSessionMetrics PackedDetectorSession::request(CircularDetector geometry,
    std::uint64_t generation, std::span<std::uint32_t> image, std::span<float> fft) {
  return impl_->request(geometry,generation,image,fft);
}
PackedSelectedDiffractionMetrics PackedDetectorSession::selected_diffraction(std::uint32_t row,
    std::uint32_t column, std::uint64_t generation, std::span<std::uint32_t> diffraction) {
  return impl_->selected_diffraction(row,column,generation,diffraction);
}
PackedPreparedDpcMetrics PackedDetectorSession::prepared_dpc(
    std::span<float> row, std::span<float> column) {
  return impl_->prepared_dpc(row, column);
}
} // namespace quantem::gpu::vulkan
