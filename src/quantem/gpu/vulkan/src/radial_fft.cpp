#include "quantem/gpu/vulkan/radial_fft.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantem::gpu::vulkan {
namespace {

const std::uint32_t kDifference[] =
#include "radial_difference_spirv.inc"
    ;
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
constexpr VkDeviceSize kPlaneBytes = RadialFftSession::plane_values * sizeof(std::uint32_t);
constexpr VkDeviceSize kPrefixBytes = kPlaneBytes * RadialFftSession::prefix_planes;

double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

template <typename T> T structure(VkStructureType type) {
  T value{};
  value.sType = type;
  return value;
}

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string("Radial FFT ") + operation +
                             " failed with Vulkan result " + std::to_string(result));
  }
}

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize bytes = 0;
  void *mapped = nullptr;
  bool coherent = false;
};

} // namespace

struct RadialFftSession::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t family = 0;
  std::uint32_t timestamp_bits = 0;
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkDeviceSize max_allocation_bytes = 0;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  std::array<VkPipeline, 4> pipelines{};
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  Buffer prefix, image, row_fft, column_fft, magnitude, twiddles;
  RadialFftAdmission admission;
  std::mutex mutex;
  bool failed = false;

  explicit Impl(std::span<const std::uint32_t> values) {
    const auto start = Clock::now();
    if (values.size() != RadialFftSession::plane_values * RadialFftSession::prefix_planes) {
      throw std::invalid_argument("Radial prefix must be uint32[137,512,512]; no crop or binning is implicit");
    }
    const auto validation = Clock::now();
    for (std::uint32_t i = 0; i < RadialFftSession::plane_values; ++i) {
      if (values[i] != 0) throw std::invalid_argument("Radial P[0] must be zero; rebuild the source-authenticated prefix");
    }
    for (std::size_t i = RadialFftSession::plane_values; i < values.size(); ++i) {
      if (values[i] < values[i - RadialFftSession::plane_values]) {
        throw std::invalid_argument("Radial prefix is not cumulative uint32; rebuild the source-authenticated prefix");
      }
    }
    admission.validation_milliseconds = milliseconds(Clock::now() - validation);
    try {
      initialize();
      const auto upload = Clock::now();
      std::memcpy(prefix.mapped, values.data(), kPrefixBytes);
      flush(prefix);
      admission.prefix_copy_milliseconds = milliseconds(Clock::now() - upload);
      admission.prefix_upload_bytes = kPrefixBytes;
      admission.initialization_milliseconds = milliseconds(Clock::now() - start);
    } catch (...) {
      destroy();
      throw;
    }
  }

  ~Impl() { destroy(); }

  void initialize() {
    auto application = structure<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.pApplicationName = "QuantEM resident radial FFT";
    application.apiVersion = VK_API_VERSION_1_1;
    auto instance_info = structure<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "create instance");
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "count devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
    for (auto candidate : devices) {
      std::uint32_t family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
      for (std::uint32_t i = 0; i < family_count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          physical = candidate;
          family = i;
          timestamp_bits = families[i].timestampValidBits;
          break;
        }
      }
      if (physical != VK_NULL_HANDLE) break;
    }
    if (physical == VK_NULL_HANDLE) throw std::runtime_error("Radial FFT needs a Vulkan compute device");
    auto maintenance = structure<VkPhysicalDeviceMaintenance3Properties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
    auto properties2 = structure<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    properties2.pNext = &maintenance;
    vkGetPhysicalDeviceProperties2(physical, &properties2);
    properties = properties2.properties;
    max_allocation_bytes = maintenance.maxMemoryAllocationSize;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    const auto &limits = properties.limits;
    if (limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 256 ||
        limits.maxComputeSharedMemorySize < 512 * sizeof(float) * 2 ||
        limits.maxComputeWorkGroupCount[0] < 1024 || limits.maxStorageBufferRange < 2 * kPlaneBytes ||
        limits.maxPerStageDescriptorStorageBuffers < 7 || limits.maxDescriptorSetStorageBuffers < 7 ||
        (limits.minStorageBufferOffsetAlignment != 0 && kPlaneBytes % limits.minStorageBufferOffsetAlignment != 0)) {
      throw std::runtime_error("Vulkan limits cannot support the exact 512-square radial FFT; keep the source unchanged");
    }
    admission.device_name = properties.deviceName;
    admission.driver_version = properties.driverVersion;
    admission.timestamps_available = timestamp_bits != 0;
    const float priority = 1.0F;
    auto queue_info = structure<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    auto device_info = structure<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical, &device_info, nullptr, &device), "create device");
    vkGetDeviceQueue(device, family, 0, &queue);

    prefix = allocate(kPrefixBytes, true);
    image = allocate(kPlaneBytes, true);
    row_fft = allocate(2 * kPlaneBytes, false);
    column_fft = allocate(2 * kPlaneBytes, false);
    magnitude = allocate(kPlaneBytes, true);
    twiddles = allocate(256 * sizeof(float) * 2, true);
    auto *table = static_cast<float *>(twiddles.mapped);
    for (std::uint32_t i = 0; i < 256; ++i) {
      const double phase = -2.0 * std::numbers::pi * i / 512.0;
      table[2 * i] = static_cast<float>(std::cos(phase));
      table[2 * i + 1] = static_cast<float>(std::sin(phase));
    }
    table[0] = 1.0F; table[1] = 0.0F;
    table[256] = 0.0F; table[257] = -1.0F;
    flush(twiddles);

    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    auto layout_info = structure<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_layout), "create descriptor layout");
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
    auto pool_info = structure<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "create descriptor pool");
    auto allocate_info = structure<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    allocate_info.descriptorPool = descriptor_pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &descriptor_layout;
    check(vkAllocateDescriptorSets(device, &allocate_info, &descriptor), "allocate descriptors");
    updateDescriptors(0, 1);
    auto pipeline_info = structure<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    pipeline_info.setLayoutCount = 1;
    pipeline_info.pSetLayouts = &descriptor_layout;
    check(vkCreatePipelineLayout(device, &pipeline_info, nullptr, &pipeline_layout), "create pipeline layout");
    pipelines[0] = makePipeline(kDifference, sizeof(kDifference));
    pipelines[1] = makePipeline(kRows, sizeof(kRows));
    pipelines[2] = makePipeline(kColumns, sizeof(kColumns));
    pipelines[3] = makePipeline(kMagnitude, sizeof(kMagnitude));
    auto command_info = structure<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    command_info.queueFamilyIndex = family;
    command_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device, &command_info, nullptr, &command_pool), "create command pool");
    auto command_allocate = structure<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_allocate.commandPool = command_pool;
    command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device, &command_allocate, &command), "allocate command buffer");
    auto fence_info = structure<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "create fence");
    if (admission.timestamps_available) {
      auto query_info = structure<VkQueryPoolCreateInfo>(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
      query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_info.queryCount = 8;
      check(vkCreateQueryPool(device, &query_info, nullptr, &queries), "create timestamps");
    }
  }

  Buffer allocate(VkDeviceSize bytes, bool host) {
    Buffer buffer;
    buffer.bytes = bytes;
    try {
      auto info = structure<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
      info.size = bytes;
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      check(vkCreateBuffer(device, &info, nullptr, &buffer.buffer), "create buffer");
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
      if (requirements.size > max_allocation_bytes) throw std::runtime_error("Radial workspace exceeds Vulkan allocation limit");
      std::uint32_t selected = UINT32_MAX;
      int best_score = -1;
      for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const auto flags = memory_properties.memoryTypes[i].propertyFlags;
        if (!(requirements.memoryTypeBits & (1u << i)) || (host && !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) continue;
        const int score = ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 4 : 0) +
                          ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 2 : 0) +
                          ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
        if (score > best_score) { selected = i; best_score = score; }
      }
      if (selected == UINT32_MAX) throw std::runtime_error("No suitable Vulkan memory for radial workspace");
      auto allocation = structure<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = selected;
      check(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory), "allocate buffer memory");
      check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "bind buffer");
      buffer.coherent = (memory_properties.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      if (host) check(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "map scalar buffer");
      admission.committed_bytes += requirements.size;
      return buffer;
    } catch (...) { freeBuffer(buffer); throw; }
  }

  void flush(const Buffer &buffer) {
    if (buffer.coherent) return;
    auto range = structure<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory;
    range.size = VK_WHOLE_SIZE;
    check(vkFlushMappedMemoryRanges(device, 1, &range), "flush host upload");
  }

  void invalidate(const Buffer &buffer) {
    if (buffer.coherent) return;
    auto range = structure<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory;
    range.size = VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidate scalar readback");
  }

  VkPipeline makePipeline(const std::uint32_t *code, std::size_t bytes) {
    VkShaderModule module = VK_NULL_HANDLE;
    auto module_info = structure<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    module_info.codeSize = bytes;
    module_info.pCode = code;
    check(vkCreateShaderModule(device, &module_info, nullptr, &module), "create shader");
    auto info = structure<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    info.stage = structure<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    check(result, "create compute pipeline");
    return pipeline;
  }

  void updateDescriptors(std::uint32_t inner, std::uint32_t outer) {
    const std::array<VkDescriptorBufferInfo, 7> infos{{
        {prefix.buffer, inner * kPlaneBytes, kPlaneBytes},
        {prefix.buffer, outer * kPlaneBytes, kPlaneBytes},
        {image.buffer, 0, image.bytes}, {row_fft.buffer, 0, row_fft.bytes},
        {column_fft.buffer, 0, column_fft.bytes}, {magnitude.buffer, 0, magnitude.bytes},
        {twiddles.buffer, 0, twiddles.bytes}}};
    std::array<VkWriteDescriptorSet, 7> writes{};
    for (std::uint32_t i = 0; i < writes.size(); ++i) {
      writes[i] = structure<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
      writes[i].dstSet = descriptor;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  RadialFftMetrics request(std::uint32_t inner, std::uint32_t outer, std::uint64_t generation,
                          std::span<std::uint32_t> output, std::span<float> fft) {
    if (inner >= outer || outer >= RadialFftSession::prefix_planes) {
      throw std::invalid_argument("Detector radii require integer pixels 0 <= inner < outer <= 136");
    }
    if (output.size() != RadialFftSession::plane_values ||
        (!fft.empty() && fft.size() != RadialFftSession::plane_values)) {
      throw std::invalid_argument("Radial image/FFT destinations must contain exactly 512 x 512 values");
    }
    std::lock_guard lock(mutex);
    if (failed) throw std::runtime_error("Radial Vulkan session failed; recreate it from the authenticated prefix");
    const auto start = Clock::now();
    RadialFftMetrics metrics;
    metrics.generation = generation;
    const std::uint32_t stages = fft.empty() ? 1 : 4;
    try {
      // The preceding request waited for its own fence. No descriptor/buffer is in flight.
      updateDescriptors(inner, outer);
      check(vkResetFences(device, 1, &fence), "reset request fence");
      check(vkResetCommandBuffer(command, 0), "reset request command");
      auto begin = structure<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(command, &begin), "begin radial request");
      if (queries != VK_NULL_HANDLE) vkCmdResetQueryPool(command, queries, 0, stages * 2);
      auto barrier = structure<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
      barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &barrier, 0, nullptr, 0, nullptr);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                              &descriptor, 0, nullptr);
      for (std::uint32_t i = 0; i < stages; ++i) {
        if (queries != VK_NULL_HANDLE) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, i * 2);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
        vkCmdDispatch(command, (i == 0 || i == 3) ? 1024 : 512, 1, 1);
        if (queries != VK_NULL_HANDLE) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, i * 2 + 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        if (i + 1 < stages) {
          vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
      }
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                           0, 1, &barrier, 0, nullptr, 0, nullptr);
      check(vkEndCommandBuffer(command), "end radial request");
      auto submit = structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command;
      check(vkQueueSubmit(queue, 1, &submit, fence), "submit radial request");
      metrics.queue_submit_count = 1;
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "wait radial result");
      metrics.fence_wait_count = 1;
      metrics.dispatch_count = stages;
      if (queries != VK_NULL_HANDLE) {
        std::array<std::uint64_t, 8> ticks{};
        check(vkGetQueryPoolResults(device, queries, 0, stages * 2, stages * 2 * sizeof(std::uint64_t),
                                    ticks.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT), "read GPU timings");
        const std::uint64_t mask = timestamp_bits == 64 ? UINT64_MAX : ((std::uint64_t{1} << timestamp_bits) - 1);
        const auto elapsed = [&](std::uint32_t a, std::uint32_t b) {
          return static_cast<double>((ticks[b] - ticks[a]) & mask) * properties.limits.timestampPeriod / 1.0e6;
        };
        metrics.gpu_annulus_milliseconds = elapsed(0, 1);
        metrics.gpu_total_milliseconds = elapsed(0, stages * 2 - 1);
        if (stages == 4) {
          metrics.gpu_fft_rows_milliseconds = elapsed(2, 3);
          metrics.gpu_fft_columns_milliseconds = elapsed(4, 5);
          metrics.gpu_magnitude_milliseconds = elapsed(6, 7);
        }
      }
      const auto copy_start = Clock::now();
      invalidate(image);
      if (!fft.empty()) invalidate(magnitude);
      std::memcpy(output.data(), image.mapped, kPlaneBytes);
      if (!fft.empty()) std::memcpy(fft.data(), magnitude.mapped, kPlaneBytes);
      metrics.output_copy_bytes = fft.empty() ? kPlaneBytes : 2 * kPlaneBytes;
      metrics.output_copy_milliseconds = milliseconds(Clock::now() - copy_start);
      metrics.wall_milliseconds = milliseconds(Clock::now() - start);
      return metrics;
    } catch (...) {
      failed = true;
      throw;
    }
  }

  void freeBuffer(Buffer &buffer) noexcept {
    if (buffer.mapped != nullptr) vkUnmapMemory(device, buffer.memory);
    if (buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
  }

  void destroy() noexcept {
    if (device != VK_NULL_HANDLE) {
      // Lifetime boundary only. Requests never synchronize the entire device.
      vkDeviceWaitIdle(device);
      if (queries != VK_NULL_HANDLE) vkDestroyQueryPool(device, queries, nullptr);
      if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
      if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
      for (auto pipeline : pipelines) if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
      if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      if (descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
      freeBuffer(twiddles); freeBuffer(magnitude); freeBuffer(column_fft);
      freeBuffer(row_fft); freeBuffer(image); freeBuffer(prefix);
      vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
  }
};

RadialFftSession::RadialFftSession(std::span<const std::uint32_t> prefix)
    : impl_(std::make_unique<Impl>(prefix)) {}
RadialFftSession::~RadialFftSession() = default;
const RadialFftAdmission &RadialFftSession::admission() const noexcept { return impl_->admission; }
RadialFftMetrics RadialFftSession::request(std::uint32_t inner, std::uint32_t outer,
    std::uint64_t generation, std::span<std::uint32_t> image, std::span<float> fft) {
  return impl_->request(inner, outer, generation, image, fft);
}

} // namespace quantem::gpu::vulkan
