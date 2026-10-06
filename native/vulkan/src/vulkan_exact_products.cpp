#include "quantem/gpu/vulkan/exact_products.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace quantem::gpu::vulkan {
namespace {

const std::uint32_t kScanProductsSpirv[] =
#include "scan_products_u8_spirv.inc"
    ;

const std::uint32_t kMeanDiffractionSpirv[] =
#include "mean_diffraction_u8_spirv.inc"
    ;

const std::uint32_t kScanProductsUint16Spirv[] =
#include "scan_products_u16_spirv.inc"
    ;

const std::uint32_t kMeanDiffractionUint16Spirv[] =
#include "mean_diffraction_u16_spirv.inc"
    ;

const std::uint32_t kQh5Lz4DecodeSpirv[] =
#include "qh5_lz4_decode_spirv.inc"
    ;

const std::uint32_t kQh5BitunshuffleUint16Spirv[] =
#include "qh5_bitunshuffle_u16_spirv.inc"
    ;

const std::uint32_t kPackedLz4DecodeSpirv[] =
#include "packed_lz4_decode_spirv.inc"
    ;

using Clock = std::chrono::steady_clock;

double milliseconds(const Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

double timeval_milliseconds(const timeval value) {
  return static_cast<double>(value.tv_sec) * 1000.0 +
         static_cast<double>(value.tv_usec) / 1000.0;
}

template <typename Value> Value vulkan_structure(const VkStructureType type) {
  Value value{};
  value.sType = type;
  return value;
}

void check(const VkResult result, const char *operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with Vulkan result " +
                             std::to_string(result));
  }
}

std::uint64_t checked_product(const std::uint64_t left,
                              const std::uint64_t right, const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " exceeds uint64 range");
  }
  return left * right;
}

struct Buffer {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize requested_bytes = 0;
  VkDeviceSize committed_bytes = 0;
  void *mapped = nullptr;
  bool coherent = false;
  bool device_local = false;

  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept { *this = std::move(other); }
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      destroy();
      device = other.device;
      buffer = other.buffer;
      memory = other.memory;
      requested_bytes = other.requested_bytes;
      committed_bytes = other.committed_bytes;
      mapped = other.mapped;
      coherent = other.coherent;
      device_local = other.device_local;
      other.device = VK_NULL_HANDLE;
      other.buffer = VK_NULL_HANDLE;
      other.memory = VK_NULL_HANDLE;
      other.mapped = nullptr;
    }
    return *this;
  }
  ~Buffer() { destroy(); }

  void destroy() {
    if (device != VK_NULL_HANDLE && mapped != nullptr) {
      vkUnmapMemory(device, memory);
    }
    if (device != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, buffer, nullptr);
    }
    if (device != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, nullptr);
    }
    device = VK_NULL_HANDLE;
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    mapped = nullptr;
  }
};

struct VulkanContext {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkPhysicalDeviceVulkan11Properties properties11{};
  VkPhysicalDeviceMaintenance3Properties maintenance3{};
  VkPhysicalDeviceFeatures features{};
  VkPhysicalDeviceVulkan12Features features12{};
  std::uint32_t timestamp_valid_bits = 0;

  VulkanContext() {
    auto application =
        vulkan_structure<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.pApplicationName = "quantem.gpu Android Vulkan";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "quantem.gpu";
    application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_1;
    auto instance_info = vulkan_structure<VkInstanceCreateInfo>(
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &instance),
          "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    if (device_count == 0) {
      throw std::runtime_error("no Vulkan physical device is available");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices(list)");
    physical_device = devices.front();

    properties11 = vulkan_structure<VkPhysicalDeviceVulkan11Properties>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES);
    maintenance3 = vulkan_structure<VkPhysicalDeviceMaintenance3Properties>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
    properties11.pNext = &maintenance3;
    auto properties2 = vulkan_structure<VkPhysicalDeviceProperties2>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    properties2.pNext = &properties11;
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    properties = properties2.properties;

    features12 = vulkan_structure<VkPhysicalDeviceVulkan12Features>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    auto features2 = vulkan_structure<VkPhysicalDeviceFeatures2>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    features2.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    features = features2.features;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count,
                                             families.data());
    bool found = false;
    for (std::uint32_t index = 0; index < family_count; ++index) {
      if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        queue_family = index;
        timestamp_valid_bits = families[index].timestampValidBits;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("no Vulkan compute queue family is available");
    }

    constexpr float priority = 1.0F;
    auto queue_info = vulkan_structure<VkDeviceQueueCreateInfo>(
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    auto device_info = vulkan_structure<VkDeviceCreateInfo>(
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderInt64 = features.shaderInt64;
    auto enabled_features12 =
        vulkan_structure<VkPhysicalDeviceVulkan12Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    enabled_features12.storageBuffer8BitAccess =
        features12.storageBuffer8BitAccess;
    device_info.pNext = &enabled_features12;
    device_info.pEnabledFeatures = &enabled_features;
    check(vkCreateDevice(physical_device, &device_info, nullptr, &device),
          "vkCreateDevice");
    vkGetDeviceQueue(device, queue_family, 0, &queue);
  }

  VulkanContext(const VulkanContext &) = delete;
  VulkanContext &operator=(const VulkanContext &) = delete;
  ~VulkanContext() {
    if (device != VK_NULL_HANDLE) {
      vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
      vkDestroyInstance(instance, nullptr);
    }
  }

  std::uint32_t memory_type(const std::uint32_t type_bits,
                            const VkMemoryPropertyFlags required,
                            const VkMemoryPropertyFlags preferred) const {
    for (int pass = 0; pass < 2; ++pass) {
      for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount;
           ++index) {
        const VkMemoryPropertyFlags flags =
            memory_properties.memoryTypes[index].propertyFlags;
        if ((type_bits & (1U << index)) != 0 &&
            (flags & required) == required &&
            (pass != 0 || (flags & preferred) == preferred)) {
          return index;
        }
      }
    }
    throw std::runtime_error(
        "no Vulkan memory type satisfies the required flags");
  }

  Buffer make_host_buffer(const VkDeviceSize bytes,
                          const VkBufferUsageFlags usage,
                          const bool prefer_device_local = true) const {
    Buffer result;
    result.device = device;
    result.requested_bytes = bytes;
    auto info = vulkan_structure<VkBufferCreateInfo>(
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device, &info, nullptr, &result.buffer),
          "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
    const VkMemoryPropertyFlags preferred =
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        (prefer_device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0U);
    const std::uint32_t type_index =
        memory_type(requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, preferred);
    const VkMemoryPropertyFlags flags =
        memory_properties.memoryTypes[type_index].propertyFlags;
    result.coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    result.device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    result.committed_bytes = requirements.size;
    auto allocate = vulkan_structure<VkMemoryAllocateInfo>(
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type_index;
    check(vkAllocateMemory(device, &allocate, nullptr, &result.memory),
          "vkAllocateMemory");
    check(vkBindBufferMemory(device, result.buffer, result.memory, 0),
          "vkBindBufferMemory");
    check(vkMapMemory(device, result.memory, 0, bytes, 0, &result.mapped),
          "vkMapMemory");
    return result;
  }

  void flush(const Buffer &buffer, const VkDeviceSize bytes) const {
    if (buffer.coherent)
      return;
    auto range = vulkan_structure<VkMappedMemoryRange>(
        VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = bytes;
    check(vkFlushMappedMemoryRanges(device, 1, &range),
          "vkFlushMappedMemoryRanges");
  }

  void invalidate(const Buffer &buffer) const {
    if (buffer.coherent)
      return;
    auto range = vulkan_structure<VkMappedMemoryRange>(
        VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(device, 1, &range),
          "vkInvalidateMappedMemoryRanges");
  }
};

VulkanCapabilities query_capabilities(const VulkanContext &context) {
  bool host_visible_device_local = false;
  for (std::uint32_t index = 0;
       index < context.memory_properties.memoryTypeCount; ++index) {
    const auto flags =
        context.memory_properties.memoryTypes[index].propertyFlags;
    host_visible_device_local |=
        (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }
  return {
      context.properties.deviceName,
      context.properties.apiVersion,
      context.properties.driverVersion,
      context.properties.vendorID,
      context.properties.deviceID,
      context.properties.limits.maxStorageBufferRange,
      context.maintenance3.maxMemoryAllocationSize,
      context.properties11.subgroupSize,
      context.timestamp_valid_bits,
      context.properties.limits.timestampPeriod,
      context.features.shaderInt16 == VK_TRUE,
      context.features.shaderInt64 == VK_TRUE,
      context.features12.storageBuffer8BitAccess == VK_TRUE,
      host_visible_device_local,
  };
}

struct PipelineResources {
  VkDevice device = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout qh5_lz4_descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout qh5_lz4_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout qh5_unshuffle_descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout qh5_unshuffle_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout packed_lz4_descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout packed_lz4_pipeline_layout = VK_NULL_HANDLE;
  VkShaderModule scan_module = VK_NULL_HANDLE;
  VkShaderModule mean_module = VK_NULL_HANDLE;
  VkShaderModule scan_uint16_module = VK_NULL_HANDLE;
  VkShaderModule mean_uint16_module = VK_NULL_HANDLE;
  VkShaderModule qh5_lz4_module = VK_NULL_HANDLE;
  VkShaderModule qh5_unshuffle_module = VK_NULL_HANDLE;
  VkShaderModule packed_lz4_module = VK_NULL_HANDLE;
  VkPipeline scan_pipeline = VK_NULL_HANDLE;
  VkPipeline mean_pipeline = VK_NULL_HANDLE;
  VkPipeline scan_uint16_pipeline = VK_NULL_HANDLE;
  VkPipeline mean_uint16_pipeline = VK_NULL_HANDLE;
  VkPipeline qh5_lz4_pipeline = VK_NULL_HANDLE;
  VkPipeline qh5_unshuffle_pipeline = VK_NULL_HANDLE;
  VkPipeline packed_lz4_pipeline = VK_NULL_HANDLE;

  explicit PipelineResources(VkDevice target_device) : device(target_device) {}
  PipelineResources(const PipelineResources &) = delete;
  PipelineResources &operator=(const PipelineResources &) = delete;
  ~PipelineResources() {
    if (scan_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, scan_pipeline, nullptr);
    if (mean_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, mean_pipeline, nullptr);
    if (scan_uint16_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, scan_uint16_pipeline, nullptr);
    if (mean_uint16_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, mean_uint16_pipeline, nullptr);
    if (qh5_lz4_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, qh5_lz4_pipeline, nullptr);
    if (qh5_unshuffle_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, qh5_unshuffle_pipeline, nullptr);
    if (packed_lz4_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, packed_lz4_pipeline, nullptr);
    if (scan_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, scan_module, nullptr);
    if (mean_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, mean_module, nullptr);
    if (scan_uint16_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, scan_uint16_module, nullptr);
    if (mean_uint16_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, mean_uint16_module, nullptr);
    if (qh5_lz4_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, qh5_lz4_module, nullptr);
    if (qh5_unshuffle_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, qh5_unshuffle_module, nullptr);
    if (packed_lz4_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, packed_lz4_module, nullptr);
    if (pipeline_layout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    if (qh5_lz4_pipeline_layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, qh5_lz4_pipeline_layout, nullptr);
    if (qh5_unshuffle_pipeline_layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, qh5_unshuffle_pipeline_layout, nullptr);
    if (packed_lz4_pipeline_layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, packed_lz4_pipeline_layout, nullptr);
    if (qh5_lz4_descriptor_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, qh5_lz4_descriptor_layout, nullptr);
    if (qh5_unshuffle_descriptor_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, qh5_unshuffle_descriptor_layout,
                                   nullptr);
    if (packed_lz4_descriptor_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, packed_lz4_descriptor_layout,
                                   nullptr);
  }
};

VkShaderModule make_shader(const VkDevice device, const std::uint32_t *words,
                           const std::size_t word_count) {
  auto info = vulkan_structure<VkShaderModuleCreateInfo>(
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
  info.codeSize = word_count * sizeof(std::uint32_t);
  info.pCode = words;
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device, &info, nullptr, &module),
        "vkCreateShaderModule");
  return module;
}

VkPipeline make_pipeline(const VkDevice device, const VkPipelineLayout layout,
                         const VkShaderModule module) {
  auto info = vulkan_structure<VkComputePipelineCreateInfo>(
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
  info.layout = layout;
  info.stage = vulkan_structure<VkPipelineShaderStageCreateInfo>(
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
  info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = module;
  info.stage.pName = "main";
  VkPipeline pipeline = VK_NULL_HANDLE;
  check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                 &pipeline),
        "vkCreateComputePipelines");
  return pipeline;
}

std::unique_ptr<PipelineResources> make_pipelines(const VkDevice device) {
  auto result = std::make_unique<PipelineResources>(device);
  std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
  for (std::uint32_t index = 0; index < bindings.size(); ++index) {
    bindings[index].binding = index;
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  auto descriptor_info = vulkan_structure<VkDescriptorSetLayoutCreateInfo>(
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
  descriptor_info.bindingCount = bindings.size();
  descriptor_info.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr,
                                    &result->descriptor_layout),
        "vkCreateDescriptorSetLayout");

  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0;
  push.size = 8 * sizeof(std::uint32_t);
  auto layout_info = vulkan_structure<VkPipelineLayoutCreateInfo>(
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &result->descriptor_layout;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  check(vkCreatePipelineLayout(device, &layout_info, nullptr,
                               &result->pipeline_layout),
        "vkCreatePipelineLayout");
  result->scan_module =
      make_shader(device, kScanProductsSpirv,
                  sizeof(kScanProductsSpirv) / sizeof(kScanProductsSpirv[0]));
  result->mean_module = make_shader(device, kMeanDiffractionSpirv,
                                    sizeof(kMeanDiffractionSpirv) /
                                        sizeof(kMeanDiffractionSpirv[0]));
  result->scan_uint16_module = make_shader(
      device, kScanProductsUint16Spirv,
      sizeof(kScanProductsUint16Spirv) / sizeof(kScanProductsUint16Spirv[0]));
  result->mean_uint16_module =
      make_shader(device, kMeanDiffractionUint16Spirv,
                  sizeof(kMeanDiffractionUint16Spirv) /
                      sizeof(kMeanDiffractionUint16Spirv[0]));
  result->scan_pipeline =
      make_pipeline(device, result->pipeline_layout, result->scan_module);
  result->mean_pipeline =
      make_pipeline(device, result->pipeline_layout, result->mean_module);
  result->scan_uint16_pipeline = make_pipeline(device, result->pipeline_layout,
                                               result->scan_uint16_module);
  result->mean_uint16_pipeline = make_pipeline(device, result->pipeline_layout,
                                               result->mean_uint16_module);

  const auto make_storage_layout = [&](const std::uint32_t binding_count) {
    std::vector<VkDescriptorSetLayoutBinding> storage_bindings(binding_count);
    for (std::uint32_t index = 0; index < binding_count; ++index) {
      storage_bindings[index].binding = index;
      storage_bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      storage_bindings[index].descriptorCount = 1;
      storage_bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    auto storage_info = vulkan_structure<VkDescriptorSetLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    storage_info.bindingCount = binding_count;
    storage_info.pBindings = storage_bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(device, &storage_info, nullptr, &layout),
          "vkCreateDescriptorSetLayout(QH5)");
    return layout;
  };
  const auto make_qh5_pipeline_layout = [&](VkDescriptorSetLayout set_layout) {
    VkPushConstantRange decode_push{};
    decode_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    decode_push.offset = 0;
    decode_push.size = 4U * sizeof(std::uint32_t);
    auto decode_layout_info = vulkan_structure<VkPipelineLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    decode_layout_info.setLayoutCount = 1;
    decode_layout_info.pSetLayouts = &set_layout;
    decode_layout_info.pushConstantRangeCount = 1;
    decode_layout_info.pPushConstantRanges = &decode_push;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(device, &decode_layout_info, nullptr, &layout),
          "vkCreatePipelineLayout(QH5)");
    return layout;
  };
  result->qh5_lz4_descriptor_layout = make_storage_layout(7U);
  result->qh5_lz4_pipeline_layout =
      make_qh5_pipeline_layout(result->qh5_lz4_descriptor_layout);
  result->qh5_unshuffle_descriptor_layout = make_storage_layout(2U);
  result->qh5_unshuffle_pipeline_layout =
      make_qh5_pipeline_layout(result->qh5_unshuffle_descriptor_layout);
  result->packed_lz4_descriptor_layout = make_storage_layout(4U);
  {
    VkPushConstantRange packed_push{};
    packed_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    packed_push.offset = 0;
    packed_push.size = 3U * sizeof(std::uint32_t);
    auto packed_layout_info = vulkan_structure<VkPipelineLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    packed_layout_info.setLayoutCount = 1;
    packed_layout_info.pSetLayouts = &result->packed_lz4_descriptor_layout;
    packed_layout_info.pushConstantRangeCount = 1;
    packed_layout_info.pPushConstantRanges = &packed_push;
    check(vkCreatePipelineLayout(device, &packed_layout_info, nullptr,
                                 &result->packed_lz4_pipeline_layout),
          "vkCreatePipelineLayout(packed LZ4)");
  }
  result->qh5_lz4_module = make_shader(
      device, kQh5Lz4DecodeSpirv,
      sizeof(kQh5Lz4DecodeSpirv) / sizeof(kQh5Lz4DecodeSpirv[0]));
  result->qh5_unshuffle_module = make_shader(
      device, kQh5BitunshuffleUint16Spirv,
      sizeof(kQh5BitunshuffleUint16Spirv) /
          sizeof(kQh5BitunshuffleUint16Spirv[0]));
  result->packed_lz4_module = make_shader(
      device, kPackedLz4DecodeSpirv,
      sizeof(kPackedLz4DecodeSpirv) / sizeof(kPackedLz4DecodeSpirv[0]));
  result->qh5_lz4_pipeline = make_pipeline(
      device, result->qh5_lz4_pipeline_layout, result->qh5_lz4_module);
  result->qh5_unshuffle_pipeline = make_pipeline(
      device, result->qh5_unshuffle_pipeline_layout,
      result->qh5_unshuffle_module);
  result->packed_lz4_pipeline = make_pipeline(
      device, result->packed_lz4_pipeline_layout,
      result->packed_lz4_module);
  return result;
}

struct ExpectedByShift {
  std::array<std::uint32_t, 6> products{};
};

std::vector<std::uint8_t>
pack_bytes(const std::vector<std::uint8_t> &unpacked) {
  std::vector<std::uint8_t> packed((unpacked.size() + 3U) & ~std::size_t{3}, 0);
  std::copy(unpacked.begin(), unpacked.end(), packed.begin());
  return packed;
}

struct SyntheticReference {
  std::vector<std::uint8_t> frames_by_shift;
  std::array<ExpectedByShift, 256> expected_by_shift{};

  SyntheticReference(const Shape4D shape,
                     const std::vector<std::uint8_t> &membership) {
    const std::uint64_t detector_pixels64 = shape.detector_pixel_count();
    const auto detector_pixels = static_cast<std::size_t>(detector_pixels64);
    frames_by_shift.resize(256U * detector_pixels);
    for (std::uint32_t shift = 0; shift < 256; ++shift) {
      auto &expected = expected_by_shift[shift].products;
      for (std::size_t pixel = 0; pixel < detector_pixels; ++pixel) {
        const std::uint32_t value =
            static_cast<std::uint32_t>((shift + pixel) & 255U);
        frames_by_shift[static_cast<std::size_t>(shift) * detector_pixels +
                        pixel] = static_cast<std::uint8_t>(value);
        expected[0] += value;
        if ((membership[pixel] & 1U) != 0)
          expected[1] += value;
        if ((membership[pixel] & 2U) != 0)
          expected[2] += value;
        if ((membership[pixel] & 4U) != 0)
          expected[3] += value;
        expected[4] +=
            value * static_cast<std::uint32_t>(pixel / shape.detector_columns);
        expected[5] +=
            value * static_cast<std::uint32_t>(pixel % shape.detector_columns);
      }
    }
  }
};

struct OwnedPreparedSegment {
  int file_descriptor = -1;
  std::uint64_t file_offset_bytes = 0;
  std::uint64_t length_bytes = 0;

  OwnedPreparedSegment() = default;
  OwnedPreparedSegment(const OwnedPreparedSegment &) = delete;
  OwnedPreparedSegment &operator=(const OwnedPreparedSegment &) = delete;
  OwnedPreparedSegment(OwnedPreparedSegment &&other) noexcept
      : file_descriptor(other.file_descriptor),
        file_offset_bytes(other.file_offset_bytes),
        length_bytes(other.length_bytes) {
    other.file_descriptor = -1;
  }
  ~OwnedPreparedSegment() {
    if (file_descriptor >= 0)
      close(file_descriptor);
  }
};

class PreparedSourceReader {
public:
  PreparedSourceReader(const std::vector<PreparedSourceSegment> &source,
                       const std::uint64_t required_bytes) {
    if (source.empty()) {
      throw std::invalid_argument(
          "prepared source requires at least one file segment");
    }
    std::uint64_t total = 0;
    segments_.reserve(source.size());
    for (const auto &input : source) {
      if (input.borrowed_file_descriptor < 0 || input.length_bytes == 0) {
        throw std::invalid_argument("prepared source file descriptors and "
                                    "segment lengths must be valid");
      }
      struct stat status{};
      if (fstat(input.borrowed_file_descriptor, &status) != 0 ||
          status.st_size < 0) {
        throw std::invalid_argument(
            "prepared source file descriptor cannot be inspected");
      }
      if (input.file_offset_bytes >
              static_cast<std::uint64_t>(status.st_size) ||
          input.length_bytes > static_cast<std::uint64_t>(status.st_size) -
                                   input.file_offset_bytes) {
        throw std::invalid_argument(
            "prepared source segment extends past its file");
      }
      const int duplicate = dup(input.borrowed_file_descriptor);
      if (duplicate < 0) {
        throw SourceIoError(
            "failed to duplicate a prepared source file descriptor");
      }
      if (input.length_bytes >
          std::numeric_limits<std::uint64_t>::max() - total) {
        close(duplicate);
        throw std::invalid_argument(
            "prepared source segment lengths exceed uint64 range");
      }
      OwnedPreparedSegment segment;
      segment.file_descriptor = duplicate;
      segment.file_offset_bytes = input.file_offset_bytes;
      segment.length_bytes = input.length_bytes;
      segments_.push_back(std::move(segment));
      total += input.length_bytes;
    }
    if (total != required_bytes) {
      throw std::invalid_argument("ordered prepared source segment bytes do "
                                  "not match the exact source geometry");
    }
  }

  void read(const std::uint64_t logical_offset, const std::uint64_t bytes,
            std::uint8_t *destination) const {
    std::uint64_t source_position = 0;
    std::uint64_t requested_position = logical_offset;
    std::uint64_t remaining = bytes;
    for (const auto &segment : segments_) {
      const std::uint64_t segment_end = source_position + segment.length_bytes;
      if (requested_position >= segment_end) {
        source_position = segment_end;
        continue;
      }
      const std::uint64_t within = requested_position - source_position;
      const std::uint64_t take =
          std::min(remaining, segment.length_bytes - within);
      std::uint64_t completed = 0;
      while (completed < take) {
        const std::uint64_t request = std::min<std::uint64_t>(
            take - completed,
            static_cast<std::uint64_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t result = pread(
            segment.file_descriptor, destination + completed,
            static_cast<std::size_t>(request),
            static_cast<off_t>(segment.file_offset_bytes + within + completed));
        if (result < 0 && errno == EINTR)
          continue;
        if (result <= 0) {
          throw SourceIoError(
              "prepared source range read ended before completion");
        }
        completed += static_cast<std::uint64_t>(result);
      }
      destination += take;
      remaining -= take;
      requested_position += take;
      source_position = segment_end;
      if (remaining == 0)
        return;
    }
    throw SourceIoError(
        "prepared source range is outside the ordered segments");
  }

private:
  std::vector<OwnedPreparedSegment> segments_;
};

struct RunResources {
  VkDevice device = VK_NULL_HANDLE;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkQueryPool query_pool = VK_NULL_HANDLE;
  std::vector<VkFence> fences;
  bool queue_work_may_be_active = false;

  explicit RunResources(VkDevice target_device) : device(target_device) {}
  RunResources(const RunResources &) = delete;
  RunResources &operator=(const RunResources &) = delete;
  ~RunResources() {
    if (queue_work_may_be_active)
      vkDeviceWaitIdle(device);
    for (const VkFence fence : fences) {
      if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, fence, nullptr);
      }
    }
    if (query_pool != VK_NULL_HANDLE)
      vkDestroyQueryPool(device, query_pool, nullptr);
    if (descriptor_pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    }
    if (command_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(device, command_pool, nullptr);
  }
};

class Executor final : public ExactProductExecutor {
public:
  Executor()
      : context_(std::make_unique<VulkanContext>()),
        capabilities_(query_capabilities(*context_)),
        pipelines_(make_pipelines(context_->device)) {}

  const VulkanCapabilities &capabilities() const override {
    return capabilities_;
  }

  BenchmarkMetrics
  run_synthetic(const BenchmarkOptions &options,
                const std::vector<std::uint8_t> &detector_band_membership,
                ExactProducts *products) override {
    SyntheticReference reference(options.source_shape,
                                 detector_band_membership);
    return run(options, SourceDType::uint8, detector_band_membership, products,
               &reference, nullptr, nullptr, nullptr);
  }

  BenchmarkMetrics run_prepared_contiguous_u8(
      const BenchmarkOptions &options,
      const std::vector<PreparedSourceSegment> &ordered_segments,
      const std::vector<std::uint8_t> &detector_band_membership,
      ExactProducts *products) override {
    PreparedSourceReader reader(ordered_segments,
                                options.source_shape.value_count());
    return run(options, SourceDType::uint8, detector_band_membership, products,
               nullptr, &reader, nullptr, nullptr);
  }

  BenchmarkMetrics run_prepared_contiguous_u16(
      const BenchmarkOptions &options,
      const std::vector<PreparedSourceSegment> &ordered_segments,
      const std::vector<std::uint8_t> &detector_band_membership,
      ExactProducts *products) override {
    PreparedSourceReader reader(
        ordered_segments,
        checked_product(options.source_shape.value_count(),
                        sizeof(std::uint16_t), "prepared uint16 source bytes"));
    return run(options, SourceDType::uint16, detector_band_membership, products,
               nullptr, &reader, nullptr, nullptr);
  }

  BenchmarkMetrics
  run_indexed_qh5_u16(const BenchmarkOptions &options,
                      const Qh5IndexedSource &source,
                      const std::vector<std::uint8_t> &detector_band_membership,
                      ExactProducts *products) override {
    return run(options, SourceDType::uint16, detector_band_membership, products,
               nullptr, nullptr, &source, nullptr);
  }

  BenchmarkMetrics run_indexed_qh5_audited_low8(
      const BenchmarkOptions &options, const Qh5IndexedSource &source,
      const std::vector<std::uint8_t> &detector_band_membership,
      const std::vector<std::uint8_t> &excluded_detector_pixels,
      ExactProducts *products) override {
    return run(options, SourceDType::uint8, detector_band_membership, products,
               nullptr, nullptr, &source, &excluded_detector_pixels);
  }

  Qh5SelectedFrameMetrics read_indexed_qh5_frame_u16(
      const Qh5IndexedSource &source, const std::uint64_t frame,
      std::uint16_t *destination,
      const std::size_t destination_value_capacity) override {
    constexpr std::uint32_t block_elements = 4096U;
    if (destination == nullptr || frame >= source.frame_count()) {
      throw std::invalid_argument(
          "selected QH5 GPU decode requires an in-range frame and destination");
    }
    const std::uint64_t required_values = checked_product(
        source.blocks_per_frame(), block_elements,
        "selected QH5 detector value count");
    if (required_values > destination_value_capacity ||
        required_values > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "selected QH5 GPU destination is smaller than one frame");
    }

    const auto total_started = Clock::now();
    Qh5SelectedFrameMetrics metrics;
    const auto staging_started = Clock::now();
    const Qh5CompressedBatch batch = source.read_compressed_frames(frame, 1U);
    metrics.storage_read_milliseconds = batch.metrics.storage_read_milliseconds;
    metrics.source_staging_milliseconds =
        milliseconds(Clock::now() - staging_started) -
        metrics.storage_read_milliseconds;
    metrics.source_bytes_read = batch.metrics.source_bytes_read;
    metrics.source_frame_count = batch.metrics.source_frame_count;
    metrics.source_block_count = batch.metrics.source_block_count;
    if (batch.metrics.source_block_count == 0U ||
        batch.metrics.source_block_count >
            std::numeric_limits<std::uint32_t>::max() ||
        batch.compressed_bytes.empty()) {
      throw std::invalid_argument(
          "selected QH5 compressed frame exceeds Vulkan indexing");
    }
    const auto block_count =
        static_cast<std::uint32_t>(batch.metrics.source_block_count);
    const VkDeviceSize metadata_bytes =
        batch.block_metadata.size() * sizeof(std::uint32_t);
    const VkDeviceSize decoded_bytes =
        required_values * sizeof(std::uint16_t);
    const VkDeviceSize status_bytes =
        static_cast<VkDeviceSize>(block_count) * sizeof(std::uint32_t);
    for (const VkDeviceSize bytes :
         {static_cast<VkDeviceSize>(batch.compressed_bytes.size()),
          metadata_bytes, decoded_bytes, status_bytes}) {
      if (bytes > capabilities_.max_storage_buffer_range_bytes ||
          bytes > capabilities_.max_memory_allocation_bytes) {
        throw std::invalid_argument(
            "selected QH5 GPU buffer exceeds the device allocation or storage range");
      }
    }

    Buffer compressed = context_->make_host_buffer(
        batch.compressed_bytes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer metadata = context_->make_host_buffer(
        metadata_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer decoded = context_->make_host_buffer(
        decoded_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer status = context_->make_host_buffer(
        status_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer unused = context_->make_host_buffer(
        sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    metrics.vulkan_committed_bytes =
        compressed.committed_bytes + metadata.committed_bytes +
        decoded.committed_bytes + status.committed_bytes +
        unused.committed_bytes;

    std::memcpy(compressed.mapped, batch.compressed_bytes.data(),
                batch.compressed_bytes.size());
    std::memcpy(metadata.mapped, batch.block_metadata.data(), metadata_bytes);
    std::memset(status.mapped, 0, static_cast<std::size_t>(status_bytes));
    const auto visibility_started = Clock::now();
    context_->flush(compressed, compressed.requested_bytes);
    context_->flush(metadata, metadata.requested_bytes);
    context_->flush(status, status.requested_bytes);
    metrics.vulkan_visibility_milliseconds =
        milliseconds(Clock::now() - visibility_started);

    RunResources run(context_->device);
    auto pool_info = vulkan_structure<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.queueFamilyIndex = context_->queue_family;
    check(vkCreateCommandPool(context_->device, &pool_info, nullptr,
                              &run.command_pool),
          "vkCreateCommandPool(selected QH5)");
    VkCommandBuffer command = VK_NULL_HANDLE;
    auto command_info = vulkan_structure<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = run.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1U;
    check(vkAllocateCommandBuffers(context_->device, &command_info, &command),
          "vkAllocateCommandBuffers(selected QH5)");
    VkDescriptorPoolSize descriptor_pool_size{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7U};
    auto descriptor_pool_info = vulkan_structure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptor_pool_info.maxSets = 1U;
    descriptor_pool_info.poolSizeCount = 1U;
    descriptor_pool_info.pPoolSizes = &descriptor_pool_size;
    check(vkCreateDescriptorPool(context_->device, &descriptor_pool_info,
                                 nullptr, &run.descriptor_pool),
          "vkCreateDescriptorPool(selected QH5)");
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    auto descriptor_set_info =
        vulkan_structure<VkDescriptorSetAllocateInfo>(
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    descriptor_set_info.descriptorPool = run.descriptor_pool;
    descriptor_set_info.descriptorSetCount = 1U;
    descriptor_set_info.pSetLayouts = &pipelines_->qh5_lz4_descriptor_layout;
    check(vkAllocateDescriptorSets(context_->device, &descriptor_set_info,
                                   &descriptor_set),
          "vkAllocateDescriptorSets(selected QH5)");
    std::array<VkDescriptorBufferInfo, 7> buffers{{
        {compressed.buffer, 0, compressed.requested_bytes},
        {metadata.buffer, 0, metadata.requested_bytes},
        {decoded.buffer, 0, decoded.requested_bytes},
        {status.buffer, 0, status.requested_bytes},
        {unused.buffer, 0, unused.requested_bytes},
        {unused.buffer, 0, unused.requested_bytes},
        {unused.buffer, 0, unused.requested_bytes},
    }};
    std::array<VkWriteDescriptorSet, 7> writes{};
    for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
      writes[binding] = vulkan_structure<VkWriteDescriptorSet>(
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
      writes[binding].dstSet = descriptor_set;
      writes[binding].dstBinding = binding;
      writes[binding].descriptorCount = 1U;
      writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[binding].pBufferInfo = &buffers[binding];
    }
    vkUpdateDescriptorSets(context_->device,
                           static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    const bool timestamps = context_->timestamp_valid_bits != 0U;
    if (timestamps) {
      auto query_info = vulkan_structure<VkQueryPoolCreateInfo>(
          VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
      query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_info.queryCount = 2U;
      check(vkCreateQueryPool(context_->device, &query_info, nullptr,
                              &run.query_pool),
            "vkCreateQueryPool(selected QH5)");
    }
    auto fence_info = vulkan_structure<VkFenceCreateInfo>(
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    run.fences.resize(1U, VK_NULL_HANDLE);
    check(vkCreateFence(context_->device, &fence_info, nullptr,
                        &run.fences.front()),
          "vkCreateFence(selected QH5)");

    auto begin = vulkan_structure<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin),
          "vkBeginCommandBuffer(selected QH5)");
    if (timestamps) {
      vkCmdResetQueryPool(command, run.query_pool, 0U, 2U);
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          run.query_pool, 0U);
    }
    const std::array<std::uint32_t, 4> parameters{
        block_count, batch.compressed_byte_count, 8192U, 0U};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_->qh5_lz4_pipeline_layout, 0U, 1U,
                            &descriptor_set, 0U, nullptr);
    vkCmdPushConstants(command, pipelines_->qh5_lz4_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(parameters),
                       parameters.data());
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelines_->qh5_lz4_pipeline);
    vkCmdDispatch(command, block_count, 1U, 1U);
    if (timestamps) {
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          run.query_pool, 1U);
    }
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    const std::array<VkBuffer, 2> barrier_buffers{status.buffer,
                                                  decoded.buffer};
    const std::array<VkDeviceSize, 2> barrier_sizes{status.requested_bytes,
                                                    decoded.requested_bytes};
    for (std::size_t index = 0; index < barriers.size(); ++index) {
      barriers[index] = vulkan_structure<VkBufferMemoryBarrier>(
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
      barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barriers[index].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].buffer = barrier_buffers[index];
      barriers[index].size = barrier_sizes[index];
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0U, 0U, nullptr,
                         static_cast<std::uint32_t>(barriers.size()),
                         barriers.data(), 0U, nullptr);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer(selected QH5)");
    auto submit = vulkan_structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command;
    const auto decode_started = Clock::now();
    check(vkQueueSubmit(context_->queue, 1U, &submit, run.fences.front()),
          "vkQueueSubmit(selected QH5)");
    metrics.queue_submit_count = 1U;
    run.queue_work_may_be_active = true;
    check(vkWaitForFences(context_->device, 1U, &run.fences.front(), VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "vkWaitForFences(selected QH5)");
    run.queue_work_may_be_active = false;
    metrics.gpu_decode_milliseconds =
        milliseconds(Clock::now() - decode_started);
    if (timestamps) {
      std::array<std::uint64_t, 2> timestamp_values{};
      check(vkGetQueryPoolResults(
                context_->device, run.query_pool, 0U, 2U,
                sizeof(timestamp_values), timestamp_values.data(),
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults(selected QH5)");
      metrics.gpu_decode_milliseconds =
          static_cast<double>(timestamp_values[1] - timestamp_values[0]) *
          capabilities_.timestamp_period_nanoseconds / 1.0e6;
    }
    context_->invalidate(status);
    context_->invalidate(decoded);
    const auto *status_values =
        static_cast<const std::uint32_t *>(status.mapped);
    for (std::uint32_t block = 0; block < block_count; ++block) {
      if (status_values[block] != 0U) {
        throw std::runtime_error(
            "selected QH5 GPU decode rejected block " +
            std::to_string(block) + " with status " +
            std::to_string(status_values[block]));
      }
    }
    std::memcpy(destination, decoded.mapped,
                static_cast<std::size_t>(decoded_bytes));
    metrics.total_milliseconds = milliseconds(Clock::now() - total_started);
    return metrics;
  }

  Qh5PackedShardMetrics pack_indexed_qh5_audited_low8_shard(
      const Qh5IndexedSource &source, const std::uint64_t first_frame,
      const std::uint32_t frame_count,
      const std::vector<std::uint32_t> &packed_descriptors,
      const std::uint32_t packed_payload_word_count,
      const std::vector<std::uint8_t> &excluded_detector_pixels,
      std::vector<std::uint32_t> *packed_payload) override {
    constexpr std::uint32_t detector_pixels = 192U * 192U;
    constexpr std::uint32_t scan_tile = 128U;
    if (frame_count == 0U || frame_count % scan_tile != 0U ||
        first_frame > source.frame_count() ||
        frame_count > source.frame_count() - first_frame) {
      throw std::invalid_argument(
          "packed QH5 shard must be a nonempty, in-range multiple of 128 frames");
    }
    const std::uint32_t tile_count = frame_count / scan_tile;
    if (tile_count > 255U || source.blocks_per_frame() > 255U ||
        packed_descriptors.size() !=
            static_cast<std::size_t>(detector_pixels) * tile_count ||
        excluded_detector_pixels.size() != detector_pixels ||
        std::any_of(excluded_detector_pixels.begin(),
                    excluded_detector_pixels.end(),
                    [](const std::uint8_t value) { return value > 1U; })) {
      throw std::invalid_argument(
          "packed QH5 descriptors, tiles, blocks, or exclusions are invalid");
    }
    if (packed_payload == nullptr || packed_payload_word_count == 0U) {
      throw std::invalid_argument(
          "packed QH5 output requires a nonempty payload destination");
    }

    const auto started = Clock::now();
    Qh5PackedShardMetrics metrics;
    const auto read_started = Clock::now();
    const Qh5CompressedBatch batch =
        source.read_compressed_frames(first_frame, frame_count);
    metrics.storage_read_milliseconds = batch.metrics.storage_read_milliseconds;
    metrics.source_staging_milliseconds =
        milliseconds(Clock::now() - read_started) -
        metrics.storage_read_milliseconds;
    metrics.source_bytes_read = batch.metrics.source_bytes_read;
    metrics.compressed_bytes_staged = batch.compressed_byte_count;
    if (batch.metrics.source_block_count >
            std::numeric_limits<std::uint32_t>::max() ||
        batch.compressed_bytes.empty()) {
      throw std::invalid_argument("packed QH5 batch exceeds Vulkan indexing");
    }
    const auto block_count =
        static_cast<std::uint32_t>(batch.metrics.source_block_count);
    const VkDeviceSize metadata_bytes =
        batch.block_metadata.size() * sizeof(std::uint32_t);
    const VkDeviceSize status_bytes =
        static_cast<VkDeviceSize>(block_count) * sizeof(std::uint32_t);
    const VkDeviceSize descriptor_bytes =
        packed_descriptors.size() * sizeof(std::uint32_t);
    const VkDeviceSize payload_bytes =
        static_cast<VkDeviceSize>(packed_payload_word_count) *
        sizeof(std::uint32_t);
    for (const VkDeviceSize bytes :
         {static_cast<VkDeviceSize>(batch.compressed_bytes.size()),
          metadata_bytes, status_bytes, descriptor_bytes, payload_bytes}) {
      if (bytes > capabilities_.max_storage_buffer_range_bytes ||
          bytes > capabilities_.max_memory_allocation_bytes) {
        throw std::invalid_argument(
            "packed QH5 buffer exceeds the device allocation or storage range");
      }
    }

    Buffer compressed = context_->make_host_buffer(
        batch.compressed_bytes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer metadata = context_->make_host_buffer(
        metadata_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer decoded_dummy = context_->make_host_buffer(
        sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer status = context_->make_host_buffer(
        status_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer exclusions = context_->make_host_buffer(
        static_cast<VkDeviceSize>(detector_pixels) * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer descriptors = context_->make_host_buffer(
        descriptor_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer payload = context_->make_host_buffer(
        payload_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    metrics.vulkan_committed_bytes =
        compressed.committed_bytes + metadata.committed_bytes +
        decoded_dummy.committed_bytes + status.committed_bytes +
        exclusions.committed_bytes +
        descriptors.committed_bytes + payload.committed_bytes;

    std::memcpy(compressed.mapped, batch.compressed_bytes.data(),
                batch.compressed_bytes.size());
    std::memcpy(metadata.mapped, batch.block_metadata.data(), metadata_bytes);
    std::memset(status.mapped, 0, static_cast<std::size_t>(status_bytes));
    auto *exclusion_words = static_cast<std::uint32_t *>(exclusions.mapped);
    std::transform(excluded_detector_pixels.begin(),
                   excluded_detector_pixels.end(), exclusion_words,
                   [](const std::uint8_t value) {
                     return static_cast<std::uint32_t>(value);
                   });
    std::memcpy(descriptors.mapped, packed_descriptors.data(),
                descriptor_bytes);
    std::memset(payload.mapped, 0, static_cast<std::size_t>(payload_bytes));
    const auto visibility_started = Clock::now();
    context_->flush(compressed, compressed.requested_bytes);
    context_->flush(metadata, metadata.requested_bytes);
    context_->flush(status, status.requested_bytes);
    context_->flush(exclusions, exclusions.requested_bytes);
    context_->flush(descriptors, descriptors.requested_bytes);
    context_->flush(payload, payload.requested_bytes);
    metrics.vulkan_visibility_milliseconds =
        milliseconds(Clock::now() - visibility_started);

    RunResources run(context_->device);
    auto pool_info = vulkan_structure<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.queueFamilyIndex = context_->queue_family;
    check(vkCreateCommandPool(context_->device, &pool_info, nullptr,
                              &run.command_pool),
          "vkCreateCommandPool(packed QH5)");
    VkCommandBuffer command = VK_NULL_HANDLE;
    auto command_info = vulkan_structure<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = run.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1U;
    check(vkAllocateCommandBuffers(context_->device, &command_info, &command),
          "vkAllocateCommandBuffers(packed QH5)");
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7U};
    auto descriptor_pool_info = vulkan_structure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptor_pool_info.maxSets = 1U;
    descriptor_pool_info.poolSizeCount = 1U;
    descriptor_pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(context_->device, &descriptor_pool_info,
                                 nullptr, &run.descriptor_pool),
          "vkCreateDescriptorPool(packed QH5)");
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    auto set_info = vulkan_structure<VkDescriptorSetAllocateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    set_info.descriptorPool = run.descriptor_pool;
    set_info.descriptorSetCount = 1U;
    set_info.pSetLayouts = &pipelines_->qh5_lz4_descriptor_layout;
    check(vkAllocateDescriptorSets(context_->device, &set_info,
                                   &descriptor_set),
          "vkAllocateDescriptorSets(packed QH5)");
    std::array<VkDescriptorBufferInfo, 7> buffers{{
        {compressed.buffer, 0, compressed.requested_bytes},
        {metadata.buffer, 0, metadata.requested_bytes},
        {decoded_dummy.buffer, 0, decoded_dummy.requested_bytes},
        {status.buffer, 0, status.requested_bytes},
        {exclusions.buffer, 0, exclusions.requested_bytes},
        {descriptors.buffer, 0, descriptors.requested_bytes},
        {payload.buffer, 0, payload.requested_bytes},
    }};
    std::array<VkWriteDescriptorSet, 7> writes{};
    for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
      writes[binding] = vulkan_structure<VkWriteDescriptorSet>(
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
      writes[binding].dstSet = descriptor_set;
      writes[binding].dstBinding = binding;
      writes[binding].descriptorCount = 1U;
      writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[binding].pBufferInfo = &buffers[binding];
    }
    vkUpdateDescriptorSets(context_->device,
                           static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    const bool timestamps = context_->timestamp_valid_bits != 0U;
    if (timestamps) {
      auto query_info = vulkan_structure<VkQueryPoolCreateInfo>(
          VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
      query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_info.queryCount = 2U;
      check(vkCreateQueryPool(context_->device, &query_info, nullptr,
                              &run.query_pool),
            "vkCreateQueryPool(packed QH5)");
    }
    auto fence_info = vulkan_structure<VkFenceCreateInfo>(
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    run.fences.resize(1U, VK_NULL_HANDLE);
    check(vkCreateFence(context_->device, &fence_info, nullptr,
                        &run.fences.front()),
          "vkCreateFence(packed QH5)");

    auto begin = vulkan_structure<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin),
          "vkBeginCommandBuffer(packed QH5)");
    if (timestamps) {
      vkCmdResetQueryPool(command, run.query_pool, 0U, 2U);
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          run.query_pool, 0U);
    }
    const std::array<std::uint32_t, 4> parameters{
        block_count, batch.compressed_byte_count, 8192U,
        0xc0000000U | (tile_count << 8U) | source.blocks_per_frame()};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_->qh5_lz4_pipeline_layout, 0, 1,
                            &descriptor_set, 0, nullptr);
    vkCmdPushConstants(command, pipelines_->qh5_lz4_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters),
                       parameters.data());
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelines_->qh5_lz4_pipeline);
    vkCmdDispatch(command, block_count, 1U, 1U);
    if (timestamps) {
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          run.query_pool, 1U);
    }
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    const std::array<VkBuffer, 2> barrier_buffers{status.buffer,
                                                  payload.buffer};
    const std::array<VkDeviceSize, 2> barrier_sizes{status.requested_bytes,
                                                    payload.requested_bytes};
    for (std::size_t index = 0; index < barriers.size(); ++index) {
      barriers[index] = vulkan_structure<VkBufferMemoryBarrier>(
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
      barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barriers[index].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].buffer = barrier_buffers[index];
      barriers[index].size = barrier_sizes[index];
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(barriers.size()),
                         barriers.data(), 0, nullptr);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer(packed QH5)");
    auto submit = vulkan_structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(context_->queue, 1, &submit, run.fences.front()),
          "vkQueueSubmit(packed QH5)");
    run.queue_work_may_be_active = true;
    check(vkWaitForFences(context_->device, 1, &run.fences.front(), VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "vkWaitForFences(packed QH5)");
    run.queue_work_may_be_active = false;
    if (timestamps) {
      std::array<std::uint64_t, 2> values{};
      check(vkGetQueryPoolResults(
                context_->device, run.query_pool, 0U, 2U, sizeof(values),
                values.data(), sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults(packed QH5)");
      const double period =
          capabilities_.timestamp_period_nanoseconds / 1.0e6;
      metrics.gpu_decode_and_pack_milliseconds =
          static_cast<double>(values[1] - values[0]) * period;
      metrics.gpu_decode_milliseconds =
          metrics.gpu_decode_and_pack_milliseconds;
    }
    context_->invalidate(status);
    context_->invalidate(payload);
    const auto *status_values = static_cast<const std::uint32_t *>(status.mapped);
    for (std::uint32_t block = 0; block < block_count; ++block) {
      if (status_values[block] != 0U) {
        throw std::runtime_error(
            "QH5 GPU direct pack rejected block " + std::to_string(block) +
            " with status " + std::to_string(status_values[block]));
      }
    }
    const auto *payload_words =
        static_cast<const std::uint32_t *>(payload.mapped);
    packed_payload->assign(payload_words,
                           payload_words + packed_payload_word_count);
    metrics.ready_milliseconds = milliseconds(Clock::now() - started);
    return metrics;
  }

  PackedLz4Metrics decode_packed_lz4(
      const std::vector<std::uint8_t> &compressed_input,
      const std::vector<std::uint32_t> &chunk_metadata,
      const std::uint32_t decoded_word_count,
      std::vector<std::uint32_t> *decoded_output) override {
    if (compressed_input.empty() || chunk_metadata.empty() ||
        chunk_metadata.size() % 4U != 0U || decoded_word_count == 0U ||
        decoded_output == nullptr) {
      throw std::invalid_argument(
          "packed LZ4 decode requires compressed bytes, chunk records, and output");
    }
    const std::uint32_t chunk_count =
        static_cast<std::uint32_t>(chunk_metadata.size() / 4U);
    std::uint64_t expected_output_bytes = 0U;
    for (std::uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
      const std::uint32_t input_offset = chunk_metadata[chunk * 4U];
      const std::uint32_t input_bytes = chunk_metadata[chunk * 4U + 1U];
      const std::uint32_t output_word = chunk_metadata[chunk * 4U + 2U];
      const std::uint32_t output_bytes = chunk_metadata[chunk * 4U + 3U];
      if (input_bytes == 0U || input_offset > compressed_input.size() ||
          input_bytes > compressed_input.size() - input_offset ||
          output_bytes == 0U || (output_bytes & 3U) != 0U ||
          static_cast<std::uint64_t>(output_word) * 4U !=
              expected_output_bytes) {
        throw std::invalid_argument(
            "packed LZ4 metadata is noncanonical or outside its buffers");
      }
      expected_output_bytes += output_bytes;
    }
    if (expected_output_bytes !=
        static_cast<std::uint64_t>(decoded_word_count) * 4U) {
      throw std::invalid_argument(
          "packed LZ4 chunks do not cover the decoded payload exactly");
    }
    if (compressed_input.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("packed LZ4 source exceeds Vulkan indexing");
    }
    const auto started = Clock::now();
    PackedLz4Metrics metrics;
    std::vector<std::uint8_t> compressed = compressed_input;
    compressed.resize((compressed.size() + 3U) & ~std::size_t{3}, 0U);
    const VkDeviceSize metadata_bytes =
        chunk_metadata.size() * sizeof(std::uint32_t);
    const VkDeviceSize decoded_bytes =
        static_cast<VkDeviceSize>(decoded_word_count) * sizeof(std::uint32_t);
    const VkDeviceSize status_bytes =
        static_cast<VkDeviceSize>(chunk_count) * sizeof(std::uint32_t);
    for (const VkDeviceSize bytes :
         {static_cast<VkDeviceSize>(compressed.size()), metadata_bytes,
          decoded_bytes, status_bytes}) {
      if (bytes > capabilities_.max_storage_buffer_range_bytes ||
          bytes > capabilities_.max_memory_allocation_bytes) {
        throw std::invalid_argument(
            "packed LZ4 buffer exceeds the device allocation or storage range");
      }
    }
    Buffer compressed_buffer = context_->make_host_buffer(
        compressed.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer metadata_buffer = context_->make_host_buffer(
        metadata_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer decoded_buffer = context_->make_host_buffer(
        decoded_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer status_buffer = context_->make_host_buffer(
        status_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    metrics.compressed_bytes_staged = compressed_input.size();
    metrics.decoded_bytes = decoded_bytes;
    metrics.vulkan_committed_bytes =
        compressed_buffer.committed_bytes + metadata_buffer.committed_bytes +
        decoded_buffer.committed_bytes + status_buffer.committed_bytes;
    std::memcpy(compressed_buffer.mapped, compressed.data(), compressed.size());
    std::memcpy(metadata_buffer.mapped, chunk_metadata.data(), metadata_bytes);
    std::memset(decoded_buffer.mapped, 0,
                static_cast<std::size_t>(decoded_bytes));
    std::memset(status_buffer.mapped, 0,
                static_cast<std::size_t>(status_bytes));
    const auto visibility_started = Clock::now();
    context_->flush(compressed_buffer, compressed_buffer.requested_bytes);
    context_->flush(metadata_buffer, metadata_buffer.requested_bytes);
    context_->flush(decoded_buffer, decoded_buffer.requested_bytes);
    context_->flush(status_buffer, status_buffer.requested_bytes);
    metrics.vulkan_visibility_milliseconds =
        milliseconds(Clock::now() - visibility_started);

    RunResources run(context_->device);
    auto pool_info = vulkan_structure<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.queueFamilyIndex = context_->queue_family;
    check(vkCreateCommandPool(context_->device, &pool_info, nullptr,
                              &run.command_pool),
          "vkCreateCommandPool(packed LZ4)");
    VkCommandBuffer command = VK_NULL_HANDLE;
    auto command_info = vulkan_structure<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = run.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1U;
    check(vkAllocateCommandBuffers(context_->device, &command_info, &command),
          "vkAllocateCommandBuffers(packed LZ4)");
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4U};
    auto descriptor_pool_info = vulkan_structure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptor_pool_info.maxSets = 1U;
    descriptor_pool_info.poolSizeCount = 1U;
    descriptor_pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(context_->device, &descriptor_pool_info,
                                 nullptr, &run.descriptor_pool),
          "vkCreateDescriptorPool(packed LZ4)");
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    auto set_info = vulkan_structure<VkDescriptorSetAllocateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    set_info.descriptorPool = run.descriptor_pool;
    set_info.descriptorSetCount = 1U;
    set_info.pSetLayouts = &pipelines_->packed_lz4_descriptor_layout;
    check(vkAllocateDescriptorSets(context_->device, &set_info,
                                   &descriptor_set),
          "vkAllocateDescriptorSets(packed LZ4)");
    std::array<VkDescriptorBufferInfo, 4> buffers{{
        {compressed_buffer.buffer, 0, compressed_buffer.requested_bytes},
        {metadata_buffer.buffer, 0, metadata_buffer.requested_bytes},
        {decoded_buffer.buffer, 0, decoded_buffer.requested_bytes},
        {status_buffer.buffer, 0, status_buffer.requested_bytes},
    }};
    std::array<VkWriteDescriptorSet, 4> writes{};
    for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
      writes[binding] = vulkan_structure<VkWriteDescriptorSet>(
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
      writes[binding].dstSet = descriptor_set;
      writes[binding].dstBinding = binding;
      writes[binding].descriptorCount = 1U;
      writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[binding].pBufferInfo = &buffers[binding];
    }
    vkUpdateDescriptorSets(context_->device,
                           static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
    const bool timestamps = context_->timestamp_valid_bits != 0U;
    if (timestamps) {
      auto query_info = vulkan_structure<VkQueryPoolCreateInfo>(
          VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
      query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_info.queryCount = 2U;
      check(vkCreateQueryPool(context_->device, &query_info, nullptr,
                              &run.query_pool),
            "vkCreateQueryPool(packed LZ4)");
    }
    run.fences.resize(1U, VK_NULL_HANDLE);
    auto fence_info = vulkan_structure<VkFenceCreateInfo>(
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    check(vkCreateFence(context_->device, &fence_info, nullptr,
                        &run.fences.front()),
          "vkCreateFence(packed LZ4)");
    auto begin = vulkan_structure<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin),
          "vkBeginCommandBuffer(packed LZ4)");
    if (timestamps) {
      vkCmdResetQueryPool(command, run.query_pool, 0U, 2U);
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          run.query_pool, 0U);
    }
    constexpr std::uint32_t dispatch_width = 32768U;
    const std::array<std::uint32_t, 3> parameters{
        chunk_count, static_cast<std::uint32_t>(compressed_input.size()),
        dispatch_width};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_->packed_lz4_pipeline_layout, 0, 1,
                            &descriptor_set, 0, nullptr);
    vkCmdPushConstants(command, pipelines_->packed_lz4_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters),
                       parameters.data());
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelines_->packed_lz4_pipeline);
    vkCmdDispatch(command, std::min(chunk_count, dispatch_width),
                  (chunk_count + dispatch_width - 1U) / dispatch_width, 1U);
    if (timestamps) {
      vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          run.query_pool, 1U);
    }
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    const std::array<VkBuffer, 2> barrier_buffers{decoded_buffer.buffer,
                                                  status_buffer.buffer};
    const std::array<VkDeviceSize, 2> barrier_sizes{decoded_bytes,
                                                    status_bytes};
    for (std::size_t index = 0; index < barriers.size(); ++index) {
      barriers[index] = vulkan_structure<VkBufferMemoryBarrier>(
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
      barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barriers[index].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].buffer = barrier_buffers[index];
      barriers[index].size = barrier_sizes[index];
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(barriers.size()),
                         barriers.data(), 0, nullptr);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer(packed LZ4)");
    auto submit = vulkan_structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(context_->queue, 1, &submit, run.fences.front()),
          "vkQueueSubmit(packed LZ4)");
    run.queue_work_may_be_active = true;
    check(vkWaitForFences(context_->device, 1, &run.fences.front(), VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "vkWaitForFences(packed LZ4)");
    run.queue_work_may_be_active = false;
    if (timestamps) {
      std::array<std::uint64_t, 2> values{};
      check(vkGetQueryPoolResults(
                context_->device, run.query_pool, 0U, 2U, sizeof(values),
                values.data(), sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults(packed LZ4)");
      metrics.gpu_decode_milliseconds =
          static_cast<double>(values[1] - values[0]) *
          capabilities_.timestamp_period_nanoseconds / 1.0e6;
    }
    context_->invalidate(status_buffer);
    context_->invalidate(decoded_buffer);
    const auto *statuses =
        static_cast<const std::uint32_t *>(status_buffer.mapped);
    for (std::uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
      if (statuses[chunk] != 0U) {
        throw std::runtime_error(
            "packed LZ4 GPU decode rejected chunk " +
            std::to_string(chunk) + " with status " +
            std::to_string(statuses[chunk]));
      }
    }
    const auto *words =
        static_cast<const std::uint32_t *>(decoded_buffer.mapped);
    decoded_output->assign(words, words + decoded_word_count);
    metrics.ready_milliseconds = milliseconds(Clock::now() - started);
    return metrics;
  }

private:
  BenchmarkMetrics
  run(const BenchmarkOptions &options, const SourceDType source_dtype,
      const std::vector<std::uint8_t> &detector_band_membership,
      ExactProducts *products, const SyntheticReference *reference,
      const PreparedSourceReader *prepared_source,
      const Qh5IndexedSource *indexed_source,
      const std::vector<std::uint8_t> *qh5_exclusions) {
    rusage usage_before{};
    if (getrusage(RUSAGE_SELF, &usage_before) != 0) {
      throw std::runtime_error("getrusage failed before the benchmark");
    }
    const auto setup_started = Clock::now();
    const ScientificRequest request{
        options.source_shape,
        source_dtype,
        1,
        1,
        true,
        true,
        true,
        true,
        true,
        true,
    };
    validate_scientific_request(request);
    const std::uint64_t scan_count64 = options.source_shape.scan_count();
    const std::uint64_t detector_pixels64 =
        options.source_shape.detector_pixel_count();
    if (scan_count64 > std::numeric_limits<std::uint32_t>::max() ||
        detector_pixels64 > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument(
          "synthetic Vulkan geometry exceeds uint32 shader indexing");
    }
    if (detector_band_membership.size() != detector_pixels64) {
      throw std::invalid_argument(
          "detector-band membership must contain one byte per detector pixel");
    }
    if (options.selected_scan_row >= options.source_shape.scan_rows ||
        options.selected_scan_column >= options.source_shape.scan_columns) {
      throw std::invalid_argument(
          "selected diffraction coordinate is outside the scan");
    }
    if (options.shard_scan_rows == 0 ||
        options.shard_scan_rows > options.source_shape.scan_rows) {
      throw std::invalid_argument(
          "shard_scan_rows must be within the source scan");
    }
    if (options.staging_ring_depth == 0 || options.staging_ring_depth > 4) {
      throw std::invalid_argument(
          "staging_ring_depth must be between one and four");
    }
    const std::uint64_t source_bytes_per_value = bytes_per_value(source_dtype);
    const bool gpu_indexed_qh5 = indexed_source != nullptr;
    if (qh5_exclusions != nullptr &&
        (!gpu_indexed_qh5 || source_dtype != SourceDType::uint8 ||
         qh5_exclusions->size() != detector_pixels64 ||
         std::any_of(qh5_exclusions->begin(), qh5_exclusions->end(),
                     [](std::uint8_t value) { return value > 1U; }))) {
      throw std::invalid_argument(
          "audited QH5 low8 exclusions must contain one zero/one byte per "
          "detector pixel");
    }
    if ((checked_product(detector_pixels64, source_bytes_per_value,
                         "selected diffraction bytes") &
         3U) != 0) {
      throw std::invalid_argument(
          "the current exact selected-diffraction copy requires a "
          "four-byte-aligned detector frame");
    }

    const auto scan_count = static_cast<std::uint32_t>(scan_count64);
    const auto detector_pixels = static_cast<std::uint32_t>(detector_pixels64);
    const std::uint64_t scans_per_shard64 =
        checked_product(options.shard_scan_rows,
                        options.source_shape.scan_columns, "scans per shard");
    const std::uint64_t maximum_shard_bytes64 =
        checked_product(checked_product(scans_per_shard64, detector_pixels64,
                                        "maximum shard values"),
                        source_bytes_per_value, "maximum shard bytes");
    if (maximum_shard_bytes64 > capabilities_.max_storage_buffer_range_bytes ||
        maximum_shard_bytes64 > capabilities_.max_memory_allocation_bytes) {
      throw std::invalid_argument(
          "requested shard exceeds the Fold8 Vulkan buffer/allocation limits");
    }
    const VkDeviceSize maximum_shard_bytes = maximum_shard_bytes64;
    const std::uint64_t maximum_qh5_blocks = gpu_indexed_qh5
        ? checked_product(scans_per_shard64, indexed_source->blocks_per_frame(),
                          "maximum QH5 blocks")
        : 0;
    const VkDeviceSize maximum_qh5_compressed_bytes = gpu_indexed_qh5
        ? checked_product(maximum_qh5_blocks, 8192U + 64U,
                          "maximum QH5 compressed staging bytes")
        : 0;
    const VkDeviceSize maximum_qh5_metadata_bytes = gpu_indexed_qh5
        ? checked_product(maximum_qh5_blocks, 2U * sizeof(std::uint32_t),
                          "maximum QH5 metadata bytes")
        : 0;
    const VkDeviceSize maximum_qh5_status_bytes = gpu_indexed_qh5
        ? checked_product(maximum_qh5_blocks, sizeof(std::uint32_t),
                          "maximum QH5 status bytes")
        : 0;
    if (gpu_indexed_qh5 &&
        (maximum_qh5_compressed_bytes >
             capabilities_.max_storage_buffer_range_bytes ||
         maximum_qh5_metadata_bytes >
             capabilities_.max_storage_buffer_range_bytes ||
         maximum_qh5_status_bytes >
             capabilities_.max_storage_buffer_range_bytes)) {
      throw std::invalid_argument(
          "QH5 GPU decode staging exceeds the Fold8 Vulkan buffer limits");
    }
    const std::uint32_t shard_count =
        (options.source_shape.scan_rows + options.shard_scan_rows - 1) /
        options.shard_scan_rows;
    const std::uint64_t product_bytes_per_value =
        source_dtype == SourceDType::uint8 ? sizeof(std::uint32_t)
                                           : sizeof(std::uint64_t);
    const VkDeviceSize scan_product_bytes =
        checked_product(checked_product(scan_count64, 6U, "scan products"),
                        product_bytes_per_value, "scan product bytes");
    const VkDeviceSize diffraction_sum_bytes = checked_product(
        detector_pixels64, product_bytes_per_value, "diffraction sums");
    const VkDeviceSize selected_bytes = checked_product(
        detector_pixels64, source_bytes_per_value, "selected diffraction");
    const std::vector<std::uint8_t> packed_membership =
        pack_bytes(detector_band_membership);

    std::vector<Buffer> source_ring;
    source_ring.reserve(options.staging_ring_depth);
    for (std::uint32_t index = 0; index < options.staging_ring_depth; ++index) {
      source_ring.push_back(context_->make_host_buffer(
          maximum_shard_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }
    std::vector<Buffer> qh5_compressed_ring;
    std::vector<Buffer> qh5_metadata_ring;
    std::vector<Buffer> qh5_bitshuffled_ring;
    std::vector<Buffer> qh5_status_ring;
    if (gpu_indexed_qh5) {
      qh5_compressed_ring.reserve(options.staging_ring_depth);
      qh5_metadata_ring.reserve(options.staging_ring_depth);
      qh5_bitshuffled_ring.reserve(options.staging_ring_depth);
      qh5_status_ring.reserve(options.staging_ring_depth);
      for (std::uint32_t index = 0; index < options.staging_ring_depth;
           ++index) {
        qh5_compressed_ring.push_back(context_->make_host_buffer(
            maximum_qh5_compressed_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        qh5_metadata_ring.push_back(context_->make_host_buffer(
            maximum_qh5_metadata_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        // The retained decoder keeps its 8 KiB block in shared memory and
        // writes the final typed shard directly. Keep only a valid dummy
        // binding for the legacy, undispatched unshuffle descriptor set.
        qh5_bitshuffled_ring.push_back(context_->make_host_buffer(
            4U, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        qh5_status_ring.push_back(context_->make_host_buffer(
            maximum_qh5_status_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false));
      }
    }
    Buffer membership = context_->make_host_buffer(
        packed_membership.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer scan_products = context_->make_host_buffer(
        scan_product_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer diffraction_sums = context_->make_host_buffer(
        diffraction_sum_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer selected = context_->make_host_buffer(
        selected_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    Buffer qh5_exclusion_mask;
    if (gpu_indexed_qh5) {
      std::vector<std::uint32_t> exclusion_values(
          static_cast<std::size_t>(detector_pixels), 0U);
      if (qh5_exclusions != nullptr) {
        std::transform(qh5_exclusions->begin(), qh5_exclusions->end(),
                       exclusion_values.begin(),
                       [](std::uint8_t value) {
                         return static_cast<std::uint32_t>(value);
                       });
      }
      qh5_exclusion_mask = context_->make_host_buffer(
          exclusion_values.size() * sizeof(std::uint32_t),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          false);
      std::memcpy(qh5_exclusion_mask.mapped, exclusion_values.data(),
                  exclusion_values.size() * sizeof(std::uint32_t));
      context_->flush(qh5_exclusion_mask,
                      qh5_exclusion_mask.requested_bytes);
    }
    std::memcpy(membership.mapped, packed_membership.data(),
                packed_membership.size());
    std::memset(scan_products.mapped, 0,
                static_cast<std::size_t>(scan_product_bytes));
    std::memset(diffraction_sums.mapped, 0,
                static_cast<std::size_t>(diffraction_sum_bytes));
    std::memset(selected.mapped, 0, static_cast<std::size_t>(selected_bytes));
    context_->flush(membership, membership.requested_bytes);
    context_->flush(scan_products, scan_products.requested_bytes);
    context_->flush(diffraction_sums, diffraction_sums.requested_bytes);
    context_->flush(selected, selected.requested_bytes);

    RunResources run(context_->device);
    auto pool_info = vulkan_structure<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = context_->queue_family;
    check(vkCreateCommandPool(context_->device, &pool_info, nullptr,
                              &run.command_pool),
          "vkCreateCommandPool");
    std::vector<VkCommandBuffer> commands(options.staging_ring_depth);
    auto command_info = vulkan_structure<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = run.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = options.staging_ring_depth;
    check(vkAllocateCommandBuffers(context_->device, &command_info,
                                   commands.data()),
          "vkAllocateCommandBuffers");

    VkDescriptorPoolSize pool_size{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        (gpu_indexed_qh5 ? 14U : 5U) * options.staging_ring_depth,
    };
    auto descriptor_pool_info = vulkan_structure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptor_pool_info.maxSets =
        (gpu_indexed_qh5 ? 3U : 1U) * options.staging_ring_depth;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(context_->device, &descriptor_pool_info,
                                 nullptr, &run.descriptor_pool),
          "vkCreateDescriptorPool");
    std::vector<VkDescriptorSetLayout> layouts(options.staging_ring_depth,
                                               pipelines_->descriptor_layout);
    std::vector<VkDescriptorSet> descriptor_sets(options.staging_ring_depth);
    auto set_info = vulkan_structure<VkDescriptorSetAllocateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    set_info.descriptorPool = run.descriptor_pool;
    set_info.descriptorSetCount = options.staging_ring_depth;
    set_info.pSetLayouts = layouts.data();
    check(vkAllocateDescriptorSets(context_->device, &set_info,
                                   descriptor_sets.data()),
          "vkAllocateDescriptorSets");
    for (std::uint32_t slot = 0; slot < options.staging_ring_depth; ++slot) {
      std::array<VkDescriptorBufferInfo, 5> buffer_info{{
          {source_ring[slot].buffer, 0, maximum_shard_bytes},
          {membership.buffer, 0, membership.requested_bytes},
          {scan_products.buffer, 0, scan_products.requested_bytes},
          {diffraction_sums.buffer, 0, diffraction_sums.requested_bytes},
          {selected.buffer, 0, selected.requested_bytes},
      }};
      std::array<VkWriteDescriptorSet, 5> writes{};
      for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
        writes[binding] = vulkan_structure<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[binding].dstSet = descriptor_sets[slot];
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &buffer_info[binding];
      }
      vkUpdateDescriptorSets(context_->device,
                             static_cast<std::uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }

    std::vector<VkDescriptorSet> qh5_lz4_descriptor_sets;
    std::vector<VkDescriptorSet> qh5_unshuffle_descriptor_sets;
    if (gpu_indexed_qh5) {
      std::vector<VkDescriptorSetLayout> lz4_layouts(
          options.staging_ring_depth, pipelines_->qh5_lz4_descriptor_layout);
      qh5_lz4_descriptor_sets.resize(options.staging_ring_depth);
      set_info.descriptorSetCount = options.staging_ring_depth;
      set_info.pSetLayouts = lz4_layouts.data();
      check(vkAllocateDescriptorSets(context_->device, &set_info,
                                     qh5_lz4_descriptor_sets.data()),
            "vkAllocateDescriptorSets(QH5 LZ4)");
      std::vector<VkDescriptorSetLayout> unshuffle_layouts(
          options.staging_ring_depth,
          pipelines_->qh5_unshuffle_descriptor_layout);
      qh5_unshuffle_descriptor_sets.resize(options.staging_ring_depth);
      set_info.pSetLayouts = unshuffle_layouts.data();
      check(vkAllocateDescriptorSets(context_->device, &set_info,
                                     qh5_unshuffle_descriptor_sets.data()),
            "vkAllocateDescriptorSets(QH5 unshuffle)");

      for (std::uint32_t slot = 0; slot < options.staging_ring_depth; ++slot) {
        std::array<VkDescriptorBufferInfo, 7> lz4_buffers{{
            {qh5_compressed_ring[slot].buffer, 0,
             qh5_compressed_ring[slot].requested_bytes},
            {qh5_metadata_ring[slot].buffer, 0,
             qh5_metadata_ring[slot].requested_bytes},
            {source_ring[slot].buffer, 0, source_ring[slot].requested_bytes},
            {qh5_status_ring[slot].buffer, 0,
             qh5_status_ring[slot].requested_bytes},
            {qh5_exclusion_mask.buffer, 0,
             qh5_exclusion_mask.requested_bytes},
            {qh5_bitshuffled_ring[slot].buffer, 0,
             qh5_bitshuffled_ring[slot].requested_bytes},
            {qh5_bitshuffled_ring[slot].buffer, 0,
             qh5_bitshuffled_ring[slot].requested_bytes},
        }};
        std::array<VkWriteDescriptorSet, 7> lz4_writes{};
        for (std::uint32_t binding = 0; binding < lz4_writes.size();
             ++binding) {
          lz4_writes[binding] = vulkan_structure<VkWriteDescriptorSet>(
              VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
          lz4_writes[binding].dstSet = qh5_lz4_descriptor_sets[slot];
          lz4_writes[binding].dstBinding = binding;
          lz4_writes[binding].descriptorCount = 1;
          lz4_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          lz4_writes[binding].pBufferInfo = &lz4_buffers[binding];
        }
        vkUpdateDescriptorSets(context_->device,
                               static_cast<std::uint32_t>(lz4_writes.size()),
                               lz4_writes.data(), 0, nullptr);

        std::array<VkDescriptorBufferInfo, 2> unshuffle_buffers{{
            {qh5_bitshuffled_ring[slot].buffer, 0,
             qh5_bitshuffled_ring[slot].requested_bytes},
            {source_ring[slot].buffer, 0, source_ring[slot].requested_bytes},
        }};
        std::array<VkWriteDescriptorSet, 2> unshuffle_writes{};
        for (std::uint32_t binding = 0; binding < unshuffle_writes.size();
             ++binding) {
          unshuffle_writes[binding] = vulkan_structure<VkWriteDescriptorSet>(
              VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
          unshuffle_writes[binding].dstSet =
              qh5_unshuffle_descriptor_sets[slot];
          unshuffle_writes[binding].dstBinding = binding;
          unshuffle_writes[binding].descriptorCount = 1;
          unshuffle_writes[binding].descriptorType =
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          unshuffle_writes[binding].pBufferInfo = &unshuffle_buffers[binding];
        }
        vkUpdateDescriptorSets(
            context_->device,
            static_cast<std::uint32_t>(unshuffle_writes.size()),
            unshuffle_writes.data(), 0, nullptr);
      }
    }

    run.fences.resize(options.staging_ring_depth, VK_NULL_HANDLE);
    auto fence_info = vulkan_structure<VkFenceCreateInfo>(
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (VkFence &fence : run.fences) {
      check(vkCreateFence(context_->device, &fence_info, nullptr, &fence),
            "vkCreateFence");
    }
    const bool timestamps = context_->timestamp_valid_bits != 0;
    const std::uint32_t queries_per_shard = gpu_indexed_qh5 ? 5U : 3U;
    if (timestamps) {
      auto query_info = vulkan_structure<VkQueryPoolCreateInfo>(
          VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
      query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_info.queryCount = shard_count * queries_per_shard;
      check(vkCreateQueryPool(context_->device, &query_info, nullptr,
                              &run.query_pool),
            "vkCreateQueryPool");
    }

    BenchmarkMetrics metrics;
    metrics.parity_checked = reference != nullptr;
    for (const Buffer &buffer : source_ring) {
      metrics.vulkan_committed_bytes += buffer.committed_bytes;
    }
    for (const Buffer &buffer : qh5_compressed_ring)
      metrics.vulkan_committed_bytes += buffer.committed_bytes;
    for (const Buffer &buffer : qh5_metadata_ring)
      metrics.vulkan_committed_bytes += buffer.committed_bytes;
    for (const Buffer &buffer : qh5_bitshuffled_ring)
      metrics.vulkan_committed_bytes += buffer.committed_bytes;
    for (const Buffer &buffer : qh5_status_ring)
      metrics.vulkan_committed_bytes += buffer.committed_bytes;
    metrics.vulkan_committed_bytes += membership.committed_bytes;
    metrics.vulkan_committed_bytes += scan_products.committed_bytes;
    metrics.vulkan_committed_bytes += diffraction_sums.committed_bytes;
    metrics.vulkan_committed_bytes += selected.committed_bytes;
    metrics.vulkan_committed_bytes += qh5_exclusion_mask.committed_bytes;
    metrics.setup_milliseconds = milliseconds(Clock::now() - setup_started);
    const auto execution_started = Clock::now();
    std::vector<bool> slot_in_flight(options.staging_ring_depth, false);
    bool first_correct_recorded = false;
    std::uint32_t completed_scans = 0;

    const auto verify_scan_range = [&](const std::uint32_t first_scan,
                                       const std::uint32_t count) {
      if (reference == nullptr)
        return std::uint64_t{0};
      context_->invalidate(scan_products);
      const auto *values =
          static_cast<const std::uint32_t *>(scan_products.mapped);
      std::uint64_t mismatches = 0;
      for (std::uint32_t local = 0; local < count; ++local) {
        const std::uint32_t scan = first_scan + local;
        const auto &expected =
            reference->expected_by_shift[scan & 255U].products;
        for (std::uint32_t product = 0; product < 6; ++product) {
          if (values[static_cast<std::uint64_t>(product) * scan_count + scan] !=
              expected[product]) {
            ++mismatches;
          }
        }
      }
      return mismatches;
    };
    const auto validate_qh5_decode = [&](const std::uint32_t slot,
                                         const std::uint32_t scans) {
      if (!gpu_indexed_qh5)
        return;
      context_->invalidate(qh5_status_ring[slot]);
      const auto *values =
          static_cast<const std::uint32_t *>(qh5_status_ring[slot].mapped);
      const std::uint64_t block_count = checked_product(
          scans, indexed_source->blocks_per_frame(), "QH5 completed blocks");
      for (std::uint64_t block = 0; block < block_count; ++block) {
        if (values[block] != 0U) {
          throw std::runtime_error(
              "QH5 GPU LZ4 decode rejected block " +
              std::to_string(block) + " with status " +
              std::to_string(values[block]));
        }
      }
    };

    struct Parameters {
      std::uint32_t scan_count;
      std::uint32_t detector_rows;
      std::uint32_t detector_columns;
      std::uint32_t global_scan_offset;
      std::uint32_t total_scan_count;
      std::uint32_t selected_global_scan;
      std::uint32_t reserved0;
      std::uint32_t reserved1;
    };
    const std::uint32_t selected_global_scan =
        options.selected_scan_row * options.source_shape.scan_columns +
        options.selected_scan_column;
    std::vector<std::uint32_t> submitted_first_scan(options.staging_ring_depth,
                                                    0);
    std::vector<std::uint32_t> submitted_scan_count(options.staging_ring_depth,
                                                    0);

    for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
      if (options.cancellation_flag != nullptr &&
          options.cancellation_flag->load(std::memory_order_relaxed)) {
        throw OperationCancelled();
      }
      const std::uint32_t slot = shard % options.staging_ring_depth;
      if (slot_in_flight[slot]) {
        check(vkWaitForFences(context_->device, 1, &run.fences[slot], VK_TRUE,
                              std::numeric_limits<std::uint64_t>::max()),
              "vkWaitForFences(reuse)");
        ++metrics.fence_wait_count;
        validate_qh5_decode(slot, submitted_scan_count[slot]);
        bool first_product_ready = false;
        if (!first_correct_recorded) {
          metrics.mismatch_count += verify_scan_range(
              submitted_first_scan[slot], submitted_scan_count[slot]);
          metrics.first_correct_product_milliseconds =
              milliseconds(Clock::now() - execution_started);
          first_correct_recorded = true;
          first_product_ready = true;
        }
        completed_scans =
            std::max(completed_scans,
                     submitted_first_scan[slot] + submitted_scan_count[slot]);
        if (options.progress_callback != nullptr) {
          options.progress_callback(options.progress_context, completed_scans,
                                    scan_count, first_product_ready);
        }
      }
      check(vkResetCommandBuffer(commands[slot], 0), "vkResetCommandBuffer");

      const std::uint32_t row_start = shard * options.shard_scan_rows;
      const std::uint32_t rows = std::min(
          options.shard_scan_rows, options.source_shape.scan_rows - row_start);
      const std::uint32_t first_scan =
          row_start * options.source_shape.scan_columns;
      const std::uint32_t scans = rows * options.source_shape.scan_columns;
      const VkDeviceSize shard_bytes = checked_product(
          checked_product(scans, detector_pixels, "shard values"),
          source_bytes_per_value, "shard bytes");
      std::uint32_t qh5_block_count = 0;
      std::uint32_t qh5_compressed_byte_count = 0;
      const auto stage_started = Clock::now();
      auto *destination = static_cast<std::uint8_t *>(source_ring[slot].mapped);
      if (reference != nullptr) {
        for (std::uint32_t local_scan = 0; local_scan < scans; ++local_scan) {
          const std::uint32_t global_scan = first_scan + local_scan;
          const std::uint8_t *source =
              reference->frames_by_shift.data() +
              static_cast<std::size_t>(global_scan & 255U) * detector_pixels;
          std::memcpy(destination + static_cast<std::size_t>(local_scan) *
                                        detector_pixels,
                      source, detector_pixels);
        }
        metrics.source_staging_milliseconds +=
            milliseconds(Clock::now() - stage_started);
      } else {
        if (indexed_source != nullptr) {
          const Qh5CompressedBatch batch =
              indexed_source->read_compressed_frames(first_scan, scans);
          const std::uint64_t metadata_bytes =
              batch.block_metadata.size() * sizeof(std::uint32_t);
          if (batch.compressed_bytes.size() >
                  qh5_compressed_ring[slot].requested_bytes ||
              metadata_bytes > qh5_metadata_ring[slot].requested_bytes ||
              batch.metrics.source_block_count >
                  std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "QH5 compressed batch exceeds its admitted GPU staging ring");
          }
          qh5_block_count = static_cast<std::uint32_t>(
              batch.metrics.source_block_count);
          qh5_compressed_byte_count = batch.compressed_byte_count;
          std::memcpy(qh5_compressed_ring[slot].mapped,
                      batch.compressed_bytes.data(),
                      batch.compressed_bytes.size());
          std::memcpy(qh5_metadata_ring[slot].mapped,
                      batch.block_metadata.data(),
                      static_cast<std::size_t>(metadata_bytes));
          metrics.storage_read_milliseconds +=
              batch.metrics.storage_read_milliseconds;
          metrics.source_bytes_read += batch.metrics.source_bytes_read;
          metrics.source_staging_milliseconds +=
              milliseconds(Clock::now() - stage_started) -
              batch.metrics.storage_read_milliseconds;
          const auto visibility_started = Clock::now();
          context_->flush(qh5_compressed_ring[slot],
                          batch.compressed_bytes.size());
          context_->flush(qh5_metadata_ring[slot], metadata_bytes);
          metrics.vulkan_visibility_milliseconds +=
              milliseconds(Clock::now() - visibility_started);
          metrics.source_bytes_staged += batch.compressed_byte_count;
        } else {
          if (prepared_source == nullptr) {
            throw std::logic_error("Vulkan execution has no source reader");
          }
          prepared_source->read(
              checked_product(checked_product(first_scan, detector_pixels,
                                              "prepared source value offset"),
                              source_bytes_per_value,
                              "prepared source byte offset"),
              shard_bytes, destination);
          metrics.storage_read_milliseconds +=
              milliseconds(Clock::now() - stage_started);
          metrics.source_bytes_read += shard_bytes;
        }
      }
      if (!gpu_indexed_qh5) {
        const auto visibility_started = Clock::now();
        context_->flush(source_ring[slot], shard_bytes);
        metrics.vulkan_visibility_milliseconds +=
            milliseconds(Clock::now() - visibility_started);
        metrics.source_bytes_staged += shard_bytes;
      }

      const Parameters parameters{
          scans,
          options.source_shape.detector_rows,
          options.source_shape.detector_columns,
          first_scan,
          scan_count,
          selected_global_scan,
          0,
          0,
      };
      auto begin = vulkan_structure<VkCommandBufferBeginInfo>(
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(commands[slot], &begin),
            "vkBeginCommandBuffer");
      const std::uint32_t query_base = shard * queries_per_shard;
      if (timestamps) {
        vkCmdResetQueryPool(commands[slot], run.query_pool, query_base,
                            queries_per_shard);
        vkCmdWriteTimestamp(commands[slot], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            run.query_pool, query_base);
      }
      if (gpu_indexed_qh5) {
        const std::array<std::uint32_t, 4> lz4_parameters{
            qh5_block_count, qh5_compressed_byte_count, 8192U,
            qh5_exclusions != nullptr
                ? (0x80000000U | indexed_source->blocks_per_frame())
                : 0U};
        vkCmdBindDescriptorSets(
            commands[slot], VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelines_->qh5_lz4_pipeline_layout, 0, 1,
            &qh5_lz4_descriptor_sets[slot], 0, nullptr);
        vkCmdPushConstants(commands[slot],
                           pipelines_->qh5_lz4_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(lz4_parameters), lz4_parameters.data());
        vkCmdBindPipeline(commands[slot], VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelines_->qh5_lz4_pipeline);
        vkCmdDispatch(commands[slot], qh5_block_count, 1, 1);
        if (timestamps) {
          vkCmdWriteTimestamp(commands[slot],
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              run.query_pool, query_base + 1U);
        }
        if (timestamps) {
          vkCmdWriteTimestamp(commands[slot],
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              run.query_pool, query_base + 2U);
        }
        auto decoded_barrier = vulkan_structure<VkBufferMemoryBarrier>(
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
        decoded_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        decoded_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        decoded_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        decoded_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        decoded_barrier.buffer = source_ring[slot].buffer;
        decoded_barrier.size = shard_bytes;
        vkCmdPipelineBarrier(commands[slot],
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &decoded_barrier, 0, nullptr);
      }
      vkCmdBindDescriptorSets(commands[slot], VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelines_->pipeline_layout, 0, 1,
                              &descriptor_sets[slot], 0, nullptr);
      vkCmdPushConstants(commands[slot], pipelines_->pipeline_layout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters),
                         &parameters);
      vkCmdBindPipeline(commands[slot], VK_PIPELINE_BIND_POINT_COMPUTE,
                        source_dtype == SourceDType::uint8
                            ? pipelines_->scan_pipeline
                            : pipelines_->scan_uint16_pipeline);
      vkCmdDispatch(commands[slot], scans, 1, 1);
      if (timestamps) {
        vkCmdWriteTimestamp(commands[slot],
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            run.query_pool,
                            query_base + (gpu_indexed_qh5 ? 3U : 1U));
      }
      vkCmdBindPipeline(commands[slot], VK_PIPELINE_BIND_POINT_COMPUTE,
                        source_dtype == SourceDType::uint8
                            ? pipelines_->mean_pipeline
                            : pipelines_->mean_uint16_pipeline);
      vkCmdDispatch(commands[slot], (detector_pixels + 63U) / 64U, 1, 1);
      if (timestamps) {
        vkCmdWriteTimestamp(commands[slot],
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            run.query_pool,
                            query_base + (gpu_indexed_qh5 ? 4U : 2U));
      }
      std::vector<VkBufferMemoryBarrier> barriers(gpu_indexed_qh5 ? 4U : 3U);
      const std::array<VkBuffer, 4> barrier_buffers{
          scan_products.buffer, diffraction_sums.buffer, selected.buffer,
          gpu_indexed_qh5 ? qh5_status_ring[slot].buffer : VK_NULL_HANDLE};
      const std::array<VkDeviceSize, 4> barrier_sizes{
          scan_products.requested_bytes, diffraction_sums.requested_bytes,
          selected.requested_bytes,
          gpu_indexed_qh5 ? qh5_status_ring[slot].requested_bytes : 0U};
      for (std::uint32_t index = 0; index < barriers.size(); ++index) {
        barriers[index] = vulkan_structure<VkBufferMemoryBarrier>(
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
        barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[index].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[index].buffer = barrier_buffers[index];
        barriers[index].size = barrier_sizes[index];
      }
      vkCmdPipelineBarrier(commands[slot], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                           static_cast<std::uint32_t>(barriers.size()),
                           barriers.data(), 0, nullptr);
      check(vkEndCommandBuffer(commands[slot]), "vkEndCommandBuffer");
      auto submit =
          vulkan_structure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &commands[slot];
      check(vkResetFences(context_->device, 1, &run.fences[slot]),
            "vkResetFences");
      check(vkQueueSubmit(context_->queue, 1, &submit, run.fences[slot]),
            "vkQueueSubmit");
      run.queue_work_may_be_active = true;
      ++metrics.queue_submit_count;
      slot_in_flight[slot] = true;
      submitted_first_scan[slot] = first_scan;
      submitted_scan_count[slot] = scans;

      if (!first_correct_recorded || options.wait_after_each_shard) {
        check(vkWaitForFences(context_->device, 1, &run.fences[slot], VK_TRUE,
                              std::numeric_limits<std::uint64_t>::max()),
              "vkWaitForFences(serial)");
        ++metrics.fence_wait_count;
        validate_qh5_decode(slot, scans);
        metrics.mismatch_count += verify_scan_range(first_scan, scans);
        slot_in_flight[slot] = false;
        bool first_product_ready = false;
        if (!first_correct_recorded) {
          metrics.first_correct_product_milliseconds =
              milliseconds(Clock::now() - execution_started);
          first_correct_recorded = true;
          first_product_ready = true;
        }
        completed_scans = std::max(completed_scans, first_scan + scans);
        if (options.progress_callback != nullptr) {
          options.progress_callback(options.progress_context, completed_scans,
                                    scan_count, first_product_ready);
        }
      }
    }

    for (std::uint32_t slot = 0; slot < options.staging_ring_depth; ++slot) {
      if (!slot_in_flight[slot])
        continue;
      check(vkWaitForFences(context_->device, 1, &run.fences[slot], VK_TRUE,
                            std::numeric_limits<std::uint64_t>::max()),
            "vkWaitForFences(final)");
      ++metrics.fence_wait_count;
      validate_qh5_decode(slot, submitted_scan_count[slot]);
      bool first_product_ready = false;
      if (!first_correct_recorded) {
        metrics.mismatch_count += verify_scan_range(submitted_first_scan[slot],
                                                    submitted_scan_count[slot]);
        metrics.first_correct_product_milliseconds =
            milliseconds(Clock::now() - execution_started);
        first_correct_recorded = true;
        first_product_ready = true;
      }
      completed_scans =
          std::max(completed_scans,
                   submitted_first_scan[slot] + submitted_scan_count[slot]);
      if (options.progress_callback != nullptr) {
        options.progress_callback(options.progress_context, completed_scans,
                                  scan_count, first_product_ready);
      }
    }
    run.queue_work_may_be_active = false;
    context_->invalidate(scan_products);
    context_->invalidate(diffraction_sums);
    context_->invalidate(selected);
    metrics.full_exact_completion_milliseconds =
        milliseconds(Clock::now() - execution_started);
    metrics.shader_source_bytes_read = checked_product(
        checked_product(options.source_shape.value_count(),
                        source_bytes_per_value, "logical source bytes"),
        2, "shader source reads");

    const auto *product_values_u8 =
        static_cast<const std::uint32_t *>(scan_products.mapped);
    if (reference != nullptr) {
      for (std::uint32_t scan = 0; scan < scan_count; ++scan) {
        const auto &expected =
            reference->expected_by_shift[scan & 255U].products;
        for (std::uint32_t product = 0; product < 6; ++product) {
          if (product_values_u8[static_cast<std::uint64_t>(product) *
                                    scan_count +
                                scan] != expected[product]) {
            ++metrics.mismatch_count;
          }
        }
      }
      const auto *detector_sum_values =
          static_cast<const std::uint32_t *>(diffraction_sums.mapped);
      const std::uint32_t full_cycles = scan_count / 256U;
      const std::uint32_t remaining = scan_count % 256U;
      for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
        std::uint32_t expected = full_cycles * 32640U;
        for (std::uint32_t shift = 0; shift < remaining; ++shift) {
          expected += (pixel + shift) & 255U;
        }
        if (detector_sum_values[pixel] != expected)
          ++metrics.mismatch_count;
      }
      const auto *selected_values =
          static_cast<const std::uint8_t *>(selected.mapped);
      const std::uint8_t *expected_selected =
          reference->frames_by_shift.data() +
          static_cast<std::size_t>(selected_global_scan & 255U) *
              detector_pixels;
      for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
        if (selected_values[pixel] != expected_selected[pixel]) {
          ++metrics.mismatch_count;
        }
      }
    }

    if (timestamps) {
      std::vector<std::uint64_t> values(static_cast<std::size_t>(shard_count) *
                                        queries_per_shard);
      check(vkGetQueryPoolResults(
                context_->device, run.query_pool, 0,
                shard_count * queries_per_shard,
                values.size() * sizeof(std::uint64_t), values.data(),
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults");
      for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
        const double period =
            capabilities_.timestamp_period_nanoseconds / 1.0e6;
        const std::uint32_t base = shard * queries_per_shard;
        if (gpu_indexed_qh5) {
          const double lz4_milliseconds =
              static_cast<double>(values[base + 1U] - values[base]) * period;
          const double bitunshuffle_milliseconds =
              static_cast<double>(values[base + 2U] - values[base + 1U]) *
              period;
          metrics.gpu_source_lz4_milliseconds += lz4_milliseconds;
          metrics.gpu_source_bitunshuffle_milliseconds +=
              bitunshuffle_milliseconds;
          metrics.source_decode_milliseconds +=
              lz4_milliseconds + bitunshuffle_milliseconds;
          metrics.gpu_scan_products_milliseconds +=
              static_cast<double>(values[base + 3U] - values[base + 2U]) *
              period;
          metrics.gpu_mean_diffraction_milliseconds +=
              static_cast<double>(values[base + 4U] - values[base + 3U]) *
              period;
        } else {
          metrics.gpu_scan_products_milliseconds +=
              static_cast<double>(values[base + 1U] - values[base]) * period;
          metrics.gpu_mean_diffraction_milliseconds +=
              static_cast<double>(values[base + 2U] - values[base + 1U]) *
              period;
        }
      }
    }

    if (products != nullptr) {
      products->source_shape = options.source_shape;
      products->source_dtype = source_dtype;
      if (source_dtype == SourceDType::uint8) {
        const auto *detector_sum_values =
            static_cast<const std::uint32_t *>(diffraction_sums.mapped);
        const auto *selected_values =
            static_cast<const std::uint8_t *>(selected.mapped);
        products->total_intensity.assign(product_values_u8,
                                         product_values_u8 + scan_count);
        products->band1.assign(product_values_u8 + scan_count,
                               product_values_u8 + 2ULL * scan_count);
        products->band2.assign(product_values_u8 + 2ULL * scan_count,
                               product_values_u8 + 3ULL * scan_count);
        products->band4.assign(product_values_u8 + 3ULL * scan_count,
                               product_values_u8 + 4ULL * scan_count);
        products->detector_row_moment.assign(
            product_values_u8 + 4ULL * scan_count,
            product_values_u8 + 5ULL * scan_count);
        products->detector_column_moment.assign(
            product_values_u8 + 5ULL * scan_count,
            product_values_u8 + 6ULL * scan_count);
        products->diffraction_sum.assign(detector_sum_values,
                                         detector_sum_values + detector_pixels);
        products->selected_diffraction_uint8.assign(
            selected_values, selected_values + detector_pixels);
        products->selected_diffraction_uint16.clear();
      } else {
        const auto *product_values =
            static_cast<const std::uint64_t *>(scan_products.mapped);
        const auto *detector_sum_values =
            static_cast<const std::uint64_t *>(diffraction_sums.mapped);
        const auto *selected_values =
            static_cast<const std::uint16_t *>(selected.mapped);
        products->total_intensity.assign(product_values,
                                         product_values + scan_count);
        products->band1.assign(product_values + scan_count,
                               product_values + 2ULL * scan_count);
        products->band2.assign(product_values + 2ULL * scan_count,
                               product_values + 3ULL * scan_count);
        products->band4.assign(product_values + 3ULL * scan_count,
                               product_values + 4ULL * scan_count);
        products->detector_row_moment.assign(product_values + 4ULL * scan_count,
                                             product_values +
                                                 5ULL * scan_count);
        products->detector_column_moment.assign(
            product_values + 5ULL * scan_count,
            product_values + 6ULL * scan_count);
        products->diffraction_sum.assign(detector_sum_values,
                                         detector_sum_values + detector_pixels);
        products->selected_diffraction_uint16.assign(
            selected_values, selected_values + detector_pixels);
        products->selected_diffraction_uint8.clear();
      }
    }
    rusage usage_after{};
    if (getrusage(RUSAGE_SELF, &usage_after) != 0) {
      throw std::runtime_error("getrusage failed after the benchmark");
    }
    metrics.user_cpu_milliseconds = timeval_milliseconds(usage_after.ru_utime) -
                                    timeval_milliseconds(usage_before.ru_utime);
    metrics.system_cpu_milliseconds =
        timeval_milliseconds(usage_after.ru_stime) -
        timeval_milliseconds(usage_before.ru_stime);
    metrics.maximum_resident_set_kibibytes =
        static_cast<std::uint64_t>(usage_after.ru_maxrss);
    metrics.minor_page_fault_count = static_cast<std::uint64_t>(
        usage_after.ru_minflt - usage_before.ru_minflt);
    metrics.major_page_fault_count = static_cast<std::uint64_t>(
        usage_after.ru_majflt - usage_before.ru_majflt);
    return metrics;
  }

  std::unique_ptr<VulkanContext> context_;
  VulkanCapabilities capabilities_;
  std::unique_ptr<PipelineResources> pipelines_;
};

} // namespace

std::unique_ptr<ExactProductExecutor> ExactProductExecutor::create() {
  return std::make_unique<Executor>();
}

} // namespace quantem::gpu::vulkan
