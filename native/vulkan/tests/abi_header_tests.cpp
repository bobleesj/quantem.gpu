#include "quantem/gpu/vulkan/c_api.h"
#include "quantem/gpu/vulkan/session.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

static_assert(QGPU_VULKAN_ABI_VERSION == 2);
static_assert(std::is_standard_layout_v<qgpu_source_segment>);
static_assert(std::is_standard_layout_v<qgpu_qh5_index_segment>);
static_assert(std::is_standard_layout_v<qgpu_dataset_descriptor>);
static_assert(std::is_standard_layout_v<qgpu_vulkan_open_request>);
static_assert(std::is_standard_layout_v<qgpu_vulkan_product_request>);
static_assert(
    std::is_standard_layout_v<qgpu_vulkan_selected_diffraction_request>);
static_assert(
    std::is_standard_layout_v<qgpu_vulkan_selected_diffraction_metrics>);
static_assert(
    std::is_standard_layout_v<qgpu_vulkan_selected_diffraction_result>);
static_assert(std::is_standard_layout_v<qgpu_vulkan_event>);
static_assert(std::is_standard_layout_v<qgpu_vulkan_result_view>);
static_assert(offsetof(qgpu_error, struct_size) == 0);
static_assert(offsetof(qgpu_source_segment, struct_size) == 0);
static_assert(offsetof(qgpu_dataset_descriptor, struct_size) == 0);
static_assert(offsetof(qgpu_vulkan_open_request, struct_size) == 0);
static_assert(offsetof(qgpu_vulkan_product_request, struct_size) == 0);
static_assert(offsetof(qgpu_vulkan_selected_diffraction_request, struct_size) ==
              0);
static_assert(offsetof(qgpu_vulkan_selected_diffraction_result, struct_size) ==
              0);
static_assert(offsetof(qgpu_vulkan_event, struct_size) == 0);
static_assert(offsetof(qgpu_vulkan_result_view, struct_size) == 0);
using SelectedDiffractionFunction = qgpu_status (*)(
    qgpu_vulkan_session *, const qgpu_vulkan_selected_diffraction_request *,
    qgpu_vulkan_selected_diffraction_result *, qgpu_error *);
static_assert(
    std::is_same_v<decltype(&qgpu_vulkan_read_selected_diffraction_v1),
                   SelectedDiffractionFunction>);

int main() {
  qgpu_vulkan_open_request open{};
  open.struct_size = sizeof(open);
  open.abi_version = QGPU_VULKAN_ABI_VERSION;
  open.dataset.struct_size = sizeof(open.dataset);
  qgpu_vulkan_product_request request{};
  request.struct_size = sizeof(request);
  qgpu_vulkan_selected_diffraction_request selected{};
  selected.struct_size = sizeof(selected);
  qgpu_vulkan_selected_diffraction_result selected_result{};
  selected_result.struct_size = sizeof(selected_result);
  if (open.abi_version != 2 || request.struct_size == 0 ||
      selected.struct_size == 0 || selected_result.struct_size == 0)
    return 1;
  std::cout << "PASS: quantem.gpu Android Vulkan ABI headers\n";
  return 0;
}
