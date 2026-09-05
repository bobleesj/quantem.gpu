#include "quantem/gpu/vulkan/c_api.h"

#include <stddef.h>

_Static_assert(QGPU_VULKAN_ABI_VERSION == 2u, "unexpected ABI version");
_Static_assert(offsetof(qgpu_error, struct_size) == 0, "qgpu_error prefix");
_Static_assert(offsetof(qgpu_vulkan_event, struct_size) == 0,
               "qgpu_vulkan_event prefix");
_Static_assert(offsetof(qgpu_vulkan_selected_diffraction_request,
                        struct_size) == 0,
               "selected diffraction request prefix");
_Static_assert(offsetof(qgpu_vulkan_selected_diffraction_result, struct_size) ==
                   0,
               "selected diffraction result prefix");
typedef qgpu_status (*qgpu_selected_diffraction_function)(
    qgpu_vulkan_session *, const qgpu_vulkan_selected_diffraction_request *,
    qgpu_vulkan_selected_diffraction_result *, qgpu_error *);
_Static_assert(_Generic(&qgpu_vulkan_read_selected_diffraction_v1,
                   qgpu_selected_diffraction_function: 1,
                   default: 0),
               "selected diffraction function signature");

int main(void) {
  qgpu_vulkan_event event = {0};
  event.struct_size = sizeof(event);
  qgpu_vulkan_selected_diffraction_request request = {0};
  request.struct_size = sizeof(request);
  qgpu_vulkan_selected_diffraction_result result = {0};
  result.struct_size = sizeof(result);
  return event.struct_size == 0 || request.struct_size == 0 ||
                 result.struct_size == 0
             ? 1
             : 0;
}
