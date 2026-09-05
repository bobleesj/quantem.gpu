#include "quantem/gpu/vulkan/contract.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace quantem::gpu::vulkan {
namespace {

std::uint64_t checked_product(const std::uint64_t left,
                              const std::uint64_t right, const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " exceeds uint64 range");
  }
  return left * right;
}

std::uint64_t checked_sum(const std::uint64_t left, const std::uint64_t right,
                          const char *label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::invalid_argument(std::string(label) + " exceeds uint64 range");
  }
  return left + right;
}

} // namespace

std::uint64_t Shape4D::scan_count() const {
  return checked_product(scan_rows, scan_columns, "scan count");
}

std::uint64_t Shape4D::detector_pixel_count() const {
  return checked_product(detector_rows, detector_columns,
                         "detector pixel count");
}

std::uint64_t Shape4D::value_count() const {
  return checked_product(scan_count(), detector_pixel_count(),
                         "4D value count");
}

std::uint32_t bytes_per_value(const SourceDType dtype) {
  switch (dtype) {
  case SourceDType::uint8:
    return 1;
  case SourceDType::uint16:
    return 2;
  case SourceDType::uint32:
    return 4;
  }
  throw std::invalid_argument("unsupported source dtype");
}

void validate_scientific_request(const ScientificRequest &request) {
  const Shape4D &shape = request.source_shape;
  if (shape.scan_rows == 0 || shape.scan_columns == 0 ||
      shape.detector_rows == 0 || shape.detector_columns == 0) {
    throw std::invalid_argument("source shape dimensions must be positive");
  }
  if (request.scan_bin != 1) {
    throw std::invalid_argument("exact Android loading currently requires "
                                "scan_bin=1; no scan position may be hidden");
  }
  if (request.detector_bin != 1) {
    throw std::invalid_argument(
        "exact Android loading currently requires detector_bin=1; detector "
        "binning must be explicit");
  }
  if (!request.crop_is_none) {
    throw std::invalid_argument(
        "exact Android loading currently requires crop=none; a cropped result "
        "cannot be labeled full source");
  }
  const std::uint64_t detector_pixels = shape.detector_pixel_count();
  const std::uint64_t maximum_value =
      request.source_dtype == SourceDType::uint8    ? 255
      : request.source_dtype == SourceDType::uint16 ? 65535
                                                    : 4294967295ULL;
  const std::uint64_t maximum_scan_total =
      checked_product(detector_pixels, maximum_value, "per-scan total");
  const std::uint64_t maximum_coordinate =
      std::max(shape.detector_rows - 1, shape.detector_columns - 1);
  (void)checked_product(maximum_scan_total, maximum_coordinate,
                        "per-scan detector moment");
  (void)checked_product(shape.scan_count(), maximum_value,
                        "mean diffraction sum");
}

LoadPlan make_exact_load_plan(const ScientificRequest &request,
                              const DeviceLimits &limits) {
  validate_scientific_request(request);
  if (limits.max_storage_buffer_range_bytes == 0 ||
      limits.max_memory_allocation_bytes == 0) {
    throw std::invalid_argument(
        "Vulkan storage-buffer and allocation limits are required");
  }

  const std::uint64_t frame_bytes = checked_product(
      request.source_shape.detector_pixel_count(),
      bytes_per_value(request.source_dtype), "decoded frame bytes");
  const std::uint64_t logical_bytes = checked_product(
      request.source_shape.scan_count(), frame_bytes, "logical source bytes");
  const std::uint64_t row_bytes = checked_product(
      request.source_shape.scan_columns, frame_bytes, "one scan-row window");
  const std::uint64_t maximum_buffer =
      std::min(limits.max_storage_buffer_range_bytes,
               limits.max_memory_allocation_bytes);
  if (row_bytes > maximum_buffer) {
    throw std::invalid_argument("one complete scan row exceeds the device's "
                                "individual Vulkan buffer limit");
  }

  const std::uint32_t ring_depth = 3;
  std::uint64_t working_budget = maximum_buffer;
  if (limits.available_process_bytes != 0) {
    working_budget = std::min(
        working_budget,
        std::max(row_bytes, limits.available_process_bytes / (ring_depth + 2)));
  }
  std::uint64_t rows = std::max<std::uint64_t>(1, working_budget / row_bytes);
  rows = std::min<std::uint64_t>(rows, request.source_shape.scan_rows);
  const std::uint64_t shard_bytes =
      checked_product(rows, row_bytes, "maximum shard bytes");
  const std::uint64_t scan_count = request.source_shape.scan_count();
  const std::uint64_t detector_pixels =
      request.source_shape.detector_pixel_count();
  const std::uint64_t scan_product_bytes =
      checked_product(checked_product(scan_count, 6, "six scan products"),
                      sizeof(std::uint64_t), "scan product bytes");
  const std::uint64_t detector_sum_bytes = checked_product(
      detector_pixels, sizeof(std::uint64_t), "detector sum bytes");
  const std::uint64_t selected_diffraction_bytes = frame_bytes;
  const std::uint64_t persistent_bytes =
      checked_sum(checked_sum(scan_product_bytes, detector_sum_bytes,
                              "persistent products"),
                  selected_diffraction_bytes, "persistent products");
  const std::uint64_t staged_bytes =
      checked_product(shard_bytes, ring_depth, "staging ring bytes");

  return {
      request.source_shape,
      request.source_dtype,
      logical_bytes,
      frame_bytes,
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>((request.source_shape.scan_rows + rows - 1) /
                                 rows),
      ring_depth,
      shard_bytes,
      persistent_bytes,
      checked_sum(staged_bytes, persistent_bytes, "estimated Vulkan bytes"),
      logical_bytes <= maximum_buffer,
  };
}

} // namespace quantem::gpu::vulkan
