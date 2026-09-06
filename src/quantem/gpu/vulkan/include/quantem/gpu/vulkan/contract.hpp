#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace quantem::gpu::vulkan {

// Representation describes the counts' encoding, independently of dtype,
// allocation location, and whether the complete scan remains resident.
enum class DataRepresentation : std::uint8_t { dense, packed, ans };

[[nodiscard]] constexpr std::string_view representation_name(DataRepresentation value) {
  switch (value) {
  case DataRepresentation::dense: return "dense";
  case DataRepresentation::packed: return "packed";
  case DataRepresentation::ans: return "ans";
  }
  throw std::invalid_argument("Unknown data representation");
}

enum class SourceDType : std::uint8_t {
  uint8,
  uint16,
  uint32,
};

struct Shape4D {
  std::uint32_t scan_rows = 0;
  std::uint32_t scan_columns = 0;
  std::uint32_t detector_rows = 0;
  std::uint32_t detector_columns = 0;

  [[nodiscard]] std::uint64_t scan_count() const;
  [[nodiscard]] std::uint64_t detector_pixel_count() const;
  [[nodiscard]] std::uint64_t value_count() const;
};

struct ScientificRequest {
  Shape4D source_shape;
  SourceDType source_dtype = SourceDType::uint8;
  std::uint32_t scan_bin = 1;
  std::uint32_t detector_bin = 1;
  bool crop_is_none = true;
  bool selected_diffraction = true;
  bool detector_bands = true;
  bool mean_diffraction = true;
  bool total_intensity = true;
  bool center_of_mass = true;
};

struct DeviceLimits {
  std::uint64_t max_storage_buffer_range_bytes = 0;
  std::uint64_t max_memory_allocation_bytes = 0;
  std::uint64_t available_process_bytes = 0;
};

struct LoadPlan {
  Shape4D source_shape;
  SourceDType source_dtype = SourceDType::uint8;
  std::uint64_t logical_source_bytes = 0;
  std::uint64_t decoded_frame_bytes = 0;
  std::uint32_t shard_scan_rows = 0;
  std::uint32_t shard_count = 0;
  std::uint32_t staging_ring_depth = 0;
  std::uint64_t maximum_shard_bytes = 0;
  std::uint64_t persistent_product_bytes = 0;
  std::uint64_t estimated_vulkan_bytes = 0;
  bool full_volume_resident = false;
  DataRepresentation representation = DataRepresentation::dense;
};

struct ExactProducts {
  Shape4D source_shape;
  SourceDType source_dtype = SourceDType::uint8;
  std::vector<std::uint64_t> total_intensity;
  std::vector<std::uint64_t> band1;
  std::vector<std::uint64_t> band2;
  std::vector<std::uint64_t> band4;
  std::vector<std::uint64_t> detector_row_moment;
  std::vector<std::uint64_t> detector_column_moment;
  std::vector<std::uint64_t> diffraction_sum;
  std::vector<std::uint8_t> selected_diffraction_uint8;
  std::vector<std::uint16_t> selected_diffraction_uint16;
};

struct DerivedProducts {
  Shape4D source_shape;
  std::vector<float> mean_diffraction;
  std::vector<float> center_of_mass_row;
  std::vector<float> center_of_mass_column;
  std::uint64_t global_total_intensity = 0;
};

struct DpcOptions {
  bool automatic_rotation = true;
  float fixed_rotation_degrees = 0.0F;
  std::uint32_t rotation_steps = 180;
};

struct DpcProducts {
  Shape4D source_shape;
  std::vector<float> aligned_row;
  std::vector<float> aligned_column;
  std::vector<float> integrated_phase;
  float rotation_degrees = 0.0F;
  bool component_order_exchanged = false;
};

[[nodiscard]] std::uint32_t bytes_per_value(SourceDType dtype);
void validate_scientific_request(const ScientificRequest &request);
[[nodiscard]] LoadPlan make_exact_load_plan(const ScientificRequest &request,
                                            const DeviceLimits &limits);
[[nodiscard]] DerivedProducts derive_products(const ExactProducts &products);
[[nodiscard]] DpcProducts derive_dpc(const DerivedProducts &products,
                                     const DpcOptions &options = {});

} // namespace quantem::gpu::vulkan
