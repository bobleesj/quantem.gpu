#include "quantem/gpu/vulkan/contract.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using quantem::gpu::vulkan::DeviceLimits;
using quantem::gpu::vulkan::ExactProducts;
using quantem::gpu::vulkan::ScientificRequest;
using quantem::gpu::vulkan::Shape4D;
using quantem::gpu::vulkan::SourceDType;

namespace {

void require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void expect_invalid(const ScientificRequest &request) {
  try {
    quantem::gpu::vulkan::validate_scientific_request(request);
    require(false, "request should have failed validation");
  } catch (const std::invalid_argument &) {
  }
}

} // namespace

int main() {
  const ScientificRequest request{
      Shape4D{512, 512, 192, 192},
      SourceDType::uint8,
      1,
      1,
      true,
      true,
      true,
      true,
      true,
      true,
  };
  quantem::gpu::vulkan::validate_scientific_request(request);
  const auto plan = quantem::gpu::vulkan::make_exact_load_plan(
      request,
      DeviceLimits{2147483647ULL, 1073741824ULL, 4ULL * 1024 * 1024 * 1024});
  require(plan.logical_source_bytes == 9663676416ULL, "logical source bytes");
  require(plan.decoded_frame_bytes == 36864ULL, "decoded frame bytes");
  require(plan.shard_scan_rows > 0, "positive shard rows");
  require(plan.shard_scan_rows < 512, "bounded shard rows");
  require(plan.staging_ring_depth == 3, "staging ring depth");
  require(!plan.full_volume_resident, "full volume must not be resident");
  require(quantem::gpu::vulkan::representation_name(plan.representation) == "dense",
          "dense staging representation is independent of full residency");
  require(quantem::gpu::vulkan::representation_name(
              quantem::gpu::vulkan::DataRepresentation::packed) ==
              "packed", "shared packed representation name");
  require(quantem::gpu::vulkan::representation_name(
              quantem::gpu::vulkan::DataRepresentation::ans) ==
              "ans", "shared ANS name does not imply Vulkan ANS kernel support");
  require(plan.maximum_shard_bytes <= 1073741824ULL, "allocation bound");

  auto binned = request;
  binned.detector_bin = 2;
  expect_invalid(binned);
  auto cropped = request;
  cropped.crop_is_none = false;
  expect_invalid(cropped);
  auto u16 = request;
  u16.source_dtype = SourceDType::uint16;
  quantem::gpu::vulkan::validate_scientific_request(u16);
  const auto u16_plan = quantem::gpu::vulkan::make_exact_load_plan(
      u16,
      DeviceLimits{2147483647ULL, 1073741824ULL, 4ULL * 1024 * 1024 * 1024});
  require(u16_plan.logical_source_bytes == 19327352832ULL,
          "uint16 logical source bytes");
  require(u16_plan.decoded_frame_bytes == 73728ULL,
          "uint16 decoded frame bytes");
  require(!u16_plan.full_volume_resident,
          "uint16 full volume must not be resident");

  ExactProducts exact;
  exact.source_shape = Shape4D{1, 2, 1, 2};
  exact.total_intensity = {4, 0};
  exact.detector_row_moment = {0, 0};
  exact.detector_column_moment = {3, 0};
  exact.diffraction_sum = {1, 3};
  const auto derived = quantem::gpu::vulkan::derive_products(exact);
  require(derived.global_total_intensity == 4, "global total");
  require(std::fabs(derived.mean_diffraction[0] - 0.5F) < 1e-6F,
          "mean pixel zero");
  require(std::fabs(derived.mean_diffraction[1] - 1.5F) < 1e-6F,
          "mean pixel one");
  require(std::fabs(derived.center_of_mass_row[0]) < 1e-6F, "row CoM");
  require(std::fabs(derived.center_of_mass_column[0] - 0.375F) < 1e-6F,
          "column CoM zero");
  require(std::fabs(derived.center_of_mass_column[1] + 0.375F) < 1e-6F,
          "column CoM one");

  quantem::gpu::vulkan::DerivedProducts dpc_input;
  dpc_input.source_shape = Shape4D{4, 4, 1, 1};
  dpc_input.center_of_mass_row = {
      -1.0F, -1.0F, -1.0F, -1.0F, -0.5F, -0.5F, -0.5F, -0.5F,
      0.5F,  0.5F,  0.5F,  0.5F,  1.0F,  1.0F,  1.0F,  1.0F,
  };
  dpc_input.center_of_mass_column.assign(16, 0.0F);
  const auto dpc = quantem::gpu::vulkan::derive_dpc(
      dpc_input, quantem::gpu::vulkan::DpcOptions{false, 0.0F, 180});
  require(dpc.aligned_row == dpc_input.center_of_mass_row, "fixed DPC row");
  require(dpc.aligned_column == dpc_input.center_of_mass_column,
          "fixed DPC column");
  require(dpc.integrated_phase.size() == 16, "iDPC shape");
  float phase_sum = 0.0F;
  for (const float value : dpc.integrated_phase)
    phase_sum += value;
  require(std::fabs(phase_sum) < 1e-6F, "zero-mean iDPC");
  require(std::fabs(dpc.integrated_phase[0] + 0.75F) < 1e-6F, "iDPC first row");
  require(std::fabs(dpc.integrated_phase[4] - 0.75F) < 1e-6F,
          "iDPC second row");

  std::cout << "PASS: quantem.gpu Android Vulkan contract tests\n";
  return 0;
}
