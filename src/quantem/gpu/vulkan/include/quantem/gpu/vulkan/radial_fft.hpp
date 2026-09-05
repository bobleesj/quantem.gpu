#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>

namespace quantem::gpu::vulkan {

struct RadialFftAdmission {
  std::string device_name;
  std::uint32_t driver_version = 0;
  std::uint64_t prefix_upload_bytes = 0;
  std::uint64_t committed_bytes = 0;
  double validation_milliseconds = 0.0;
  double initialization_milliseconds = 0.0;
  double prefix_copy_milliseconds = 0.0;
  bool timestamps_available = false;
};

struct RadialFftMetrics {
  std::uint64_t generation = 0;
  std::uint64_t output_copy_bytes = 0;
  std::uint64_t source_upload_bytes = 0;
  std::uint32_t queue_submit_count = 0;
  std::uint32_t fence_wait_count = 0;
  std::uint32_t dispatch_count = 0;
  std::uint32_t device_wide_wait_count = 0;
  double wall_milliseconds = 0.0;
  double output_copy_milliseconds = 0.0;
  // NaN means not measured/not requested, never a claimed zero-cost FFT.
  double gpu_annulus_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double gpu_fft_rows_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double gpu_fft_columns_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double gpu_magnitude_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double gpu_total_milliseconds = std::numeric_limits<double>::quiet_NaN();
};

/**
 * Fixed-center, integer-radius 512 × 512 scan reduction from an admitted radial prefix.
 *
 * This backend kernel owner does not load raw 4D data or verify source-file provenance.
 * The caller must authenticate the prefix against its source identity before construction.
 * Prefix layout is uint32[137,512,512], P[0]=0 and P[r+1]>=P[r]. Detector center is
 * (95.5,95.5); this specialization does not implement movable/fractional apertures.
 * One persistent upload and bounded workspace serve all requests. Requests serialize;
 * the caller owns latest-request coalescing, display publication, and generation checks.
 */
class RadialFftSession final {
public:
  static constexpr std::uint32_t scan_rows = 512;
  static constexpr std::uint32_t scan_columns = 512;
  static constexpr std::uint32_t plane_values = scan_rows * scan_columns;
  static constexpr std::uint32_t prefix_planes = 137;

  explicit RadialFftSession(std::span<const std::uint32_t> verified_prefix);
  ~RadialFftSession();
  RadialFftSession(const RadialFftSession &) = delete;
  RadialFftSession &operator=(const RadialFftSession &) = delete;

  [[nodiscard]] const RadialFftAdmission &admission() const noexcept;

  /**
   * Compute P[outer]-P[inner] exactly in uint32, without reading/reloading the raw volume.
   * image must contain 512² values. fft_magnitude may be empty to disable FFT, otherwise
   * it must contain 512² floats. The unnormalized forward 2D FFT uses float32 arithmetic
   * and returns fftshift(abs(FFT(image))); it is NOT bit-exact integer arithmetic.
   * Caller-owned outputs are copied only after successful GPU completion. Those explicit
   * 1/2 MiB readbacks are measured; this API does not claim zero-copy display integration.
   */
  [[nodiscard]] RadialFftMetrics request(
      std::uint32_t inner_radius, std::uint32_t outer_radius,
      std::uint64_t generation, std::span<std::uint32_t> image,
      std::span<float> fft_magnitude = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace quantem::gpu::vulkan
