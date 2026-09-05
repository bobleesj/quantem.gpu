#pragma once

#include "quantem/gpu/vulkan/contract.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace quantem::gpu::vulkan {

/** Lossless raw-uint16 detector-major shard, with 128 scan values per tile.
 *
 * Descriptors are [detector_pixel, ceil(scan_count/128)], one uint32 each.
 * The low five bits store width 0..16; the upper 27 store a payload word offset.
 * Every tile occupies 4*width uint32 words, including zero padding in the final
 * scan tile. The original uint16 values, including masked pixels, are retained.
 * This is a bounded preparation/admission representation, not a CPU drag path.
 * A Vulkan owner must enforce actual allocation/storage-range and process limits
 * per shard before uploading; this type does not claim device residency.
 */
struct PackedDetectorShard {
  static constexpr std::uint32_t scan_tile = 128;
  std::uint32_t scan_count = 0;
  std::uint32_t detector_pixels = 0;
  std::vector<std::uint32_t> descriptors;
  std::vector<std::uint32_t> words;
};

[[nodiscard]] PackedDetectorShard pack_detector_shard(
    std::span<const std::uint16_t> scan_major,
    std::uint32_t scan_count, std::uint32_t detector_pixels,
    std::uint64_t maximum_packed_bytes = 128ULL * 1024 * 1024);

/// Validate descriptor bounds/order/width once, before trusting a GPU upload.
void validate_packed_detector_shard(const PackedDetectorShard &shard);

/// Non-owning equivalent; neither span is retained and payload values are not altered.
void validate_packed_detector_shard(
    std::uint32_t scan_count, std::uint32_t detector_pixels,
    std::span<const std::uint32_t> descriptors,
    std::span<const std::uint32_t> words);

/** Check device buffer bounds and a reserved process budget before upload.
 * available_process_bytes is the total budget reserved for this admission,
 * including already_admitted_bytes and staging_bytes, not the device RAM size.
 * The owner must reserve/check the complete source plan, not promote a partially
 * uploaded source as resident. Authentication is also required before upload.
 */
[[nodiscard]] std::uint64_t admit_packed_detector_shard(
    const PackedDetectorShard &shard, DeviceLimits limits,
    std::uint64_t already_admitted_bytes, std::uint64_t staging_bytes);

/// Reference/recovery operation, not used in a production pointer-rate loop.
[[nodiscard]] std::vector<std::uint16_t> unpack_detector_shard(
    const PackedDetectorShard &shard);

struct CircularDetector {
  float center_row = 0;
  float center_column = 0;
  float inner_radius = 0;
  float outer_radius = 0;
};

/** Match c01c6ec macOS Float32 d² >= inner² && d² < outer².
 * Exclusions are an explicit authenticated mask, nonzero means excluded.
 * No interpolation, radius rounding, renormalization, or hidden source crop.
 */
[[nodiscard]] std::vector<std::uint8_t> circular_detector_mask(
    std::uint32_t rows, std::uint32_t columns, CircularDetector detector,
    std::span<const std::uint8_t> excluded = {});

struct DetectorPixelChange {
  std::uint32_t pixel;
  std::int32_t coefficient; // +1 includes, -1 subtracts; same 8-byte GLSL layout.
};
static_assert(sizeof(DetectorPixelChange) == 8);

struct PackedDetectorUpdate {
  bool rebase = true;
  std::vector<DetectorPixelChange> entries;
  std::uint64_t logical_source_bytes = 0;
};

/** Source-bound exact full-detector moments, eight little-endian u32 words per scan.
 *
 * Word order is total low/high, row moment low/high, column moment low/high,
 * then two zero padding words. Coordinates are zero-based and every nonexcluded
 * detector pixel participates. This validates the authenticated prepared range;
 * it never scans or materializes the dense 4D source.
 */
struct PreparedDpcMomentBounds {
  std::uint64_t total = 0;
  std::uint64_t row = 0;
  std::uint64_t column = 0;
};

[[nodiscard]] PreparedDpcMomentBounds validate_prepared_dpc_moments(
    Shape4D shape, std::span<const std::uint8_t> excluded_detector_pixels,
    std::span<const std::uint32_t> words);

/** Independent host oracle for tests only. Production DPC priming is Vulkan. */
[[nodiscard]] std::pair<std::vector<float>, std::vector<float>>
reference_prepared_dpc(std::span<const std::uint32_t> words);

/** Plan the cheaper exact full sum or mask difference, never an approximation.
 * Empty previous_mask forces rebase. Otherwise the caller MUST retain the exact
 * result associated with that mask, independent of newer requested UI geometry.
 * A cancelled/lost base must be rebased; never apply its delta to another result.
 * column_bytes is optional admitted payload+descriptor cost per detector pixel.
 */
[[nodiscard]] PackedDetectorUpdate plan_packed_detector_update(
    std::span<const std::uint8_t> previous_mask,
    std::span<const std::uint8_t> next_mask,
    std::span<const std::uint64_t> column_bytes = {});

/// Independent host arithmetic for tests. Production reduction is the GPU shader.
[[nodiscard]] std::vector<std::uint32_t> reference_packed_detector_update(
    const PackedDetectorShard &shard, const PackedDetectorUpdate &update,
    std::span<const std::uint32_t> previous_image = {});

} // namespace quantem::gpu::vulkan
