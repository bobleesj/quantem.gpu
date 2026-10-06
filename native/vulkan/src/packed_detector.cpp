#include "quantem/gpu/vulkan/packed_detector.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

// The product oracle evaluates Float32 products/addition separately. In
// particular, an FMA on a fractional aperture boundary can change membership.
#pragma STDC FP_CONTRACT OFF

namespace quantem::gpu::vulkan {
namespace {
constexpr std::uint64_t maximum_words = std::uint64_t{1} << 27;

std::uint32_t tile_count(std::uint32_t scans) {
  return scans / PackedDetectorShard::scan_tile +
         (scans % PackedDetectorShard::scan_tile != 0);
}

void validate_shape(std::uint32_t scans, std::uint32_t pixels) {
  if (!scans || !pixels)
    throw std::invalid_argument("Packed detector requires nonzero scan and detector counts");
  if (std::uint64_t{pixels} * 65535 > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Detector can overflow exact uint32 sums; use a wide-count backend");
}

std::uint16_t sample(const PackedDetectorShard &shard, std::uint32_t scan,
                     std::uint32_t pixel) {
  const auto descriptor = shard.descriptors[
      std::size_t{pixel} * tile_count(shard.scan_count) + scan / 128];
  const auto width = descriptor & 31U;
  if (!width) return 0;
  const auto bit = (scan % 128) * width;
  const auto index = (descriptor >> 5) + bit / 32;
  const auto shift = bit % 32;
  auto value = shard.words[index] >> shift;
  if (shift + width > 32) value |= shard.words[index + 1] << (32 - shift);
  return static_cast<std::uint16_t>(value & ((1U << width) - 1));
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    throw std::invalid_argument("Detector traffic estimate exceeds uint64");
  return a + b;
}

std::uint64_t word_pair(std::span<const std::uint32_t> words,
                        std::size_t offset) {
  return std::uint64_t{words[offset]} |
         (std::uint64_t{words[offset + 1]} << 32U);
}
} // namespace

PackedDetectorShard pack_detector_shard(std::span<const std::uint16_t> input,
                                        std::uint32_t scans, std::uint32_t pixels,
                                        std::uint64_t maximum_packed_bytes) {
  validate_shape(scans, pixels);
  if (std::uint64_t{scans} * pixels != input.size())
    throw std::invalid_argument("Raw uint16 input must match [scan_count, detector_pixels]");
  const auto tiles = tile_count(scans);
  const auto descriptor_bytes = std::uint64_t{pixels} * tiles * 4;
  if (descriptor_bytes > maximum_packed_bytes)
    throw std::invalid_argument("Packed descriptors exceed the shard budget; use smaller scan shards");
  PackedDetectorShard shard{scans, pixels, {}, {}};
  shard.descriptors.resize(std::size_t{pixels} * tiles);
  std::uint64_t cursor = 0;
  for (std::uint32_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
      const auto start = tile * 128;
      const auto stop = std::min(std::uint64_t{start} + 128, std::uint64_t{scans});
      std::uint16_t combined = 0;
      for (auto scan = start; scan < stop; ++scan)
        combined |= input[std::size_t{scan} * pixels + pixel];
      const auto width = std::bit_width(combined);
      if (cursor >= maximum_words || cursor + width * 4 > maximum_words)
        throw std::invalid_argument("Packed shard exceeds offset range; use smaller scan shards");
      shard.descriptors[std::size_t{pixel} * tiles + tile] =
          (static_cast<std::uint32_t>(cursor) << 5) | width;
      cursor += width * 4;
      if (descriptor_bytes + cursor * 4 > maximum_packed_bytes)
        throw std::invalid_argument("Packed payload exceeds the shard budget; use smaller scan shards");
    }
  }
  // One exact-sized payload allocation, not geometric vector growth or a full
  // second source copy. The caller must still budget this shard's raw staging.
  shard.words.resize(static_cast<std::size_t>(cursor), 0);
  for (std::uint32_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
      const auto descriptor = shard.descriptors[std::size_t{pixel} * tiles + tile];
      const auto width = descriptor & 31U;
      if (!width) continue;
      const auto start = tile * 128;
      const auto stop = std::min(std::uint64_t{start} + 128, std::uint64_t{scans});
      for (auto scan = start; scan < stop; ++scan) {
        const auto value = std::uint32_t{input[std::size_t{scan} * pixels + pixel]};
        const auto bit = (scan - start) * width;
        const auto index = (descriptor >> 5) + bit / 32;
        const auto shift = bit % 32;
        shard.words[index] |= value << shift;
        if (shift + width > 32) shard.words[index + 1] |= value >> (32 - shift);
      }
    }
  }
  return shard;
}

void validate_packed_detector_shard(const PackedDetectorShard &shard) {
  validate_packed_detector_shard(shard.scan_count, shard.detector_pixels,
                                shard.descriptors, shard.words);
}

void validate_packed_detector_shard(std::uint32_t scans, std::uint32_t pixels,
                                  std::span<const std::uint32_t> descriptors,
                                  std::span<const std::uint32_t> words) {
  validate_packed_detector_shard(scans, pixels, descriptors, words,
                                PackedDetectorShard::scan_tile);
}

void validate_packed_detector_shard(std::uint32_t scans, std::uint32_t pixels,
                                  std::span<const std::uint32_t> descriptors,
                                  std::span<const std::uint32_t> words,
                                  std::uint32_t scan_tile) {
  validate_shape(scans, pixels);
  if (scan_tile != 32U && scan_tile != 128U)
    throw std::invalid_argument("Expanded packed descriptors require a 32- or 128-scan tile");
  const auto tiles = scans / scan_tile + (scans % scan_tile != 0U);
  if (descriptors.size() != std::size_t{pixels} * tiles)
    throw std::invalid_argument("Packed detector descriptor count does not match shape");
  std::uint64_t cursor = 0;
  for (const auto descriptor : descriptors) {
    const auto width = descriptor & 31U;
    if (width > 16 || (descriptor >> 5) != cursor)
      throw std::invalid_argument("Packed detector width or canonical word offset is invalid");
    cursor += width * (scan_tile / 32U);
    if (cursor > words.size() || cursor > maximum_words)
      throw std::invalid_argument("Packed detector payload is truncated or exceeds offset range");
  }
  if (cursor != words.size())
    throw std::invalid_argument("Packed detector payload has trailing words");
}

std::uint64_t admit_packed_detector_shard(
    const PackedDetectorShard &shard, DeviceLimits limits,
    std::uint64_t already_admitted_bytes, std::uint64_t staging_bytes) {
  validate_packed_detector_shard(shard);
  // Vulkan still needs a valid bound buffer for an all-zero payload.
  const auto payload_bytes = std::max(std::uint64_t{4}, std::uint64_t{shard.words.size()} * 4);
  const auto descriptor_bytes = std::uint64_t{shard.descriptors.size()} * 4;
  for (const auto bytes : {payload_bytes, descriptor_bytes}) {
    if (bytes > limits.max_storage_buffer_range_bytes ||
        bytes > limits.max_memory_allocation_bytes)
      throw std::invalid_argument("Packed shard exceeds Vulkan buffer limits; use smaller scan shards");
  }
  const auto resident = checked_add(already_admitted_bytes,
                                   checked_add(payload_bytes, descriptor_bytes));
  if (checked_add(resident, staging_bytes) > limits.available_process_bytes)
    throw std::invalid_argument("Packed source exceeds reserved process memory; do not promote partial residency");
  return resident;
}

std::vector<std::uint16_t> unpack_detector_shard(const PackedDetectorShard &shard) {
  validate_packed_detector_shard(shard);
  std::vector<std::uint16_t> result(std::size_t{shard.scan_count} * shard.detector_pixels);
  for (std::uint32_t scan = 0; scan < shard.scan_count; ++scan)
    for (std::uint32_t pixel = 0; pixel < shard.detector_pixels; ++pixel)
      result[std::size_t{scan} * shard.detector_pixels + pixel] = sample(shard, scan, pixel);
  return result;
}

std::vector<std::uint8_t> circular_detector_mask(
    std::uint32_t rows, std::uint32_t columns, CircularDetector detector,
    std::span<const std::uint8_t> excluded) {
  const auto count = std::uint64_t{rows} * columns;
  if (!rows || !columns || count > std::numeric_limits<std::uint32_t>::max() ||
      (!excluded.empty() && excluded.size() != count))
    throw std::invalid_argument("Detector shape/exclusion mask size is invalid");
  if (!std::isfinite(detector.center_row) || !std::isfinite(detector.center_column) ||
      !std::isfinite(detector.inner_radius) || !std::isfinite(detector.outer_radius) ||
      detector.inner_radius < 0 || detector.outer_radius < detector.inner_radius)
    throw std::invalid_argument("Use finite detector coordinates and 0 <= inner <= outer radii");
  const float inner2 = detector.inner_radius * detector.inner_radius;
  const float outer2 = detector.outer_radius * detector.outer_radius;
  if (!std::isfinite(inner2) || !std::isfinite(outer2))
    throw std::invalid_argument("Detector radius square exceeds Float32; use finite pixel radii");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float dr = static_cast<float>(row) - detector.center_row;
    for (std::uint32_t column = 0; column < columns; ++column) {
      const auto pixel = std::size_t{row} * columns + column;
      const float dc = static_cast<float>(column) - detector.center_column;
      const float radius2 = dr * dr + dc * dc;
      result[pixel] = (excluded.empty() || excluded[pixel] == 0) &&
                      radius2 >= inner2 && radius2 < outer2;
    }
  }
  return result;
}

PackedDetectorUpdate plan_packed_detector_update(
    std::span<const std::uint8_t> previous, std::span<const std::uint8_t> next,
    std::span<const std::uint64_t> column_bytes) {
  if (next.empty() || (!previous.empty() && previous.size() != next.size()) ||
      (!column_bytes.empty() && column_bytes.size() != next.size()) ||
      next.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Detector masks and optional column costs must have matching sizes");
  PackedDetectorUpdate delta{false, {}, 0};
  std::uint64_t full_logical_source_bytes = 0;
  std::size_t full_entry_count = 0;
  for (std::size_t pixel = 0; pixel < next.size(); ++pixel) {
    if (next[pixel] > 1 || (!previous.empty() && previous[pixel] > 1))
      throw std::invalid_argument("Detector selection masks must contain only zero or one");
    const auto cost = column_bytes.empty() ? 1 : column_bytes[pixel];
    if (next[pixel]) {
      ++full_entry_count;
      full_logical_source_bytes = checked_add(full_logical_source_bytes, cost);
    }
    if (!previous.empty() && previous[pixel] != next[pixel]) {
      delta.entries.push_back({static_cast<std::uint32_t>(pixel), next[pixel] ? 1 : -1});
      delta.logical_source_bytes = checked_add(delta.logical_source_bytes, cost);
    }
  }
  // Without byte costs the decision is based on selected column count; do not
  // mislabel that count as measured bytes in public telemetry.
  const bool choose_full = previous.empty() ||
      full_logical_source_bytes < delta.logical_source_bytes;
  if (!choose_full) {
    if (column_bytes.empty()) delta.logical_source_bytes = 0;
    return delta;
  }

  // Pointer-rate translations almost always choose the small delta. Do not
  // allocate and fill the complete disk/annulus entry list unless a rebase
  // actually wins the exact cost comparison.
  PackedDetectorUpdate full{true, {},
      column_bytes.empty() ? 0 : full_logical_source_bytes};
  full.entries.reserve(full_entry_count);
  for (std::size_t pixel = 0; pixel < next.size(); ++pixel) {
    if (next[pixel]) full.entries.push_back({static_cast<std::uint32_t>(pixel), 1});
  }
  return full;
}

PreparedDpcMomentBounds validate_prepared_dpc_moments(
    Shape4D shape, std::span<const std::uint8_t> excluded,
    std::span<const std::uint32_t> words) {
  const auto scans = std::uint64_t{shape.scan_rows} * shape.scan_columns;
  const auto pixels = std::uint64_t{shape.detector_rows} *
                      shape.detector_columns;
  if (!scans || !pixels || scans > std::numeric_limits<std::size_t>::max() / 8U ||
      words.size() != static_cast<std::size_t>(scans) * 8U ||
      (!excluded.empty() && excluded.size() != pixels))
    throw std::invalid_argument(
        "Prepared DPC moments do not match the admitted scan and detector shape");
  std::uint64_t selected = 0, row_coordinates = 0, column_coordinates = 0;
  for (std::uint64_t pixel = 0; pixel < pixels; ++pixel) {
    if (!excluded.empty() && excluded[pixel] != 0U)
      continue;
    ++selected;
    row_coordinates = checked_add(
        row_coordinates, pixel / shape.detector_columns);
    column_coordinates = checked_add(
        column_coordinates, pixel % shape.detector_columns);
  }
  const auto multiply = [](std::uint64_t value, std::uint64_t factor) {
    if (value && factor > UINT64_MAX / value)
      throw std::invalid_argument("Prepared DPC theoretical bound exceeds uint64");
    return value * factor;
  };
  const PreparedDpcMomentBounds bounds{
      multiply(selected, 255U), multiply(row_coordinates, 255U),
      multiply(column_coordinates, 255U)};
  for (std::size_t scan = 0; scan < scans; ++scan) {
    const auto offset = scan * 8U;
    const auto total = word_pair(words, offset);
    const auto row = word_pair(words, offset + 2U);
    const auto column = word_pair(words, offset + 4U);
    if (words[offset + 6U] != 0U || words[offset + 7U] != 0U)
      throw std::invalid_argument("Prepared DPC padding words must be zero");
    if (total > bounds.total || row > bounds.row || column > bounds.column ||
        (total == 0U && (row != 0U || column != 0U)))
      throw std::invalid_argument(
          "Prepared DPC moments violate exact source-derived bounds");
  }
  return bounds;
}

std::pair<std::vector<float>, std::vector<float>> reference_prepared_dpc(
    std::span<const std::uint32_t> words) {
  if (words.size() % 8U != 0U)
    throw std::invalid_argument("Prepared DPC oracle requires eight words per scan");
  const auto scans = words.size() / 8U;
  if (!scans)
    throw std::invalid_argument("Prepared DPC oracle requires at least one scan");
  std::vector<float> row(scans), column(scans);
  double row_sum = 0, column_sum = 0;
  for (std::size_t scan = 0; scan < scans; ++scan) {
    const auto offset = scan * 8U;
    const auto total = word_pair(words, offset);
    if (!total)
      continue;
    row[scan] = static_cast<float>(
        static_cast<double>(word_pair(words, offset + 2U)) /
        static_cast<double>(total));
    column[scan] = static_cast<float>(
        static_cast<double>(word_pair(words, offset + 4U)) /
        static_cast<double>(total));
    row_sum += row[scan];
    column_sum += column[scan];
  }
  const auto row_mean = static_cast<float>(row_sum / scans);
  const auto column_mean = static_cast<float>(column_sum / scans);
  for (std::size_t scan = 0; scan < scans; ++scan) {
    row[scan] -= row_mean;
    column[scan] -= column_mean;
  }
  return {std::move(row), std::move(column)};
}

std::vector<std::uint32_t> reference_packed_detector_update(
    const PackedDetectorShard &shard, const PackedDetectorUpdate &update,
    std::span<const std::uint32_t> previous) {
  validate_packed_detector_shard(shard);
  if (!update.rebase && previous.size() != shard.scan_count)
    throw std::invalid_argument("A delta requires its exact previous image; otherwise request rebase");
  std::uint64_t last = 0;
  bool first = true;
  for (const auto entry : update.entries) {
    if (entry.pixel >= shard.detector_pixels || (!first && entry.pixel <= last) ||
        (entry.coefficient != 1 && entry.coefficient != -1) ||
        (update.rebase && entry.coefficient != 1))
      throw std::invalid_argument("Detector entries must be unique, sorted, in range and +/-1");
    first = false;
    last = entry.pixel;
  }
  std::vector<std::uint32_t> result(shard.scan_count);
  for (std::uint32_t scan = 0; scan < shard.scan_count; ++scan) {
    std::int64_t sum = update.rebase ? 0 : previous[scan];
    for (const auto entry : update.entries)
      sum += std::int64_t{sample(shard, scan, entry.pixel)} * entry.coefficient;
    if (sum < 0 || sum > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("Delta base or count range is invalid; rebase from the exact source");
    result[scan] = static_cast<std::uint32_t>(sum);
  }
  return result;
}
} // namespace quantem::gpu::vulkan
