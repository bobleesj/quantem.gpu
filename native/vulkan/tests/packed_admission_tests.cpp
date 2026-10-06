#include "quantem/gpu/vulkan/packed_detector_session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// This executable requires an actual Vulkan runtime. It proves bounded
// synthetic integer parity, not real-file loading, presentation, or frame rate.
using namespace quantem::gpu::vulkan;

namespace {
constexpr std::uint32_t detector_rows = 5, detector_columns = 7;
constexpr std::uint32_t detector_pixels = detector_rows * detector_columns;
constexpr std::uint32_t scan_tile = 32, maximum_shard_scans = 8192;
constexpr std::uint64_t process_budget = 256ULL << 20;
constexpr std::uint64_t staging_budget = 8ULL << 20;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::uint32_t raw_value(std::uint32_t scan, std::uint32_t pixel) {
  const auto width = pixel % 17U;
  const auto maximum = (1U << width) - 1U;
  // A maximum at the start of EVERY tile guarantees widths 0 through 16.
  // Other values vary across both tile and shard boundaries.
  return scan % scan_tile == 0U
      ? maximum
      : (scan * 1664525U + pixel * 1013904223U) & maximum;
}

bool excluded(std::uint32_t pixel) { return pixel == 33U; }

std::vector<std::uint32_t> expanded_counts(std::uint32_t side) {
  std::vector<std::uint32_t> result{96U};
  auto remaining = side * side - result.front();
  while (remaining != 0U) {
    const auto count = std::min(remaining, maximum_shard_scans);
    result.push_back(count);
    remaining -= count;
  }
  return result;
}

std::vector<PackedDetectorShardSize>
expanded_plan(const std::vector<std::uint32_t> &counts) {
  std::uint32_t words_per_tile = 0;
  for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel)
    words_per_tile += pixel % 17U;
  std::vector<PackedDetectorShardSize> result;
  for (const auto count : counts) {
    const auto tiles = count / scan_tile;
    result.push_back({count, tiles * words_per_tile, 0U, 0U,
                      detector_pixels * tiles, scan_tile, 0U});
  }
  return result;
}

std::uint64_t source_bytes(std::span<const PackedDetectorShardSize> plan) {
  std::uint64_t result = 0;
  for (const auto &shard : plan)
    result += 4ULL * (shard.header_words + shard.payload_words);
  return result;
}

// Independent fixture writer: no production packer/decoder or dense source.
// Each loader invocation writes only its exact borrowed final destinations.
void write_expanded(std::uint32_t first_scan,
                    const PackedDetectorShardSize &plan,
                    PackedDetectorShardDestination destination) {
  require(destination.descriptors.size() == plan.header_words &&
              destination.words.size() == plan.payload_words,
          "Expanded loader destinations differ from the complete plan");
  std::fill(destination.words.begin(), destination.words.end(), 0U);
  const auto tiles = plan.scan_count / scan_tile;
  std::uint32_t cursor = 0;
  for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
    const auto width = pixel % 17U;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
      destination.descriptors[pixel * tiles + tile] = (cursor << 5U) | width;
      for (std::uint32_t sample = 0; sample < scan_tile && width != 0U;
           ++sample) {
        const auto value = raw_value(first_scan + tile * scan_tile + sample,
                                     pixel);
        const auto bit = sample * width;
        const auto word = cursor + bit / 32U;
        const auto shift = bit % 32U;
        destination.words[word] |= value << shift;
        if (shift + width > 32U)
          destination.words[word + 1U] |= value >> (32U - shift);
      }
      cursor += width;
    }
  }
  require(cursor == destination.words.size(),
          "Independent expanded fixture did not fill its exact payload");
}

std::array<bool, detector_pixels> detector_mask(CircularDetector detector) {
  std::array<bool, detector_pixels> mask{};
  const float inner = detector.inner_radius * detector.inner_radius;
  const float outer = detector.outer_radius * detector.outer_radius;
  for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel) {
    const float row = static_cast<float>(pixel / detector_columns) -
                      detector.center_row;
    const float column = static_cast<float>(pixel % detector_columns) -
                         detector.center_column;
    const float distance = row * row + column * column;
    mask[pixel] = !excluded(pixel) && distance >= inner && distance < outer;
  }
  return mask;
}

void check_image(std::span<const std::uint32_t> actual,
                 CircularDetector detector) {
  const auto mask = detector_mask(detector);
  for (std::uint32_t scan = 0; scan < actual.size(); ++scan) {
    std::uint64_t expected = 0;
    for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel)
      if (mask[pixel])
        expected += raw_value(scan, pixel);
    if (actual[scan] != expected)
      throw std::runtime_error("Expanded detector parity mismatch at scan " +
                               std::to_string(scan));
  }
}

void check_no_fft(const PackedDetectorSessionMetrics &metrics,
                  std::size_t shards, std::uint32_t scans) {
  require(metrics.timing.source_upload_bytes == 0U &&
              metrics.timing.device_wide_wait_count == 0U &&
              metrics.timing.output_copy_bytes == 4ULL * scans &&
              std::isnan(metrics.timing.gpu_fft_rows_milliseconds) &&
              std::isnan(metrics.timing.gpu_fft_columns_milliseconds) &&
              std::isnan(metrics.timing.gpu_magnitude_milliseconds),
          "Detector-only request uploaded a source or reported an FFT");
  require(metrics.timing.dispatch_count == (metrics.source_changed ? shards : 0U) &&
              metrics.timing.queue_submit_count == (metrics.source_changed ? 1U : 0U) &&
              metrics.timing.fence_wait_count == (metrics.source_changed ? 1U : 0U),
          "Detector-only dispatch/submit counts contain unexpected work");
}

void check_selected(PackedDetectorSession &session, std::uint32_t side,
                    std::uint32_t scan,
                    const std::vector<std::uint32_t> &counts) {
  std::uint32_t shard_index = 0, shard_first = 0;
  while (scan >= shard_first + counts[shard_index])
    shard_first += counts[shard_index++];
  std::array<std::uint32_t, detector_pixels> diffraction{};
  const auto metrics = session.selected_diffraction(
      scan / side, scan % side, UINT64_MAX, diffraction);
  require(metrics.generation == UINT64_MAX && metrics.row == scan / side &&
              metrics.column == scan % side && metrics.shard_index == shard_index &&
              metrics.shard_local_scan == scan - shard_first &&
              metrics.detector_rows == detector_rows &&
              metrics.detector_columns == detector_columns &&
              metrics.dispatch_count == 1U && metrics.queue_submit_count == 1U &&
              metrics.fence_wait_count == 1U && metrics.source_upload_bytes == 0U &&
              metrics.storage_read_bytes == 0U &&
              metrics.device_wide_wait_count == 0U &&
              metrics.output_copy_bytes == diffraction.size() * 4U,
          "Selected diffraction lost source coordinates or residency");
  for (std::uint32_t pixel = 0; pixel < detector_pixels; ++pixel)
    require(diffraction[pixel] == (excluded(pixel) ? 0U : raw_value(scan, pixel)),
            "Selected diffraction differs from independent uint16 values");
}

void expanded_admission(std::uint32_t side) {
  const auto counts = expanded_counts(side);
  const auto plan = expanded_plan(counts);
  std::vector<std::uint32_t> first_scans(counts.size());
  for (std::size_t index = 1; index < counts.size(); ++index)
    first_scans[index] = first_scans[index - 1U] + counts[index - 1U];
  std::array<std::uint8_t, detector_pixels> exclusions{};
  exclusions[33] = 1U;
  struct InjectedReadFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
  };
  std::uint32_t failed_loader_calls = 0;
  std::unique_ptr<PackedDetectorSession> unpublished;
  bool failed = false;
  try {
    unpublished = std::make_unique<PackedDetectorSession>(
        Shape4D{side, side, detector_rows, detector_columns}, plan,
        [](std::size_t, std::uint64_t) {},
        [&](std::size_t index, PackedDetectorShardDestination destination) {
          ++failed_loader_calls;
          if (index == 1U) {
            destination.descriptors.front() = UINT32_MAX;
            destination.words.front() = UINT32_MAX;
            throw InjectedReadFailure("Injected partial read/authentication failure");
          }
          write_expanded(first_scans[index], plan[index], destination);
        },
        exclusions, process_budget, staging_budget);
  } catch (const InjectedReadFailure &) {
    failed = true;
  }
  require(failed && !unpublished && failed_loader_calls == 2U,
          "Failed expanded load published a partial source or continued loading");

  std::uint32_t guard_calls = 0, loader_calls = 0;
  std::uint64_t completed_bytes = 0;
  PackedDetectorSession session(
      {side, side, detector_rows, detector_columns}, plan,
      [&](std::size_t index, std::uint64_t admitted_bytes) {
        require(index == guard_calls && index == loader_calls &&
                    admitted_bytes == completed_bytes,
                "Expanded source guard must precede each exact shard load");
        ++guard_calls;
      },
      [&](std::size_t index, PackedDetectorShardDestination destination) {
        require(guard_calls == loader_calls + 1U,
                "Expanded loader ran before its admission guard");
        write_expanded(first_scans[index], plan[index], destination);
        completed_bytes += destination.descriptors.size_bytes() +
                           destination.words.size_bytes();
        ++loader_calls;
      },
      exclusions, process_budget, staging_budget);
  const auto resident_bytes = session.admission().committed_bytes;
  require(loader_calls == plan.size() && guard_calls == plan.size() &&
              session.admission().shard_count == plan.size() &&
              session.admission().source_upload_bytes == source_bytes(plan),
          "Expanded source was not completely admitted after failed-load recovery");

  std::vector<std::uint32_t> image(std::size_t{side} * side);
  const std::array<CircularDetector, 9> detectors{{
      {2, 3, 0, 20},  // Initial full reconstruction.
      {2, 3, 1, 20},  // Remove one pixel: a strict signed delta.
      {2, 4, 1, 20},  // Translate the inner hole by one detector column.
      {2, 4, 1, 3},
      {1, 1, 0, 1.5F},
      {3, 5, 0, 1.5F}, // Disjoint BF translation must rebase.
      {2, 3, 1, 3.5F},
      {2, 3, 2, 3.5F}, // Resize an annulus by changing its inner radius.
      {2, 3, 2, 2.5F}, // Resize its outer radius independently.
  }};
  std::uint64_t generation = 0;
  for (std::size_t index = 0; index < detectors.size(); ++index) {
    const auto metrics = session.request(detectors[index], ++generation, image);
    require(metrics.committed_generation == generation,
            "Detector generation was changed by an interleaved selected DP");
    if (index == 0U || index == 5U)
      require(metrics.rebase, "Initial/disjoint detector requires an exact full rebase");
    if (index == 1U || index == 2U)
      require(!metrics.rebase && metrics.source_changed,
              "Small detector hole movement must use an exact signed delta");
    check_no_fft(metrics, plan.size(), side * side);
    check_image(image, detectors[index]);
    // Exercise a selected DP between each compute and repeat. A high DP
    // generation is not allowed to replace the committed detector base.
    check_selected(session, side, index == 0U ? 95U : 96U, counts);
    const auto repeated = session.request(detectors[index], ++generation, image);
    require(!repeated.source_changed && !repeated.rebase &&
                repeated.committed_generation == generation,
            "Unchanged detector did not reuse its committed result");
    check_no_fft(repeated, plan.size(), side * side);
    check_image(image, detectors[index]);
  }
  for (const auto scan : {0U, 31U, 32U, 63U, 64U, 95U, 96U, 127U,
                         128U, 96U + maximum_shard_scans - 1U,
                         96U + maximum_shard_scans, side * side - 1U})
    check_selected(session, side, scan, counts);
  require(session.admission().committed_bytes == resident_bytes &&
              session.admission().source_upload_bytes == completed_bytes &&
              loader_calls == plan.size(),
          "Expanded interactions grew resident storage or reloaded source data");
  std::cout << "PASS expanded-32-uint16-" << side
            << ": full scan, widths 0-16, movement/radii, full/delta/repeat, "
               "selected tile/shard edges, no FFT, failed-load recovery\n";
}

enum class CompactCorruption { none, leading_payload, tail_nibble, width, checkpoint };

std::vector<PackedDetectorShardSize> compact_plan(bool ones,
                                                CompactCorruption corruption) {
  std::vector<PackedDetectorShardSize> plan;
  for (const auto count : {96U, 512U * 512U - 96U}) {
    const auto tiles = count / scan_tile;
    const auto header_words = (tiles + 31U) / 32U + (tiles + 7U) / 8U;
    plan.push_back({count, ones ? tiles : 0U, 0U, 0U,
                    header_words, scan_tile, 1U});
  }
  if (corruption == CompactCorruption::leading_payload)
    ++plan.front().payload_words;
  if (corruption == CompactCorruption::width)
    plan.front().payload_words += 8U;
  return plan;
}

void write_compact(std::size_t index, const PackedDetectorShardSize &plan,
                   bool ones, CompactCorruption corruption,
                   PackedDetectorShardDestination destination) {
  require(destination.descriptors.size() == plan.header_words &&
              destination.words.size() == plan.payload_words,
          "Compact loader destinations differ from the complete plan");
  std::fill(destination.descriptors.begin(), destination.descriptors.end(), 0U);
  std::fill(destination.words.begin(), destination.words.end(), ones ? UINT32_MAX : 0U);
  const auto tiles = plan.scan_count / scan_tile;
  const auto checkpoints = (tiles + 31U) / 32U;
  if (ones) {
    for (std::uint32_t checkpoint = 1; checkpoint < checkpoints; ++checkpoint)
      destination.descriptors[checkpoint] = checkpoint * 32U;
    for (std::uint32_t tile = 0; tile < tiles; ++tile)
      destination.descriptors[checkpoints + tile / 8U] |=
          1U << ((tile % 8U) * 4U);
  }
  if (index == 0U) {
    if (corruption == CompactCorruption::leading_payload)
      destination.descriptors.front() = 1U;
    if (corruption == CompactCorruption::tail_nibble)
      destination.descriptors.back() |= 1U << ((tiles % 8U) * 4U);
    if (corruption == CompactCorruption::width)
      destination.descriptors[checkpoints] = 0x119U;
  }
  if (index == 1U && corruption == CompactCorruption::checkpoint)
    ++destination.descriptors[1];
}

void compact_admission() {
  for (const auto corruption : {CompactCorruption::leading_payload,
                                CompactCorruption::tail_nibble,
                                CompactCorruption::width,
                                CompactCorruption::checkpoint}) {
    const auto plan = compact_plan(true, corruption);
    bool rejected = false;
    std::unique_ptr<PackedDetectorSession> unpublished;
    try {
      unpublished = std::make_unique<PackedDetectorSession>(
          Shape4D{512, 512, 1, 1}, plan,
          [](std::size_t, std::uint64_t) {},
          [&](std::size_t index, PackedDetectorShardDestination destination) {
            write_compact(index, plan[index], true, corruption, destination);
          },
          std::span<const std::uint8_t>{}, 64ULL << 20, 1ULL << 20);
    } catch (const std::invalid_argument &error) {
      rejected = std::string_view(error.what()).find("GPU header validation") !=
                 std::string_view::npos;
    }
    require(rejected && !unpublished,
            "Malformed compact headers must fail GPU admission before publication");
  }
  std::cout << "PASS compact-malformed: compensated first base, tail nibble, "
               "invalid width, invalid checkpoint\n";

  for (const bool ones : {false, true}) {
    const auto plan = compact_plan(ones, CompactCorruption::none);
    std::uint32_t loader_calls = 0;
    PackedDetectorSession session(
        {512, 512, 1, 1}, plan, [](std::size_t, std::uint64_t) {},
        [&](std::size_t index, PackedDetectorShardDestination destination) {
          ++loader_calls;
          write_compact(index, plan[index], ones, CompactCorruption::none, destination);
        },
        {}, 64ULL << 20, 1ULL << 20);
    std::vector<std::uint32_t> image(512U * 512U, UINT32_MAX);
    const auto metrics = session.request({0, 0, 0, 1}, 1U, image);
    require(metrics.rebase && loader_calls == plan.size() &&
                session.admission().source_upload_bytes == source_bytes(plan) &&
                std::all_of(image.begin(), image.end(),
                            [ones](auto value) { return value == (ones ? 1U : 0U); }),
            "Valid compact zero/one fixture lost full-source integer parity");
    check_no_fft(metrics, plan.size(), 512U * 512U);
    for (const auto scan : {0U, 31U, 32U, 95U, 96U, 512U * 512U - 1U}) {
      std::array<std::uint32_t, 1> diffraction{UINT32_MAX};
      const auto selected = session.selected_diffraction(
          scan / 512U, scan % 512U, UINT64_MAX, diffraction);
      require(diffraction.front() == (ones ? 1U : 0U) &&
                  selected.shard_index == (scan < 96U ? 0U : 1U) &&
                  selected.shard_local_scan == (scan < 96U ? scan : scan - 96U) &&
                  selected.dispatch_count == 1U && selected.source_upload_bytes == 0U &&
                  selected.storage_read_bytes == 0U,
              "Compact selected diffraction lost zero/one or shard-edge parity");
    }
    const auto repeated = session.request({0, 0, 0, 1}, 2U, image);
    require(!repeated.source_changed && repeated.committed_generation == 2U &&
                loader_calls == plan.size(),
            "Compact repeat reread source or changed its committed detector");
    check_no_fft(repeated, plan.size(), 512U * 512U);
    std::cout << "PASS compact-" << (ones ? "ones" : "zero")
              << ": uneven 96+262048 scan shards, exact detector/DP, no FFT\n";
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view mode = argc == 2 ? argv[1] : "all";
    require(argc == 1 ||
                (argc == 2 && (mode == "--expanded-512-only" ||
                               mode == "--expanded-1024-only" ||
                               mode == "--compact-only")),
            "Use no arguments, --expanded-512-only, --expanded-1024-only, or --compact-only");
    if (mode == "all" || mode == "--expanded-512-only")
      expanded_admission(512);
    if (mode == "all" || mode == "--expanded-1024-only")
      expanded_admission(1024);
    if (mode == "all" || mode == "--compact-only")
      compact_admission();
    std::cout << "PASS packed-admission GPU synthetic parity; "
                 "real-file and physical presentation acceptance not measured\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL packed-admission: " << error.what() << '\n';
    return 1;
  }
}
