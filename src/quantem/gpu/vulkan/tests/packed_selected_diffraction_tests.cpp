#include "quantem/gpu/vulkan/packed_detector_session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

using namespace quantem::gpu::vulkan;
namespace {
constexpr std::uint32_t side = 512, scans = side*side;
// Two workgroups with an inactive tail, and non-square row/column geometry.
constexpr std::uint32_t detector_rows = 7, detector_columns = 19;
constexpr std::uint32_t pixels = detector_rows*detector_columns;
[[maybe_unused]] constexpr std::uint64_t staging_bytes = 4ULL<<20;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
std::uint16_t raw_value(std::uint32_t scan, std::uint32_t pixel) {
  const auto width = pixel % 17U;
  const auto mask = (1U << width)-1U;
  const auto value = scan % 128U == 0 ? mask :
      ((scan/side)*47831U + (scan%side)*293U + pixel*19381U) & mask;
  return static_cast<std::uint16_t>(value);
}
PackedDetectorShard make_shard(std::uint32_t first, std::uint32_t count) {
  std::vector<std::uint16_t> values(std::size_t{count}*pixels);
  for (std::uint32_t scan = 0; scan < count; ++scan)
    for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
      values[std::size_t{scan}*pixels+pixel] = raw_value(first+scan,pixel);
  return pack_detector_shard(values,count,pixels);
}
std::vector<std::uint8_t> exclusions() {
  std::vector<std::uint8_t> mask(pixels);
  mask[0] = 1; mask[16] = 20; mask[128] = 255; mask[pixels-1] = 1;
  return mask;
}
void exact_frame(std::span<const std::uint32_t> actual, std::uint32_t scan,
                 std::span<const std::uint8_t> excluded) {
  require(actual.size() == pixels, "Selected frame shape changed");
  for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
    require(actual[pixel] == (excluded.empty() || !excluded[pixel] ? raw_value(scan,pixel) : 0U),
            "Selected diffraction differs from independent raw uint16 formula/mask");
}
struct Fixture {
  std::vector<std::uint32_t> first, counts, selections;
  std::vector<PackedDetectorShardSize> plan;
  std::uint64_t source_bytes = 0, bound_source_bytes = 0;

  Fixture() {
    counts = {1,31,32,33,127,128,129,257};
    std::uint32_t covered = 0;
    for (auto count : counts) covered += count;
    while (covered < scans) {
      const auto count = std::min(4096U,scans-covered);
      counts.push_back(count); covered += count;
    }
    covered = 0;
    for (auto count : counts) {
      first.push_back(covered);
      const auto shard = make_shard(covered,count);
      plan.push_back({count,static_cast<std::uint32_t>(shard.words.size())});
      source_bytes += shard.descriptors.size()*4ULL+shard.words.size()*4ULL;
      bound_source_bytes += shard.descriptors.size()*4ULL+std::max(4ULL,shard.words.size()*4ULL);
      // Both sides of every irregular boundary and final source position.
      selections.push_back(covered); selections.push_back(covered+count-1);
      covered += count;
    }
    for (auto scan : {0U,side-1,(side-1)*side,scans-1,(side/2)*side+side/2})
      selections.push_back(scan);
    // Every position of a complete packed tile, including all cross-word offsets.
    for (std::uint32_t local = 0; local <= 129; ++local) selections.push_back(first[7]+local);
    std::uint32_t seed = 0x5a17c39U;
    for (std::uint32_t i = 0; i < 64; ++i) {
      seed = seed*1664525U+1013904223U;
      selections.push_back(seed%scans);
    }
  }
  std::size_t containing(std::uint32_t scan) const {
    for (std::size_t index = 0; index < first.size(); ++index)
      if (scan >= first[index] && scan-first[index] < counts[index]) return index;
    throw std::runtime_error("Independent fixture does not cover a selected scan");
  }
};

void reference_controls(const Fixture &fixture) {
  const auto excluded = exclusions();
  std::uint32_t widths = 0;
  bool cross_word = false, maximum = false;
  for (std::size_t index = 0; index < fixture.plan.size(); ++index) {
    const auto shard = make_shard(fixture.first[index],fixture.counts[index]);
    const auto tiles = (shard.scan_count+127)/128;
    for (const auto descriptor : shard.descriptors) widths |= 1U << (descriptor & 31U);
    for (const auto scan : fixture.selections) {
      if (fixture.containing(scan) != index) continue;
      const auto local = scan-fixture.first[index];
      std::vector<std::uint32_t> frame(pixels);
      for (std::uint32_t pixel = 0; pixel < pixels; ++pixel) {
        if (excluded[pixel]) continue;
        const auto descriptor = shard.descriptors[pixel*tiles+local/128];
        const auto width = descriptor & 31U;
        if (!width) continue;
        const auto bit = (local%128)*width;
        const auto offset = (descriptor >> 5U)+bit/32U;
        const auto shift = bit%32U;
        std::uint32_t value = shard.words.at(offset) >> shift;
        if (shift+width > 32U) {
          value |= shard.words.at(offset+1) << (32U-shift); cross_word = true;
        }
        frame[pixel] = value & ((1U << width)-1U);
        if (frame[pixel] == 65535U) maximum = true;
      }
      exact_frame(frame,scan,excluded);
    }
  }
  require(widths == (1U<<17)-1 && cross_word && maximum,
          "Fixture must cover packed widths0..16, cross-word reads and unmasked65535");
  std::cout << "{\"selectedDiffractionReferenceCases\":" << fixture.selections.size()
            << ",\"allWidths0Through16\":true,\"crossWord\":true,\"uint16Maximum\":true,\"gpuTested\":false}\n";
}

#ifndef PACKED_DETECTOR_REFERENCE_ONLY
void check_metrics(const PackedSelectedDiffractionMetrics &metrics, const Fixture &fixture,
                   std::uint32_t scan, std::uint64_t generation, bool timestamps) {
  const auto index = fixture.containing(scan);
  require(metrics.generation == generation && metrics.row == scan/side && metrics.column == scan%side &&
          metrics.detector_rows == detector_rows && metrics.detector_columns == detector_columns &&
          metrics.shard_index == index && metrics.shard_local_scan == scan-fixture.first[index],
          "Selected frame must preserve coordinates, shard-local index and opaque generation");
  require(metrics.dispatch_count == 1 && metrics.queue_submit_count == 1 && metrics.fence_wait_count == 1 &&
          metrics.device_wide_wait_count == 0 && metrics.output_copy_bytes == pixels*4 &&
          metrics.source_upload_bytes == 0 && metrics.storage_read_bytes == 0,
          "Selected DP must use one small output copy/dispatch/submit/fence and no source IO/upload");
  require(std::isfinite(metrics.wall_milliseconds) && metrics.wall_milliseconds >= 0 &&
          std::isfinite(metrics.mutex_wait_milliseconds) && metrics.mutex_wait_milliseconds >= 0 &&
          std::isfinite(metrics.output_copy_milliseconds) && metrics.output_copy_milliseconds >= 0 &&
          metrics.wall_milliseconds >= metrics.mutex_wait_milliseconds+metrics.output_copy_milliseconds &&
          (timestamps ? std::isfinite(metrics.gpu_decode_milliseconds) && metrics.gpu_decode_milliseconds >= 0 :
                        std::isnan(metrics.gpu_decode_milliseconds)),
          "Selected timing must distinguish measured wall/copy/GPU from unavailable GPU timestamps");
}

void gpu_controls(const Fixture &fixture) {
  std::uint32_t guards = 0, loads = 0;
  const auto guard = [&](std::size_t index, std::uint64_t) {
    require(index == guards && index == loads, "Guard/load order changed"); ++guards;
  };
  const auto loader = [&](std::size_t index, PackedDetectorShardDestination destination) {
    ++loads;
    const auto shard = make_shard(fixture.first[index],fixture.counts[index]);
    require(destination.descriptors.size() == shard.descriptors.size() && destination.words.size() == shard.words.size(),
            "Selected test loader receives exact admitted spans");
    std::copy(shard.descriptors.begin(),shard.descriptors.end(),destination.descriptors.begin());
    std::copy(shard.words.begin(),shard.words.end(),destination.words.begin());
  };
  const Shape4D shape{side,side,detector_rows,detector_columns};
  const auto excluded = exclusions();
  bool rejected = false;
  // An otherwise complete old work-buffer budget cannot admit the two new buffers.
  const auto old_work_bytes = 7ULL*scans*4+2048+pixels*8;
  try {
    PackedDetectorSession too_small(shape,fixture.plan,guard,loader,excluded,
        old_work_bytes+fixture.bound_source_bytes+staging_bytes,staging_bytes);
  } catch (const std::invalid_argument &error) {
    rejected = std::string_view(error.what()).find("reserved peak memory") != std::string_view::npos;
  }
  require(rejected && guards == 0 && loads == 0,
          "Selected mask/output must be included in whole-plan peak before any source guard/load");
  PackedDetectorSession session(shape,fixture.plan,guard,loader,excluded,256ULL<<20,staging_bytes);
  const auto admission = session.admission();
  std::uint64_t planned = 0;
  for (const auto &heap : admission.heaps) planned += heap.planned;
  require(planned == admission.committed_bytes &&
          planned >= old_work_bytes+fixture.bound_source_bytes+pixels*8 &&
          admission.source_upload_bytes == fixture.source_bytes,
          "Actual aligned workspace/source allocations must match the whole admitted plan");
  std::vector<std::uint32_t> frame(pixels,UINT32_MAX), retained(pixels,UINT32_MAX);
  // Selected DP before any detector request must not establish a detector base.
  const auto first = session.selected_diffraction(0,0,UINT64_MAX,retained);
  check_metrics(first,fixture,0,UINT64_MAX,admission.timestamps_available);
  exact_frame(retained,0,excluded);
  const auto expected_retained = retained;
  const std::array<std::uint64_t,6> generations{0,42,42,1,UINT64_MAX,0};
  std::size_t ordinal = 0;
  for (const auto scan : fixture.selections) {
    const auto generation = generations[ordinal++%generations.size()];
    std::fill(frame.begin(),frame.end(),UINT32_MAX);
    const auto metrics = session.selected_diffraction(scan/side,scan%side,generation,frame);
    check_metrics(metrics,fixture,scan,generation,admission.timestamps_available);
    exact_frame(frame,scan,excluded);
  }
  require(retained == expected_retained, "Later selections must not mutate a caller-owned completed frame");
  std::uint32_t invalid_cases = 0;
  for (const auto position : std::array<std::array<std::uint32_t,2>,4>{{{side,0},{0,side},{UINT32_MAX,0},{0,UINT32_MAX}}}) {
    std::fill(frame.begin(),frame.end(),UINT32_MAX); rejected = false;
    try { (void)session.selected_diffraction(position[0],position[1],0,frame); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && std::all_of(frame.begin(),frame.end(),[](auto value) { return value == UINT32_MAX; }),
            "Invalid scan coordinates must not write the caller output or dispatch");
    ++invalid_cases;
  }
  for (const auto size : {0U,pixels-1,pixels+1}) {
    std::vector<std::uint32_t> bad(size,UINT32_MAX); rejected = false;
    try { (void)session.selected_diffraction(0,0,0,bad); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && std::all_of(bad.begin(),bad.end(),[](auto value) { return value == UINT32_MAX; }),
            "Invalid destination length must fail without output writes or poisoning the session");
    ++invalid_cases;
  }
  // Concurrent callers use separate caller storage; the shared command/fence is
  // serialized internally. Each returned result must retain its own identity.
  std::array<std::exception_ptr,2> failures{};
  std::array<std::jthread,2> callers;
  for (std::size_t worker = 0; worker < callers.size(); ++worker) {
    callers[worker] = std::jthread([&,worker] {
      try {
        std::vector<std::uint32_t> output(pixels);
        for (std::uint32_t i = 0; i < 16; ++i) {
          const auto scan = static_cast<std::uint32_t>((worker*19273+i*977)%scans);
          const auto metrics = session.selected_diffraction(scan/side,scan%side,i,output);
          check_metrics(metrics,fixture,scan,i,admission.timestamps_available);
          exact_frame(output,scan,excluded);
        }
      } catch (...) { failures[worker] = std::current_exception(); }
    });
  }
  for (auto &caller : callers) caller.join();
  for (const auto &failure : failures) if (failure) std::rethrow_exception(failure);
  require(loads == fixture.plan.size() && guards == fixture.plan.size() &&
          session.admission().source_upload_bytes == admission.source_upload_bytes &&
          session.admission().committed_bytes == admission.committed_bytes,
          "Selected requests must neither reload source nor allocate additional GPU workspace");
  std::cout << "{\"selectedDiffractionGpuCases\":" << fixture.selections.size()+1+32
            << ",\"invalidCases\":" << invalid_cases << ",\"shards\":" << fixture.plan.size()
            << ",\"integerExact\":true,\"gpuTested\":true,\"syntheticOnly\":true,\"callerOwned\":true}\n";
}
#endif
} // namespace

int main() {
  try {
    const Fixture fixture;
    reference_controls(fixture);
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
    gpu_controls(fixture);
#endif
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Packed selected diffraction test failed: " << error.what() << '\n';
    return 1;
  }
}
