#include "quantem/gpu/vulkan/packed_detector_session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>

#ifndef PACKED_DETECTOR_REFERENCE_ONLY
#include <vulkan/vulkan.h>
#endif

using namespace quantem::gpu::vulkan;
namespace {
constexpr std::uint32_t side = 512, scans = side*side, pixels = 35;
[[maybe_unused]] constexpr std::uint32_t shard_scans = 4096;
constexpr double relative_l2_limit = 2e-6, magnitude_relative_limit = 2e-5, input_l2_limit = 4e-6;
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
std::uint16_t raw_value(std::uint32_t scan, std::uint32_t pixel) {
  if (pixel == 0 || pixel == pixels-1) return 65535;
  const auto row = scan / side, column = scan % side;
  return static_cast<std::uint16_t>((row*37 + column*73 + pixel*11 + row*column*11%128) % 30);
}
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
void copy_shard(const PackedDetectorShard &shard, PackedDetectorShardDestination destination) {
  require(destination.descriptors.size() == shard.descriptors.size() &&
          destination.words.size() == shard.words.size(), "Loader must receive exact admitted spans");
  std::copy(shard.descriptors.begin(), shard.descriptors.end(), destination.descriptors.begin());
  std::copy(shard.words.begin(), shard.words.end(), destination.words.begin());
}

void check_source_memory(const PackedDetectorSourceMemory &memory, std::uint64_t logical_bytes) {
  require(memory.allocation_count == 1 && memory.allocated_bytes >= std::max<std::uint64_t>(4,logical_bytes) &&
          (memory.property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
          memory.memory_type_index < VK_MAX_MEMORY_TYPES && memory.heap_index < VK_MAX_MEMORY_HEAPS,
          "Destination must describe its actual host-visible aligned source allocation");
}

PackedDetectorShard synthetic_shard(std::size_t index) {
  std::vector<std::uint16_t> raw(shard_scans*pixels);
  for (std::uint32_t scan = 0; scan < shard_scans; ++scan)
    for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
      raw[scan*pixels+pixel] = raw_value(static_cast<std::uint32_t>(index)*shard_scans+scan,pixel);
  return pack_detector_shard(raw,shard_scans,pixels);
}
void number(double value) { if (std::isfinite(value)) std::cout << value; else std::cout << "null"; }

void check_selected_raw(PackedDetectorSession &session, std::uint32_t scan,
                        std::span<const std::uint8_t> excluded) {
  std::vector<std::uint32_t> diffraction(pixels,UINT32_MAX);
  // A high/dropped DP generation must neither advance nor replace detector state.
  const auto metrics = session.selected_diffraction(scan/side,scan%side,UINT64_MAX,diffraction);
  require(metrics.generation == UINT64_MAX && metrics.row == scan/side && metrics.column == scan%side &&
          metrics.shard_index == scan/shard_scans && metrics.shard_local_scan == scan%shard_scans &&
          metrics.dispatch_count == 1 && metrics.queue_submit_count == 1 && metrics.fence_wait_count == 1 &&
          metrics.output_copy_bytes == pixels*4 && metrics.source_upload_bytes == 0 &&
          metrics.storage_read_bytes == 0 && metrics.device_wide_wait_count == 0,
          "Interleaved selected DP must retain identity and use only a single bounded decode");
  for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
    require(diffraction[pixel] == (excluded[pixel] ? 0U : raw_value(scan,pixel)),
            "Interleaved selected DP differs from independent masked raw values");
}

void destination_loader_failures_and_zero_payload() {
  constexpr std::uint32_t count = scans / 4, tiles = count / 128, word_count = tiles * 4;
  constexpr std::uint64_t shard_bytes = (tiles + word_count) * 4ULL;
  const std::vector<PackedDetectorShardSize> plan(4, {count,word_count});
  const auto fill_ones = [](PackedDetectorShardDestination destination) {
    require(destination.descriptors.size() == tiles, "Exact descriptor span from admitted scan count");
    check_source_memory(destination.descriptor_memory,destination.descriptors.size_bytes());
    check_source_memory(destination.payload_memory,destination.words.size_bytes());
    for (std::uint32_t tile = 0; tile < tiles; ++tile) destination.descriptors[tile] = (tile*4U << 5U) | 1U;
    std::fill(destination.words.begin(),destination.words.end(),UINT32_MAX);
  };
  struct InjectedFailure : std::runtime_error { using std::runtime_error::runtime_error; };
  for (const bool fail_in_guard : {true,false}) {
    for (const auto stop : {0U,2U,3U}) {
      std::uint32_t guard_calls = 0, loader_calls = 0;
      const auto guard = [&](std::size_t index, std::uint64_t admitted_bytes) {
        require(index == guard_calls && index == loader_calls && admitted_bytes == index*shard_bytes,
                "Guard must precede each shard loader and count only earlier completed shards");
        ++guard_calls;
        if (fail_in_guard && index == stop) throw InjectedFailure("guard rejected source allocation");
      };
      const auto loader = [&](std::size_t index, PackedDetectorShardDestination destination) {
        require(guard_calls == loader_calls+1, "Mandatory guard must run before destination loading");
        ++loader_calls;
        if (!fail_in_guard && index == stop) {
          // Model a short/failed read or failed authentication after partial writes.
          destination.descriptors.front() = 1;
          destination.words.front() = 65535;
          throw InjectedFailure("partial write failed authentication");
        }
        fill_ones(destination);
      };
      std::unique_ptr<PackedDetectorSession> published;
      bool rejected = false;
      try {
        published = std::make_unique<PackedDetectorSession>(Shape4D{side,side,1,1},plan,
            guard,loader,std::span<const std::uint8_t>{},32ULL<<20,1ULL<<20);
      } catch (const InjectedFailure &) { rejected = true; }
      require(rejected && !published && guard_calls == stop+1 &&
              loader_calls == stop+(fail_in_guard ? 0 : 1),
              "Guard/partial-write failure must abort construction without another load or a session");
    }
  }
  std::uint32_t unexpected_loader_calls = 0;
  const auto unexpected_loader = [&](std::size_t, PackedDetectorShardDestination) {
    ++unexpected_loader_calls;
  };
  bool rejected = false;
  try { PackedDetectorSession missing_guard({side,side,1,1},plan,{},unexpected_loader,{},32ULL<<20,1ULL<<20); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected && unexpected_loader_calls == 0, "A missing pre-allocation guard is invalid");

  // Destination extents come from the immutable plan. Bad payload lengths must
  // still fail the same canonical descriptor validation as the owning codec.
  for (const auto malformed : {0U,1U,2U,3U}) {
    auto bad_plan = plan;
    if (malformed == 2) --bad_plan[0].payload_words;
    if (malformed == 3) ++bad_plan[0].payload_words;
    std::uint32_t loader_calls = 0, guard_calls = 0;
    const auto guard = [&](std::size_t index, std::uint64_t admitted_bytes) {
      require(index == 0 && admitted_bytes == 0, "Malformed first shard cannot advance admission");
      ++guard_calls;
    };
    const auto loader = [&](std::size_t index, PackedDetectorShardDestination destination) {
      require(index == 0 && destination.words.size() == bad_plan[0].payload_words,
              "Payload extent must match the whole-source plan exactly");
      ++loader_calls;
      fill_ones(destination);
      if (malformed == 0) destination.descriptors[0] = 17;
      if (malformed == 1) destination.descriptors[1] += 32;
    };
    rejected = false;
    try { PackedDetectorSession bad({side,side,1,1},bad_plan,guard,loader,{},32ULL<<20,1ULL<<20); }
    catch (const std::invalid_argument &error) {
      const std::string_view message(error.what());
      rejected = message.find(malformed < 2 ? "canonical" : malformed == 2 ? "truncated" : "trailing") !=
                 std::string_view::npos;
    }
    require(rejected && loader_calls == 1 && guard_calls == 1,
            "Malformed descriptors or payload length must reject the unpublished shard");
  }

  // A successful construction after all injected failures also exercises cleanup
  // and a logical-empty payload with a valid private Vulkan binding.
  const std::vector<PackedDetectorShardSize> zero_plan(4, {count,0});
  std::uint32_t guard_calls = 0, loader_calls = 0;
  std::uint64_t allocated_source_bytes = 0;
  const auto guard = [&](std::size_t index, std::uint64_t admitted_bytes) {
    require(index == guard_calls && index == loader_calls && admitted_bytes == index*tiles*4ULL,
            "All-zero logical byte accounting excludes the private dummy bindings");
    ++guard_calls;
  };
  const auto loader = [&](std::size_t, PackedDetectorShardDestination destination) {
    ++loader_calls;
    require(destination.words.empty() && destination.descriptors.size() == tiles,
            "All-zero payload must expose an empty exact logical span");
    check_source_memory(destination.descriptor_memory,destination.descriptors.size_bytes());
    check_source_memory(destination.payload_memory,0);
    allocated_source_bytes += destination.descriptor_memory.allocated_bytes + destination.payload_memory.allocated_bytes;
    std::fill(destination.descriptors.begin(),destination.descriptors.end(),0);
  };
  PackedDetectorSession zero({side,side,1,1},zero_plan,guard,loader,{},32ULL<<20,1ULL<<20);
  const auto &admission = zero.admission();
  std::uint64_t diagnostic_bytes = 0;
  std::uint32_t diagnostic_allocations = 0;
  for (const auto &memory : admission.source_memory) {
    require(memory.property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            "Aggregated source properties must preserve host visibility");
    diagnostic_bytes += memory.allocated_bytes;
    diagnostic_allocations += memory.allocation_count;
  }
  require(guard_calls == 4 && loader_calls == 4 && admission.source_upload_bytes == 4ULL*tiles*4 &&
          admission.reserved_staging_bytes == (1ULL<<20) && admission.load_timing.source_copy_milliseconds == 0 &&
          diagnostic_allocations == 8 && diagnostic_bytes == allocated_source_bytes &&
          diagnostic_bytes < admission.committed_bytes,
          "Source allocation diagnostics must match actual destination allocations, not work buffers");
  std::vector<std::uint32_t> image(scans,UINT32_MAX);
  std::vector<float> fft(scans,1);
  std::array<std::uint32_t,1> diffraction{UINT32_MAX};
  const auto selected = zero.selected_diffraction(511,511,UINT64_MAX,diffraction);
  require(diffraction[0] == 0 && selected.dispatch_count == 1 && selected.source_upload_bytes == 0 &&
          selected.storage_read_bytes == 0 && selected.output_copy_bytes == 4,
          "Logical-empty source must yield a zero selected DP without dereferencing empty logical payload");
  const auto metrics = zero.request({0,0,0,1},1,image,fft);
  require(metrics.rebase && metrics.timing.source_upload_bytes == 0 &&
          std::all_of(image.begin(),image.end(),[](auto value) { return value == 0; }) &&
          std::all_of(fft.begin(),fft.end(),[](auto value) { return value == 0; }),
          "Logical-empty source must yield exact zero image and FFT through valid GPU bindings");
  std::cout << "{\"destinationLoaderFailures\":11,\"allZeroPayload\":true,\"integerExact\":true,\"gpuTested\":true}\n";
}

void compact_header_gpu_validation() {
  constexpr std::uint32_t compact_shards = 4;
  constexpr std::uint32_t scans_per_compact_shard = scans / compact_shards;
  constexpr std::uint32_t scan_tile = 32;
  constexpr std::uint32_t tiles = scans_per_compact_shard / scan_tile;
  constexpr std::uint32_t checkpoint_words = (tiles + 31) / 32;
  constexpr std::uint32_t width_words = (tiles + 7) / 8;
  constexpr std::uint32_t header_words = checkpoint_words + width_words;
  constexpr std::uint32_t payload_words = tiles;
  const auto canonical_headers = [] {
    std::array<std::uint32_t, header_words> result{};
    for (std::uint32_t checkpoint = 1; checkpoint < checkpoint_words;
         ++checkpoint)
      result[checkpoint] = checkpoint * 32U;
    std::fill(result.begin() + checkpoint_words, result.end(),
              0x11111111U);
    return result;
  }();
  const auto make_plan = [](std::uint32_t words = payload_words) {
    return std::vector<PackedDetectorShardSize>(compact_shards,
        {scans_per_compact_shard, words, 0U, 0U, header_words, scan_tile, 1U});
  };
  const auto loader_for = [&](int malformed) {
    return [&, malformed](std::size_t index,
                          PackedDetectorShardDestination destination) {
      std::copy(canonical_headers.begin(), canonical_headers.end(),
                destination.descriptors.begin());
      std::fill(destination.words.begin(), destination.words.end(),
                UINT32_MAX);
      if (index != 0)
        return;
      if (malformed == 0)
        destination.descriptors[checkpoint_words] = 0x11111119U;
      if (malformed == 1)
        ++destination.descriptors[1];
      if (malformed == 2)
        destination.descriptors[0] = 1U;
    };
  };
  for (int malformed = 0; malformed < 4; ++malformed) {
    auto plan = make_plan(payload_words + (malformed == 3 ? 1U : 0U));
    bool rejected = false;
    try {
      PackedDetectorSession invalid({side, side, 1, 1}, plan,
          [](std::size_t, std::uint64_t) {}, loader_for(malformed), {},
          64ULL << 20, 1ULL << 20);
    } catch (const std::invalid_argument &error) {
      rejected = std::string_view(error.what()).find("GPU header validation") !=
                 std::string_view::npos;
    }
    require(rejected,
            "GPU compact validation must reject width, checkpoint, base, and coverage corruption");
  }
  auto plan = make_plan();
  PackedDetectorSession valid({side, side, 1, 1}, plan,
      [](std::size_t, std::uint64_t) {}, loader_for(-1), {},
      64ULL << 20, 1ULL << 20);
  std::vector<std::uint32_t> image(scans);
  const auto metrics = valid.request({0, 0, 0, 1}, 1, image);
  require(std::all_of(image.begin(), image.end(),
                      [](auto value) { return value == 1U; }) &&
              metrics.timing.dispatch_count == compact_shards,
          "Validated compact headers must retain exact bitpacked detector values");
  std::cout << "{\"status\":\"PASS\",\"compactHeaderGpuValidation\":true,"
               "\"malformedCasesRejected\":4,\"integerExact\":true}\n";
}
#endif
std::vector<std::uint32_t> exact_image(const std::vector<std::uint8_t> &mask) {
  std::vector<std::uint32_t> result(scans);
  for (std::uint32_t scan = 0; scan < scans; ++scan) {
    std::uint64_t sum = 0;
    for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
      if (mask[pixel]) sum += raw_value(scan,pixel);
    require(sum <= UINT32_MAX, "Independent exact product overflow");
    result[scan] = static_cast<std::uint32_t>(sum);
  }
  return result;
}
// Independent Float64 Fourier reference, same frozen radial FFT acceptance gate.
void forward1d(std::vector<std::complex<double>> &values) {
  for (std::size_t i = 1, reversed = 0; i < values.size(); ++i) {
    auto bit = values.size() >> 1;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (i < reversed) std::swap(values[i],values[reversed]);
  }
  for (std::size_t width = 2; width <= values.size(); width *= 2) {
    const auto step = std::polar(1.0,-2*std::numbers::pi/static_cast<double>(width));
    for (std::size_t first = 0; first < values.size(); first += width) {
      std::complex<double> phase{1,0};
      for (std::size_t i = 0; i < width/2; ++i) {
        const auto a = values[first+i], b = values[first+i+width/2]*phase;
        values[first+i] = a+b; values[first+i+width/2] = a-b; phase *= step;
      }
    }
  }
}
std::vector<double> fft_reference(std::span<const std::uint32_t> input) {
  std::vector<std::complex<double>> values(input.begin(),input.end()), line(side);
  for (std::size_t row = 0; row < side; ++row) {
    std::copy_n(values.begin()+row*side,side,line.begin()); forward1d(line);
    std::copy(line.begin(),line.end(),values.begin()+row*side);
  }
  for (std::size_t col = 0; col < side; ++col) {
    for (std::size_t row = 0; row < side; ++row) line[row] = values[row*side+col];
    forward1d(line);
    for (std::size_t row = 0; row < side; ++row) values[row*side+col] = line[row];
  }
  std::vector<double> output(scans);
  for (std::size_t row = 0; row < side; ++row)
    for (std::size_t col = 0; col < side; ++col)
      output[((row+256)%side)*side+(col+256)%side] = std::abs(values[row*side+col]);
  return output;
}
struct Comparison { double relative_l2; std::uint32_t outside; };
Comparison compare(std::span<const std::uint32_t> image, std::span<const double> expected,
                   std::span<const float> actual) {
  double input_energy = 0, reference_energy = 0, error_energy = 0;
  std::uint32_t outside = 0;
  for (auto value : image) input_energy += double(value)*value;
  const auto absolute = input_l2_limit*std::sqrt(input_energy);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto error = std::abs(double(actual[i])-expected[i]);
    if (!std::isfinite(actual[i]) || error > absolute + magnitude_relative_limit*expected[i]) ++outside;
    reference_energy += expected[i]*expected[i]; error_energy += error*error;
  }
  return {reference_energy ? std::sqrt(error_energy/reference_energy) : (error_energy ? INFINITY : 0), outside};
}

constexpr std::uint32_t partition_pixels = 65;
constexpr std::array<std::uint32_t, 8> partition_tails{1,31,32,33,127,128,129,257};
// Start full, then use disjoint complementary masks to force exact rebases rather
// than accidentally benchmarking the delta or identical-mask path.
constexpr std::array<std::uint32_t, 10> partition_counts{65,0,1,2,3,4,5,31,32,33};

std::uint16_t partition_raw_value(std::uint32_t scan, std::uint32_t pixel) {
  const auto width = pixel % 17;
  const auto maximum = (1U << width) - 1;
  return static_cast<std::uint16_t>(scan % 128 == 0 ? maximum :
      (scan * 47831U + pixel * 19381U) & maximum);
}

PackedDetectorShard partition_shard(std::uint32_t first_scan, std::uint32_t count) {
  std::vector<std::uint16_t> raw(std::size_t{count} * partition_pixels);
  for (std::uint32_t scan = 0; scan < count; ++scan)
    for (std::uint32_t pixel = 0; pixel < partition_pixels; ++pixel)
      raw[std::size_t{scan} * partition_pixels + pixel] = partition_raw_value(first_scan + scan,pixel);
  return pack_detector_shard(raw,count,partition_pixels);
}

std::uint32_t partition_expected(std::uint32_t scan, std::uint32_t selected) {
  std::uint64_t sum = 0;
  for (std::uint32_t pixel = 0; pixel < selected; ++pixel) sum += partition_raw_value(scan,pixel);
  require(sum <= UINT32_MAX, "Partition raw oracle overflow");
  return static_cast<std::uint32_t>(sum);
}

void rebase_partition_and_tail_cases() {
  for (const auto count : partition_tails) {
    const auto shard = partition_shard(0,count);
    for (std::uint32_t width = 0; width <= 16; ++width)
      require((shard.descriptors[width * ((count+127)/128)] & 31U) == width,
              "Partition fixture must exercise every packed width including16");
    for (const auto selected : partition_counts) {
      const auto mask = circular_detector_mask(1,partition_pixels,{0,0,0,float(selected)});
      require(std::count(mask.begin(),mask.end(),1) == selected, "Partition mask entry count changed");
      const auto expected = reference_packed_detector_update(shard,plan_packed_detector_update({},mask));
      std::vector<std::uint32_t> partitioned(count);
      for (std::uint32_t first_scan = 0; first_scan < count; first_scan += 32) {
        std::array<std::uint32_t,128> partials{};
        for (std::uint32_t lane = 0; lane < partials.size(); ++lane) {
          const auto scan = first_scan + (lane & 31U);
          if (scan < count)
            for (auto pixel = lane >> 5U; pixel < selected; pixel += 4)
              partials[lane] += partition_raw_value(scan,pixel);
        }
        for (std::uint32_t lane = 0; lane < 32 && first_scan+lane < count; ++lane)
          partitioned[first_scan+lane] = partials[lane]+partials[lane+32]+partials[lane+64]+partials[lane+96];
      }
      for (std::uint32_t scan = 0; scan < count; ++scan) {
        require(partitioned[scan] == expected[scan] && expected[scan] == partition_expected(scan,selected),
                "Four-way UInt32 partition differs from packed or raw oracle");
      }
    }
  }
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
  std::vector<std::uint32_t> counts(partition_tails.begin(),partition_tails.end()), first_scans;
  std::uint32_t covered = 0;
  for (const auto count : counts) covered += count;
  while (covered < scans) {
    const auto count = std::min(shard_scans,scans-covered);
    counts.push_back(count); covered += count;
  }
  std::vector<PackedDetectorShardSize> plan;
  std::uint64_t expected_upload = 0;
  covered = 0;
  for (const auto count : counts) {
    first_scans.push_back(covered);
    const auto shard = partition_shard(covered,count);
    plan.push_back({count,static_cast<std::uint32_t>(shard.words.size())});
    expected_upload += 4ULL * (shard.words.size()+shard.descriptors.size());
    covered += count;
  }
  std::uint32_t loader_calls = 0;
  const auto loader = [&](std::size_t index, PackedDetectorShardDestination destination) {
    ++loader_calls;
    copy_shard(partition_shard(first_scans[index],counts[index]),destination);
  };
  PackedDetectorSession session({side,side,1,partition_pixels},plan,
      [](std::size_t, std::uint64_t) {},loader,{},256ULL<<20,8ULL<<20);
  const auto committed_bytes = session.admission().committed_bytes;
  std::vector<std::uint32_t> actual(scans);
  std::uint64_t generation = 0;
  for (std::size_t index = 0; index < partition_counts.size(); ++index) {
    const auto selected = partition_counts[index];
    if (index != 0)
      (void)session.request({0,0,float(selected),float(partition_pixels)},++generation,actual);
    const CircularDetector detector{0,0,0,float(selected)};
    const auto metrics = session.request(detector,++generation,actual);
    require(metrics.rebase && metrics.timing.dispatch_count == plan.size() &&
            metrics.timing.queue_submit_count == 1 && metrics.timing.fence_wait_count == 1,
            "Partition target must run a full rebase over every irregular shard");
    require(metrics.committed_generation == generation && metrics.timing.source_upload_bytes == 0 &&
            metrics.timing.device_wide_wait_count == 0, "Partition rebase changed generation or residency");
    for (std::uint32_t scan = 0; scan < scans; ++scan)
      require(actual[scan] == partition_expected(scan,selected),
              "GPU four-way rebase differs from independent raw UInt32 oracle");
    const auto repeated = session.request(detector,++generation,actual);
    require(!repeated.source_changed && repeated.timing.dispatch_count == 0 &&
            repeated.timing.queue_submit_count == 0, "Partition repeat must reuse the committed image");
    for (std::uint32_t scan = 0; scan < scans; ++scan)
      require(actual[scan] == partition_expected(scan,selected), "Cached partition image changed");
  }
  require(loader_calls == plan.size() && session.admission().source_upload_bytes == expected_upload &&
          session.admission().committed_bytes == committed_bytes,
          "Partition requests must not reload source or grow resident workspace");
  std::cout << "{\"partitionTailCases\":" << partition_counts.size() << ",\"shards\":" << plan.size()
            << ",\"allWidths0Through16\":true,\"gpuTested\":true,\"integerExact\":true}\n";
#else
  std::cout << "{\"partitionTailCases\":" << partition_counts.size()*partition_tails.size()
            << ",\"allWidths0Through16\":true,\"gpuTested\":false,\"integerExact\":true}\n";
#endif
}
} // namespace

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  try {
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
    if (argc == 2 && std::string_view(argv[1]) == "--compact-validation-only") {
      compact_header_gpu_validation();
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--expect-small-heap-rejection") {
      // Metadata-only regression for the 2 GiB SwiftShader heap. The guarded
      // loader throws immediately if reached, so the failing baseline never
      // reads or allocates this deliberately oversized source.
      const std::vector<PackedDetectorShardSize> oversized_plan(64, {4096, 8388608});
      unsigned guard_calls = 0, loader_calls = 0;
      const auto guard = [&](std::size_t, std::uint64_t) { ++guard_calls; };
      const auto guarded_loader = [&](std::size_t, PackedDetectorShardDestination) {
        ++loader_calls;
        throw std::runtime_error("Source loader reached before heap admission");
      };
      bool rejected_for_heap = false;
      try {
        PackedDetectorSession too_large({512,512,192,192}, oversized_plan, guard, guarded_loader,
                                        {}, 8ULL << 30, 40ULL << 20);
      } catch (const std::invalid_argument &error) {
        rejected_for_heap = std::string_view(error.what()).find("heap") != std::string_view::npos;
      }
      require(rejected_for_heap && guard_calls == 0 && loader_calls == 0,
              "Full selected-heap demand must be rejected before the first source load");
      std::cout << "{\"status\":\"PASS\",\"mode\":\"small-heap negative admission only\","
                   "\"sourceLoaderCalls\":0,\"scientificGpuTested\":false}\n";
      return 0;
    }
    require(argc == 1, "Use no arguments, --compact-validation-only, or --expect-small-heap-rejection on a known small-heap adapter");
    destination_loader_failures_and_zero_payload();
#endif
    rebase_partition_and_tail_cases();
    const std::array<CircularDetector,11> cases{{
      {2,3,0,1.5F}, {2,3.25F,0,1.5F}, {2,4,0,1.5F}, {2,3,1,2.5F},
      {2.25F,2.75F,0.5F,2.75F}, {0.5F,0.5F,0,3}, {-3,-4,0,1},
      {2,3,0,20}, {2,3,0,2}, {2,3,0,std::nextafter(2.0F,INFINITY)}, {2,3,0,2}
    }};
    std::vector<std::uint8_t> excluded(pixels);
    excluded[0] = 1; excluded[pixels-1] = 20;
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
    std::vector<PackedDetectorShardSize> plan;
    std::uint64_t expected_upload = 0;
    for (std::size_t index = 0; index < 64; ++index) {
      const auto shard = synthetic_shard(index);
      plan.push_back({shard.scan_count,static_cast<std::uint32_t>(shard.words.size())});
      expected_upload += (shard.words.size()+shard.descriptors.size())*4;
    }
    std::uint32_t guard_calls = 0, loader_calls = 0;
    std::uint64_t loaded_bytes = 0;
    const auto guard = [&](std::size_t index, std::uint64_t admitted_bytes) {
      require(index == guard_calls && index == loader_calls && admitted_bytes == loaded_bytes,
              "Synthetic source guard must see only previously admitted exact bytes");
      ++guard_calls;
    };
    auto loader = [&](std::size_t index, PackedDetectorShardDestination destination) {
      ++loader_calls;
      copy_shard(synthetic_shard(index),destination);
      loaded_bytes += destination.descriptors.size_bytes()+destination.words.size_bytes();
    };
    bool rejected = false;
    try { PackedDetectorSession bad({512,512,5,7},plan,guard,loader,excluded,1,1); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && guard_calls == 0 && loader_calls == 0,
            "Whole-plan budget must fail before guarding or loading any shard");
    PackedDetectorSession session({512,512,5,7},plan,guard,loader,excluded,256ULL<<20,8ULL<<20);
    const auto admission = session.admission();
    require(guard_calls == 64 && loader_calls == 64 && admission.source_upload_bytes == expected_upload,
            "Every synthetic source shard uploads once");
    std::cout << "{\"admission\":true,\"device\":" << std::quoted(admission.device_name)
              << ",\"sourceUploadBytes\":" << admission.source_upload_bytes
              << ",\"committedBytes\":" << admission.committed_bytes
              << ",\"initializationMs\":" << admission.initialization_milliseconds << "}\n";
    check_selected_raw(session,scans-1,excluded);
#endif
    [[maybe_unused]] std::uint64_t generation = 1;
    std::vector<std::uint8_t> previous_mask;
    for (const auto geometry : cases) {
      const auto mask = circular_detector_mask(5,7,geometry,excluded);
      require(mask != previous_mask, "Each dispatch-count case must actually change selection");
      previous_mask = mask;
      const auto expected = exact_image(mask);
      const auto expected_fft = fft_reference(expected);
      std::vector<std::uint32_t> actual(scans);
      std::vector<float> fft(scans);
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
      const auto metrics = session.request(geometry,generation,actual,fft);
      require(generation != 1 || metrics.rebase, "Selected DP before the first detector cannot create its base");
      require(actual == expected, "GPU packed full/delta differs from raw uint16 oracle");
      require(metrics.committed_generation == generation && metrics.timing.generation == generation,
              "Private committed generation must follow every successful compute");
      require(metrics.timing.source_upload_bytes == 0 && metrics.timing.device_wide_wait_count == 0,
              "No source upload or device-wide wait during requests");
      require(metrics.timing.queue_submit_count == 1 && metrics.timing.fence_wait_count == 1 &&
              metrics.timing.dispatch_count == 67, "64 detector shards +3 FFT in one submission/fence");
#else
      actual = expected;
      std::transform(expected_fft.begin(),expected_fft.end(),fft.begin(),[](auto value) { return static_cast<float>(value); });
#endif
      const auto accuracy = compare(expected,expected_fft,fft);
      require(accuracy.relative_l2 <= relative_l2_limit && accuracy.outside == 0,
              "Frozen FFT accuracy gate failed; do not weaken tolerance");
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
      std::cout << "{\"generation\":" << generation << ",\"integerExact\":true,\"relativeL2\":"
                << accuracy.relative_l2 << ",\"outsideTolerance\":" << accuracy.outside
                << ",\"rebase\":" << (metrics.rebase ? "true" : "false")
                << ",\"logicalSourceBytes\":" << metrics.logical_source_bytes
                << ",\"wallMs\":" << metrics.timing.wall_milliseconds << ",\"gpuMs\":";
      number(metrics.timing.gpu_total_milliseconds); std::cout << "}\n";
      // UI may ignore this exact result. Next request still uses the private base.
      check_selected_raw(session,static_cast<std::uint32_t>((generation*19731)%scans),excluded);
      const auto repeated = session.request(geometry,++generation,actual,fft);
      require(repeated.timing.dispatch_count == 0 && repeated.timing.queue_submit_count == 0 && actual == expected,
              "Identical mask and FFT should reuse committed GPU results");
#endif
      if (std::any_of(expected.begin(),expected.end(),[](auto value) { return value != 0; })) {
        for (auto &value : fft) value /= float(scans);
        require(compare(expected,expected_fft,fft).relative_l2 > relative_l2_limit,
                "Failure control accepted incorrectly normalized FFT");
      }
      ++generation;
    }
#ifndef PACKED_DETECTOR_REFERENCE_ONLY
    std::vector<std::uint32_t> image(scans);
    std::vector<float> fft(scans);
    const auto no_fft = session.request(cases[3],++generation,image);
    require(no_fft.source_changed && no_fft.timing.dispatch_count == 64 &&
            std::isnan(no_fft.timing.gpu_fft_rows_milliseconds), "FFT-off skips all FFT stages");
    check_selected_raw(session,side*256+256,excluded);
    const auto fft_only = session.request(cases[3],++generation,image,fft);
    require(!fft_only.source_changed && fft_only.timing.dispatch_count == 3 &&
            fft_only.timing.queue_submit_count == 1, "FFT-on reuses image with only3 GPU stages");
    rejected = false;
    try { (void)session.request(cases[3],++generation,std::span(image).first(10)); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected,"Wrong output size must fail before compute");
    (void)session.request(cases[3],++generation,image,fft);
    require(loader_calls == 64 && session.admission().committed_bytes == admission.committed_bytes,
            "Requests must not reload source or grow GPU workspace");
    std::cout << "{\"status\":\"PASS\",\"gpuTested\":true,\"syntheticOnly\":true,\"cases\":11}\n";
#else
    std::cout << "{\"status\":\"PASS\",\"gpuTested\":false,\"mode\":\"CPU reference preflight only\",\"cases\":11}\n";
#endif
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Packed detector session test failed: " << error.what() << '\n';
    return 1;
  }
}
