#include "quantem/gpu/vulkan/packed_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace quantem::gpu::vulkan;

namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void rejects(const std::function<void()> &action, const char *message) {
  try { action(); } catch (const std::invalid_argument &) { return; }
  throw std::runtime_error(message);
}

std::vector<std::uint32_t> raw_sum(std::span<const std::uint16_t> raw,
                                  std::span<const std::uint8_t> mask) {
  std::vector<std::uint32_t> values(raw.size() / mask.size());
  for (std::size_t scan = 0; scan < values.size(); ++scan) {
    std::uint64_t sum = 0;
    for (std::size_t pixel = 0; pixel < mask.size(); ++pixel)
      if (mask[pixel]) sum += raw[scan * mask.size() + pixel];
    require(sum <= std::numeric_limits<std::uint32_t>::max(), "oracle overflow");
    values[scan] = static_cast<std::uint32_t>(sum);
  }
  return values;
}

void all_widths_and_tails() {
  for (const auto scans : {1U, 31U, 127U, 128U, 129U, 257U, 4096U}) {
    constexpr auto pixels = 17U;
    std::vector<std::uint16_t> raw(scans * pixels);
    for (std::uint32_t scan = 0; scan < scans; ++scan) {
      for (std::uint32_t width = 0; width <= 16; ++width) {
        const auto mask = (1U << width) - 1;
        raw[scan * pixels + width] = static_cast<std::uint16_t>(
            scan % 128 == 0 ? mask : ((scan * 47831U + width * 19381U) & mask));
      }
    }
    const auto shard = pack_detector_shard(raw, scans, pixels);
    validate_packed_detector_shard(scans, pixels, shard.descriptors, shard.words);
    require(unpack_detector_shard(shard) == raw, "all-width/tail roundtrip");
    const auto tiles = (scans + 127) / 128;
    for (std::uint32_t width = 0; width <= 16; ++width)
      for (std::uint32_t tile = 0; tile < tiles; ++tile)
        require((shard.descriptors[width * tiles + tile] & 31U) == width,
                "all widths actually exercised");
  }
}

void borrowed_validation_and_lengths() {
  const std::array<std::uint16_t, 4> raw{1, 2, 3, 65535};
  const auto good = pack_detector_shard(raw, 1, 4);
  const auto descriptors = good.descriptors, words = good.words;
  validate_packed_detector_shard(1, 4, good.descriptors, good.words);
  require(good.descriptors == descriptors && good.words == words,
          "Borrowed validation must not mutate source bytes");
  rejects([&] { validate_packed_detector_shard(1, 4,
      std::span(good.descriptors).first(3), good.words); }, "Short descriptor span rejected");
  auto oversized_headers = good.descriptors;
  oversized_headers.push_back(0);
  rejects([&] { validate_packed_detector_shard(1, 4, oversized_headers, good.words); },
          "Long descriptor span rejected");
  rejects([&] { validate_packed_detector_shard(1, 4, good.descriptors,
      std::span(good.words).first(good.words.size()-1)); }, "Short payload span rejected");
  auto oversized_words = good.words;
  oversized_words.push_back(0);
  rejects([&] { validate_packed_detector_shard(1, 4, good.descriptors, oversized_words); },
          "Trailing payload span rejected");
  auto malformed = good.descriptors;
  malformed[0] |= 31;
  rejects([&] { validate_packed_detector_shard(1, 4, malformed, good.words); },
          "Borrowed invalid width rejected");
  malformed = good.descriptors;
  malformed[1] += 32;
  rejects([&] { validate_packed_detector_shard(1, 4, malformed, good.words); },
          "Borrowed noncanonical offset rejected");
  rejects([&] { validate_packed_detector_shard(0, 4, {}, {}); }, "Zero scan count rejected");
  rejects([&] { validate_packed_detector_shard(1, 0, {}, {}); }, "Zero detector count rejected");
  rejects([&] { validate_packed_detector_shard(1, 65538, {}, {}); }, "Wide-count overflow rejected");
  const auto zero = pack_detector_shard(std::vector<std::uint16_t>(129*4), 129, 4);
  require(zero.words.empty(), "All-zero fixture must have no logical payload");
  validate_packed_detector_shard(129, 4, zero.descriptors, {});
  validate_packed_detector_shard(zero);
}

void canonical_aperture_boundaries() {
  auto selected_count = [](const auto &mask) {
    return std::count(mask.begin(), mask.end(), 1);
  };
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto at = circular_detector_mask(9, 11, {4, 5, 1, 2});
  require(selected_count(at) == 8, "inner inclusive, outer exclusive");
  require(at[4 * 11 + 6] && !at[4 * 11 + 7] && !at[4 * 11 + 5],
          "exact integer boundary membership");
  require(selected_count(circular_detector_mask(9, 11,
      {4, 5, 1, std::nextafter(2.0F, infinity)})) == 12, "outer nextafter expands");
  require(selected_count(circular_detector_mask(9, 11,
      {4, 5, std::nextafter(1.0F, infinity), 2})) == 4, "inner nextafter removes");
  require(selected_count(circular_detector_mask(9, 11,
      {4, 5, 1, std::nextafter(2.0F, 0.0F)})) == 8, "outer nextafter below");
  require(selected_count(circular_detector_mask(9, 11, {4.5F, 5.5F, 0, 1})) == 4,
          "half-pixel translated disk");
  require(selected_count(circular_detector_mask(9, 11, {0, 0, 0, 2})) == 4,
          "detector edge is clipped, not wrapped");
  require(selected_count(circular_detector_mask(9, 11, {4, 5, 2, 2})) == 0,
          "equal radii empty");
  std::vector<std::uint8_t> excluded(99);
  excluded[4 * 11 + 6] = 20;
  require(selected_count(circular_detector_mask(9, 11, {4, 5, 1, 2}, excluded)) == 7,
          "nonzero authoritative mask excludes count without renormalization");
  rejects([&] { (void)circular_detector_mask(9, 11, {4, 5, 2, 1}); }, "reversed radii");
  rejects([&] { (void)circular_detector_mask(9, 11, {infinity, 5, 0, 1}); }, "nonfinite center");
}

void frozen_macos_float_masks() {
  // Independent Swift Float execution of c01c6ec MetalDatasetService.detectorMask.
  // Pins are FNV-1a over all 192*192 selection bytes, not candidate-generated.
  struct Case { CircularDetector geometry; std::uint64_t hash; };
  const std::array<Case, 13> cases{{
    {{95.5F,95.5F,0,32},10824918087180012452ULL},
    {{95.5F,95.75F,0,32},7971989059085650136ULL},
    {{95.5F,96.5F,0,32},6692809710240417664ULL},
    {{95,95,0,31.75F},10711339472306874635ULL},
    {{95.5F,95.5F,32,64},2376401356733602485ULL},
    {{96.25F,94.75F,32,64},18388171867819075109ULL},
    {{0.5F,0.5F,1.25F,20},15354836480524208872ULL},
    {{95.5F,95.5F,0,136},14679129994679128281ULL},
    {{95,95,1,32},9524284740448329568ULL},
    {{95,95,1,std::nextafter(32.0F,INFINITY)},8076834956510112388ULL},
    {{95,95,1,std::nextafter(32.0F,0.0F)},9524284740448329568ULL},
    {{95,95,std::nextafter(1.0F,INFINITY),32},353611904773846072ULL},
    {{95.12345F,96.76543F,12.34567F,31.23456F},15425092000779554110ULL},
  }};
  std::vector<std::uint8_t> excluded(192*192);
  for (const auto pixel : {27*192+135,78*192+74,113*192+14,156*192+13}) excluded[pixel] = 1;
  for (const auto &test : cases) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : circular_detector_mask(192,192,test.geometry,excluded)) {
      hash ^= value;
      hash *= 1099511628211ULL;
    }
    require(hash == test.hash, "frozen macOS Float mask mismatch; do not rewrite pin");
  }
}

void continuous_drag_and_rebase() {
  constexpr std::uint32_t scans = 257, rows = 9, columns = 11, pixels = rows * columns;
  std::vector<std::uint16_t> raw(scans * pixels);
  std::vector<std::uint8_t> excluded(pixels);
  excluded[4 * columns + 5] = 1;
  for (std::uint32_t scan = 0; scan < scans; ++scan)
    for (std::uint32_t pixel = 0; pixel < pixels; ++pixel)
      raw[scan * pixels + pixel] = excluded[pixel] ? 65535 :
          static_cast<std::uint16_t>((scan * 17 + pixel * 7) % 30);
  const auto original = raw;
  const auto shard = pack_detector_shard(raw, scans, pixels);
  std::vector<std::uint8_t> previous;
  std::vector<std::uint32_t> image;
  std::uint32_t deltas = 0, rebases = 0;
  for (int step = 0; step < 130; ++step) {
    CircularDetector geometry{4.0F, 4.0F + float(step % 40) / 20, 0, 3.25F};
    if (step >= 80) geometry = {float(step % 9), float(step % 11), 1.25F, 4.5F};
    if (step == 110) previous.clear(); // Lost/cancelled base must force exact rebase.
    const auto next = circular_detector_mask(rows, columns, geometry, excluded);
    const auto plan = plan_packed_detector_update(previous, next);
    plan.rebase ? ++rebases : ++deltas;
    image = reference_packed_detector_update(shard, plan, image);
    require(image == raw_sum(raw, next), "trajectory packed/delta vs independent raw sums");
    const auto full = reference_packed_detector_update(shard, plan_packed_detector_update({}, next));
    require(image == full, "delta versus full exact equality");
    previous = next;
  }
  require(deltas > 50 && rebases >= 2, "exercise both delta and rebase");
  require(raw == original && unpack_detector_shard(shard) == original,
          "product masking must not mutate original raw sentinels");
  const std::vector<std::uint8_t> zero(pixels);
  const auto empty = reference_packed_detector_update(shard,
      plan_packed_detector_update(previous, zero), image);
  require(std::all_of(empty.begin(), empty.end(), [](auto value) { return value == 0; }),
          "remove all selected pixels exactly");
  const auto stable = plan_packed_detector_update(previous, previous);
  require(!stable.rebase && stable.entries.empty(), "null motion needs no source work");
}

void planner_cost_and_failure_controls() {
  const std::array<std::uint8_t, 4> previous{1, 0, 0, 0}, next{0, 1, 0, 0};
  const std::array<std::uint64_t, 4> costs{100, 4, 1, 1};
  const auto plan = plan_packed_detector_update(previous, next, costs);
  require(plan.rebase && plan.entries.size() == 1 && plan.logical_source_bytes == 4,
          "large changes choose cheaper full rebase");
  const std::array<std::uint8_t, 4> delta_previous{1, 0, 1, 0};
  const std::array<std::uint8_t, 4> delta_next{1, 1, 1, 0};
  const std::array<std::uint64_t, 4> delta_costs{100, 4, 100, 1};
  const auto delta = plan_packed_detector_update(delta_previous, delta_next, delta_costs);
  require(!delta.rebase && delta.entries.size() == 1 &&
              delta.entries.front().pixel == 1 && delta.entries.front().coefficient == 1 &&
              delta.logical_source_bytes == 4,
          "small weighted change preserves exact delta traffic telemetry");
  const std::array<std::uint8_t, 2> tie_previous{1, 1}, tie_next{0, 1};
  const auto tie = plan_packed_detector_update(tie_previous, tie_next);
  require(!tie.rebase && tie.entries.size() == 1 && tie.logical_source_bytes == 0,
          "equal entry counts keep delta and do not label counts as source bytes");
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const std::array<std::uint8_t, 2> full_overflow_next{1, 1};
  const std::array<std::uint8_t, 2> delta_overflow_previous{1, 1};
  const std::array<std::uint8_t, 2> delta_overflow_next{0, 0};
  const std::array<std::uint64_t, 2> overflow_costs{maximum, 1};
  rejects([&] { (void)plan_packed_detector_update(
      {}, full_overflow_next, overflow_costs); }, "full traffic overflow rejected");
  rejects([&] { (void)plan_packed_detector_update(
      delta_overflow_previous, delta_overflow_next, overflow_costs); },
      "delta traffic overflow rejected");
  const std::array<std::uint16_t, 4> raw{1, 2, 3, 4};
  const auto good = pack_detector_shard(raw, 1, 4);
  auto corrupted = good;
  corrupted.descriptors[0] |= 31;
  rejects([&] { validate_packed_detector_shard(corrupted); }, "invalid width rejected");
  corrupted = good;
  corrupted.descriptors[1] += 32;
  rejects([&] { validate_packed_detector_shard(corrupted); }, "invalid offset rejected");
  corrupted = good;
  corrupted.words.pop_back();
  rejects([&] { validate_packed_detector_shard(corrupted); }, "truncated payload rejected");
  corrupted = good;
  corrupted.words[0] ^= 1;
  require(unpack_detector_shard(corrupted) != std::vector<std::uint16_t>(raw.begin(), raw.end()),
          "payload corruption detected by exact roundtrip/hash gate, not shape validation");
  rejects([&] { (void)reference_packed_detector_update(good, {false, {{0,-1}},0}); },
          "missing delta base rejected");
  const std::array<std::uint32_t, 1> wrong_base{0};
  rejects([&] { (void)reference_packed_detector_update(good, {false, {{0,-1}},0}, wrong_base); },
          "wrong delta sign/base produces invalid negative product");
  rejects([&] { (void)reference_packed_detector_update(good, {true, {{0,1},{0,1}},0}); },
          "duplicate entries rejected");
  rejects([&] { (void)pack_detector_shard(raw, 1, 4, 16); }, "packing budget fail closed");
  const auto bytes = good.words.size() * 4 + good.descriptors.size() * 4;
  const DeviceLimits limits{1ULL << 27, 1ULL << 30, bytes + 4096};
  require(admit_packed_detector_shard(good, limits, 0, 4096) == bytes,
          "exact memory budget accepted");
  rejects([&] { (void)admit_packed_detector_shard(good, limits, 1, 4096); }, "peak budget rejected");
  rejects([&] { (void)admit_packed_detector_shard(good, {4, 1ULL<<30, 1ULL<<30}, 0, 0); },
          "device storage range rejected");
  rejects([&] { (void)admit_packed_detector_shard(good, {}, 0, 0); }, "unknown limits rejected");
}

void adf_center_translation_uses_sparse_exact_delta() {
  constexpr std::uint32_t rows = 192, columns = 192;
  std::vector<std::uint8_t> excluded(rows * columns);
  for (const auto pixel : {27U * columns + 135U, 78U * columns + 74U,
                           113U * columns + 14U, 156U * columns + 13U})
    excluded[pixel] = 1;
  const auto previous = circular_detector_mask(
      rows, columns, {95.5F, 95.5F, 32.0F, 64.0F}, excluded);
  // One step from the physical 240-move ADF translation trajectory.
  const auto next = circular_detector_mask(
      rows, columns, {95.425F, 95.416664F, 32.0F, 64.0F}, excluded);
  const auto update = plan_packed_detector_update(previous, next);
  std::size_t changed = 0;
  for (std::size_t pixel = 0; pixel < next.size(); ++pixel)
    if (previous[pixel] != next[pixel]) ++changed;
  const auto selected = static_cast<std::size_t>(std::count(next.begin(), next.end(), 1));
  require(!update.rebase, "small ADF center translation must use an exact delta");
  require(changed > 0 && changed <= 110 && changed * 50 < selected,
          "physical ADF step must stay sparse relative to the full annulus");
  require(update.entries.size() == changed,
          "ADF delta must contain every and only changed detector pixel");
  require(update.logical_source_bytes == 0,
          "entry-count planning must not be mislabeled as measured source bytes");
  for (std::size_t index = 0; index < update.entries.size(); ++index) {
    const auto &entry = update.entries[index];
    require(index == 0 || update.entries[index - 1].pixel < entry.pixel,
            "ADF delta entries must remain strictly sorted");
    require(previous[entry.pixel] != next[entry.pixel] &&
                entry.coefficient == (next[entry.pixel] ? 1 : -1),
            "ADF delta coefficient must exactly match mask membership change");
  }

  const auto rebase = plan_packed_detector_update({}, next);
  require(rebase.rebase && rebase.entries.size() == selected,
          "missing ADF base must still materialize the complete exact annulus");
  require(std::all_of(rebase.entries.begin(), rebase.entries.end(),
      [](const auto &entry) { return entry.coefficient == 1; }),
      "ADF rebase may contain only positive selected-pixel entries");
}

void prepared_dpc_exact_oracle_and_fail_closed() {
  constexpr Shape4D shape{2, 2, 2, 3};
  const std::array<std::array<std::uint8_t, 6>, 4> source{{
      {{1, 2, 255, 3, 4, 5}},
      {{0, 7, 1, 9, 2, 4}},
      {{5, 0, 99, 1, 8, 3}},
      {{2, 6, 17, 4, 0, 10}},
  }};
  std::array<std::uint8_t, 6> excluded{};
  excluded[2] = 1;
  std::vector<std::uint32_t> words(4 * 8, 0U);
  std::array<float, 4> com_row{}, com_column{};
  for (std::size_t scan = 0; scan < source.size(); ++scan) {
    std::uint64_t total = 0, row = 0, column = 0;
    for (std::size_t pixel = 0; pixel < source[scan].size(); ++pixel) {
      if (excluded[pixel]) continue;
      total += source[scan][pixel];
      row += source[scan][pixel] * (pixel / shape.detector_columns);
      column += source[scan][pixel] * (pixel % shape.detector_columns);
    }
    words[scan * 8] = static_cast<std::uint32_t>(total);
    words[scan * 8 + 2] = static_cast<std::uint32_t>(row);
    words[scan * 8 + 4] = static_cast<std::uint32_t>(column);
    com_row[scan] = static_cast<float>(static_cast<double>(row) / total);
    com_column[scan] =
        static_cast<float>(static_cast<double>(column) / total);
  }
  const auto bounds = validate_prepared_dpc_moments(shape, excluded, words);
  require(bounds.total == 5U * 255U && bounds.row == 3U * 255U &&
              bounds.column == 4U * 255U,
          "Prepared DPC bounds must derive from every nonexcluded coordinate");
  const auto [row, column] = reference_prepared_dpc(words);
  const auto row_mean = static_cast<float>(
      (com_row[0] + com_row[1] + com_row[2] + com_row[3]) / 4.0F);
  const auto column_mean = static_cast<float>(
      (com_column[0] + com_column[1] + com_column[2] + com_column[3]) / 4.0F);
  for (std::size_t scan = 0; scan < source.size(); ++scan) {
    require(std::abs(row[scan] - (com_row[scan] - row_mean)) < 1e-6F &&
                std::abs(column[scan] -
                         (com_column[scan] - column_mean)) < 1e-6F,
            "Prepared DPC oracle must preserve exact moments and centered components");
  }
  for (const auto corruption : {0U, 1U, 2U, 3U}) {
    auto invalid = words;
    if (corruption == 0U) invalid[6] = 1U;
    if (corruption == 1U) invalid[1] = 1U;
    if (corruption == 2U) {
      invalid[0] = 0U;
      invalid[2] = 1U;
    }
    if (corruption == 3U) invalid.pop_back();
    rejects([&] {
      (void)validate_prepared_dpc_moments(shape, excluded, invalid);
    }, "Malformed prepared DPC words must fail closed");
  }
}

void full_uint32_range() {
  constexpr auto pixels = 65537U;
  const std::vector<std::uint16_t> raw(pixels, 65535);
  const auto shard = pack_detector_shard(raw, 1, pixels);
  const std::vector<std::uint8_t> mask(pixels, 1);
  const auto full = reference_packed_detector_update(shard, plan_packed_detector_update({}, mask));
  require(full[0] == std::numeric_limits<std::uint32_t>::max(), "exact full uint32 product");
  const auto removed = reference_packed_detector_update(shard, {false, {{0, -1}}, 0}, full);
  require(removed[0] == std::numeric_limits<std::uint32_t>::max() - 65535U,
          "delta does not use signed int32 accumulator");
  rejects([&] { (void)pack_detector_shard({}, 1, pixels + 1); }, "uint32 overflow bound rejected");
}
} // namespace

int main() {
  try {
    all_widths_and_tails(); std::cout << "PASS all_widths_and_tails\n";
    borrowed_validation_and_lengths(); std::cout << "PASS borrowed_validation_and_lengths\n";
    canonical_aperture_boundaries(); std::cout << "PASS canonical_aperture_boundaries\n";
    frozen_macos_float_masks(); std::cout << "PASS frozen_macos_float_masks (13)\n";
    continuous_drag_and_rebase(); std::cout << "PASS continuous_drag_and_rebase\n";
    planner_cost_and_failure_controls(); std::cout << "PASS planner_cost_and_failure_controls\n";
    adf_center_translation_uses_sparse_exact_delta();
    std::cout << "PASS adf_center_translation_uses_sparse_exact_delta\n";
    prepared_dpc_exact_oracle_and_fail_closed();
    std::cout << "PASS prepared_dpc_exact_oracle_and_fail_closed\n";
    full_uint32_range(); std::cout << "PASS full_uint32_range\n";
    std::cout << "HOST_PREFLIGHT_PASS: 9 groups, 130 exact drag states; GPU not executed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
