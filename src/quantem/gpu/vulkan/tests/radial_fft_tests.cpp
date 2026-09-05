#include "quantem/gpu/vulkan/radial_fft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using quantem::gpu::vulkan::RadialFftSession;
namespace {
constexpr std::size_t kSide = 512;
constexpr std::size_t kCount = kSide * kSide;
constexpr double kRelativeL2Tolerance = 2.0e-6;
constexpr double kMagnitudeRelativeTolerance = 2.0e-5;
constexpr double kInputL2AbsoluteTolerance = 4.0e-6;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

// Independent double-precision reference. No GPU twiddle table/kernel code is reused.
void forward1d(std::vector<std::complex<double>> &values) {
  for (std::size_t i = 1, reversed = 0; i < values.size(); ++i) {
    std::size_t bit = values.size() >> 1;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (i < reversed) std::swap(values[i], values[reversed]);
  }
  for (std::size_t length = 2; length <= values.size(); length *= 2) {
    const auto step = std::polar(1.0, -2.0 * std::numbers::pi / static_cast<double>(length));
    for (std::size_t start = 0; start < values.size(); start += length) {
      std::complex<double> phase{1.0, 0.0};
      for (std::size_t offset = 0; offset < length / 2; ++offset) {
        const auto a = values[start + offset];
        const auto b = values[start + offset + length / 2] * phase;
        values[start + offset] = a + b;
        values[start + offset + length / 2] = a - b;
        phase *= step;
      }
    }
  }
}

std::vector<double> reference(const std::vector<std::uint32_t> &image) {
  std::vector<std::complex<double>> values(image.begin(), image.end());
  std::vector<std::complex<double>> line(kSide);
  for (std::size_t row = 0; row < kSide; ++row) {
    std::copy_n(values.begin() + row * kSide, kSide, line.begin());
    forward1d(line);
    std::copy(line.begin(), line.end(), values.begin() + row * kSide);
  }
  for (std::size_t column = 0; column < kSide; ++column) {
    for (std::size_t row = 0; row < kSide; ++row) line[row] = values[row * kSide + column];
    forward1d(line);
    for (std::size_t row = 0; row < kSide; ++row) values[row * kSide + column] = line[row];
  }
  std::vector<double> magnitudes(kCount);
  for (std::size_t row = 0; row < kSide; ++row) {
    for (std::size_t column = 0; column < kSide; ++column) {
      magnitudes[((row + 256) % kSide) * kSide + (column + 256) % kSide] =
          std::abs(values[row * kSide + column]);
    }
  }
  return magnitudes;
}

struct Comparison { double relative_l2, maximum_absolute; std::size_t outside_tolerance; };
Comparison compare(const std::vector<std::uint32_t> &image, const std::vector<double> &expected,
                   const std::vector<float> &actual) {
  double input_energy = 0.0, reference_energy = 0.0, error_energy = 0.0, maximum = 0.0;
  for (auto value : image) input_energy += static_cast<double>(value) * value;
  const double absolute_tolerance = kInputL2AbsoluteTolerance * std::sqrt(input_energy);
  std::size_t outside = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    if (!std::isfinite(actual[i]) || error > kMagnitudeRelativeTolerance * expected[i] + absolute_tolerance) ++outside;
    maximum = std::max(maximum, error);
    reference_energy += expected[i] * expected[i];
    error_energy += error * error;
  }
  const double relative = reference_energy == 0.0 ? (error_energy == 0.0 ? 0.0 : INFINITY) :
      std::sqrt(error_energy / reference_energy);
  return {relative, maximum, outside};
}

std::vector<std::uint32_t> syntheticPrefix() {
  std::vector<std::uint32_t> prefix(RadialFftSession::prefix_planes * kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    const std::uint32_t row = static_cast<std::uint32_t>(i / kSide);
    const std::uint32_t column = static_cast<std::uint32_t>(i % kSide);
    prefix[kCount + i] = 7;
    prefix[2 * kCount + i] = 7 + ((row == 17 && column == 31) ? 1024 : 0);
    prefix[3 * kCount + i] = prefix[2 * kCount + i] + (row * 37 + column * 73 + row * column * 11) % 128;
    prefix[4 * kCount + i] = prefix[3 * kCount + i] +
        static_cast<std::uint32_t>(128 + std::lround(63 * std::sin(2 * std::numbers::pi * (3 * row + 7 * column) / 512.0)));
    for (std::size_t radius = 5; radius < 136; ++radius) {
      prefix[radius * kCount + i] = prefix[4 * kCount + i] + static_cast<std::uint32_t>(radius - 5);
    }
    prefix[136 * kCount + i] = UINT32_MAX;
  }
  return prefix;
}

std::vector<std::uint32_t> readPrefix(const char *path) {
  const auto count = RadialFftSession::prefix_planes * kCount;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(input.good(), "Cannot read authenticated radial prefix file");
  require(input.tellg() == static_cast<std::streamoff>(count * sizeof(std::uint32_t)), "Prefix byte count differs from uint32[137,512,512]");
  input.seekg(0);
  std::vector<std::uint32_t> prefix(count);
  input.read(reinterpret_cast<char *>(prefix.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
  require(input.good(), "Short read of radial prefix");
  return prefix;
}

struct TestCase { const char *name; std::uint32_t inner, outer; };

#ifndef RADIAL_FFT_REFERENCE_ONLY
void number(double value) { if (std::isfinite(value)) std::cout << value; else std::cout << "null"; }
#endif

void run(std::vector<std::uint32_t> &prefix, bool real) {
  const std::vector<TestCase> cases = real ? std::vector<TestCase>{
      {"real-bf", 0, 32}, {"real-abf", 24, 48}, {"real-adf", 32, 64},
      {"real-custom", 11, 57}, {"real-outer-edge", 135, 136}} : std::vector<TestCase>{
      {"constant", 0, 1}, {"off-center-impulse", 1, 2}, {"asymmetric-counts", 2, 3},
      {"oriented-wave", 3, 4}, {"zero", 4, 5}, {"uint32-extreme", 0, 136}};
#ifndef RADIAL_FFT_REFERENCE_ONLY
  RadialFftSession session(prefix);
  const auto admission = session.admission();
  require(admission.prefix_upload_bytes == prefix.size() * sizeof(std::uint32_t), "Prefix must upload exactly once");
  std::cout << "{\"admission\":true,\"device\":" << std::quoted(admission.device_name)
            << ",\"prefixUploadBytes\":" << admission.prefix_upload_bytes
            << ",\"committedBytes\":" << admission.committed_bytes
            << ",\"initializationMs\":" << admission.initialization_milliseconds
            << ",\"validationMs\":" << admission.validation_milliseconds << "}\n";
#endif
#ifndef RADIAL_FFT_REFERENCE_ONLY
  std::uint64_t generation = 100;
#endif
  for (const auto &test : cases) {
    std::vector<std::uint32_t> expected(kCount), actual(kCount);
    for (std::size_t i = 0; i < kCount; ++i) expected[i] = prefix[test.outer * kCount + i] - prefix[test.inner * kCount + i];
    const auto fft_reference = reference(expected);
    const std::string name = test.name;
    if (name == "constant" || name == "uint32-extreme") {
      const double dc = static_cast<double>(expected.front()) * kCount;
      for (std::size_t i = 0; i < kCount; ++i) require(fft_reference[i] == (i == 256 * kSide + 256 ? dc : 0.0), "Independent reference DC/fftshift control failed");
    }
    if (name == "off-center-impulse") {
      for (double value : fft_reference) require(std::abs(value - 1024.0) < 1e-8, "Independent reference impulse magnitude failed");
    }
    std::vector<float> fft(kCount);
#ifndef RADIAL_FFT_REFERENCE_ONLY
    const auto metrics = session.request(test.inner, test.outer, generation, actual, fft);
    require(actual == expected, name + ": integer annulus differs");
    require(metrics.generation == generation++, "Generation identity changed");
    require(metrics.queue_submit_count == 1 && metrics.dispatch_count == 4 && metrics.fence_wait_count == 1,
            "Request must issue one four-dispatch submission and one result fence wait");
    require(metrics.source_upload_bytes == 0 && metrics.device_wide_wait_count == 0 &&
            metrics.output_copy_bytes == 2 * kCount * sizeof(std::uint32_t), "Unexpected request copy/synchronization budget");
    const auto comparison = compare(expected, fft_reference, fft);
    std::cout << "{\"case\":" << std::quoted(name) << ",\"integerExact\":true,\"relativeL2\":" << comparison.relative_l2
              << ",\"maxAbsolute\":" << comparison.maximum_absolute << ",\"outsideTolerance\":" << comparison.outside_tolerance
              << ",\"wallMs\":" << metrics.wall_milliseconds << ",\"copyMs\":" << metrics.output_copy_milliseconds
              << ",\"gpuMs\":";
    number(metrics.gpu_total_milliseconds);
    std::cout << ",\"annulusGpuMs\":"; number(metrics.gpu_annulus_milliseconds);
    std::cout << ",\"rowsGpuMs\":"; number(metrics.gpu_fft_rows_milliseconds);
    std::cout << ",\"columnsGpuMs\":"; number(metrics.gpu_fft_columns_milliseconds);
    std::cout << ",\"magnitudeGpuMs\":"; number(metrics.gpu_magnitude_milliseconds);
    std::cout << "}\n";
    if (comparison.outside_tolerance != 0) {
      double input_energy = 0.0;
      for (auto value : expected) input_energy += static_cast<double>(value) * value;
      for (std::size_t i = 0; i < kCount; ++i) {
        const double error = std::abs(static_cast<double>(fft[i]) - fft_reference[i]);
        const double allowed = kMagnitudeRelativeTolerance * fft_reference[i] +
                               kInputL2AbsoluteTolerance * std::sqrt(input_energy);
        if (error > allowed) {
          std::cout << "{\"failedCase\":" << std::quoted(name) << ",\"row\":" << i / kSide
                    << ",\"column\":" << i % kSide << ",\"reference\":" << fft_reference[i]
                    << ",\"actual\":" << fft[i] << ",\"error\":" << error << ",\"allowed\":" << allowed << "}\n";
        }
      }
    }
    require(comparison.outside_tolerance == 0 && comparison.relative_l2 <= kRelativeL2Tolerance,
            name + ": float FFT fails predeclared scale-aware accuracy gate");
#else
    std::transform(fft_reference.begin(), fft_reference.end(), fft.begin(), [](double x) { return static_cast<float>(x); });
#endif
    // Falsification control: a 1/N-normalized transform must not pass as the required FFT.
    if (name != "zero") {
      for (auto &value : fft) value /= static_cast<float>(kCount);
      require(compare(expected, fft_reference, fft).relative_l2 > kRelativeL2Tolerance, "FFT gate accepted incorrect normalization");
    }
  }
#ifndef RADIAL_FFT_REFERENCE_ONLY
  std::vector<std::uint32_t> image(kCount);
  const auto no_fft = session.request(3, 29, generation, image);
  require(no_fft.dispatch_count == 1 && no_fft.output_copy_bytes == kCount * sizeof(std::uint32_t) &&
          std::isnan(no_fft.gpu_fft_rows_milliseconds), "FFT-off must skip FFT work and copy only integer image");
  bool rejected = false;
  try { (void)session.request(5, 5, generation, image); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "Equal detector radii must fail before GPU work");
  rejected = false;
  try { (void)session.request(0, 137, generation, image); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "Out-of-prefix detector radius must fail");
  rejected = false;
  try { (void)session.request(0, 1, generation, std::span(image).first(10)); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "Wrong-sized output must fail without changing allocation");
  (void)session.request(0, 1, generation, image); // Invalid input must not poison a valid session.
  require(session.admission().committed_bytes == admission.committed_bytes, "Workspace grew during requests");
  std::cout << "{\"status\":\"PASS\",\"gpuTested\":true,\"realPrefix\":" << (real ? "true" : "false") << ",\"cases\":" << cases.size() << "}\n";
#else
  std::cout << "{\"status\":\"PASS\",\"gpuTested\":false,\"mode\":\"independent CPU reference controls only\",\"cases\":" << cases.size() << "}\n";
#endif
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc <= 2, "Usage: radial_fft_tests [authenticated-prefix-u32-le.bin]");
    auto prefix = argc == 2 ? readPrefix(argv[1]) : syntheticPrefix();
    run(prefix, argc == 2);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Radial FFT test failed: " << error.what() << '\n';
    return 1;
  }
}
