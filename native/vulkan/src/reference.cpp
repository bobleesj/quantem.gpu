#include "quantem/gpu/vulkan/contract.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <numbers>
#include <numeric>

namespace quantem::gpu::vulkan {
namespace {

bool is_power_of_two(const std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void fft(std::vector<std::complex<double>> &values, const bool inverse) {
  const std::size_t size = values.size();
  if (!is_power_of_two(size)) {
    throw std::invalid_argument("the Android iDPC reference currently requires "
                                "power-of-two scan dimensions");
  }
  for (std::size_t index = 1, reversed = 0; index < size; ++index) {
    std::size_t bit = size >> 1;
    for (; (reversed & bit) != 0; bit >>= 1)
      reversed ^= bit;
    reversed ^= bit;
    if (index < reversed)
      std::swap(values[index], values[reversed]);
  }
  for (std::size_t length = 2; length <= size; length <<= 1) {
    const double angle =
        (inverse ? 2.0 : -2.0) * std::numbers::pi / static_cast<double>(length);
    const std::complex<double> step(std::cos(angle), std::sin(angle));
    for (std::size_t offset = 0; offset < size; offset += length) {
      std::complex<double> factor(1.0, 0.0);
      for (std::size_t index = 0; index < length / 2; ++index) {
        const std::complex<double> even = values[offset + index];
        const std::complex<double> odd =
            values[offset + index + length / 2] * factor;
        values[offset + index] = even + odd;
        values[offset + index + length / 2] = even - odd;
        factor *= step;
      }
    }
  }
  if (inverse) {
    for (auto &value : values)
      value /= static_cast<double>(size);
  }
}

void fft2(std::vector<std::complex<double>> &values, const std::size_t rows,
          const std::size_t columns, const bool inverse) {
  std::vector<std::complex<double>> line(std::max(rows, columns));
  for (std::size_t row = 0; row < rows; ++row) {
    std::copy_n(values.begin() + row * columns, columns, line.begin());
    line.resize(columns);
    fft(line, inverse);
    std::copy(line.begin(), line.end(), values.begin() + row * columns);
    line.resize(std::max(rows, columns));
  }
  for (std::size_t column = 0; column < columns; ++column) {
    line.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      line[row] = values[row * columns + column];
    }
    fft(line, inverse);
    for (std::size_t row = 0; row < rows; ++row) {
      values[row * columns + column] = line[row];
    }
    line.resize(std::max(rows, columns));
  }
}

struct CurlMoments {
  double curl_squared = 0.0;
  double divergence_squared = 0.0;
  double curl_divergence = 0.0;
};

CurlMoments curl_moments(const std::vector<float> &row,
                         const std::vector<float> &column,
                         const std::size_t rows, const std::size_t columns) {
  CurlMoments result;
  std::uint64_t count = 0;
  for (std::size_t scan_row = 1; scan_row + 1 < rows; ++scan_row) {
    for (std::size_t scan_column = 1; scan_column + 1 < columns;
         ++scan_column) {
      const auto at = [columns](const std::size_t r, const std::size_t c) {
        return r * columns + c;
      };
      const double original_curl =
          0.5 * (static_cast<double>(column[at(scan_row + 1, scan_column)]) -
                 static_cast<double>(column[at(scan_row - 1, scan_column)]) -
                 static_cast<double>(row[at(scan_row, scan_column + 1)]) +
                 static_cast<double>(row[at(scan_row, scan_column - 1)]));
      const double divergence =
          0.5 * (static_cast<double>(row[at(scan_row + 1, scan_column)]) -
                 static_cast<double>(row[at(scan_row - 1, scan_column)]) +
                 static_cast<double>(column[at(scan_row, scan_column + 1)]) -
                 static_cast<double>(column[at(scan_row, scan_column - 1)]));
      result.curl_squared += original_curl * original_curl;
      result.divergence_squared += divergence * divergence;
      result.curl_divergence += original_curl * divergence;
      ++count;
    }
  }
  if (count != 0) {
    const double scale = 1.0 / static_cast<double>(count);
    result.curl_squared *= scale;
    result.divergence_squared *= scale;
    result.curl_divergence *= scale;
  }
  return result;
}

double curl_score(const CurlMoments moments, const double cosine,
                  const double sine) {
  return cosine * cosine * moments.curl_squared +
         sine * sine * moments.divergence_squared +
         2.0 * cosine * sine * moments.curl_divergence;
}

double fft_frequency(const std::size_t index, const std::size_t size) {
  const std::int64_t signed_index =
      index < (size + 1) / 2
          ? static_cast<std::int64_t>(index)
          : static_cast<std::int64_t>(index) - static_cast<std::int64_t>(size);
  return static_cast<double>(signed_index) / static_cast<double>(size);
}

} // namespace

DerivedProducts derive_products(const ExactProducts &products) {
  const std::size_t scans =
      static_cast<std::size_t>(products.source_shape.scan_count());
  const std::size_t detector_pixels =
      static_cast<std::size_t>(products.source_shape.detector_pixel_count());
  if (products.total_intensity.size() != scans ||
      products.detector_row_moment.size() != scans ||
      products.detector_column_moment.size() != scans ||
      products.diffraction_sum.size() != detector_pixels) {
    throw std::invalid_argument(
        "exact product arrays do not match their declared source shape");
  }

  DerivedProducts derived;
  derived.source_shape = products.source_shape;
  derived.mean_diffraction.resize(detector_pixels);
  derived.center_of_mass_row.resize(scans);
  derived.center_of_mass_column.resize(scans);
  for (std::size_t pixel = 0; pixel < detector_pixels; ++pixel) {
    derived.mean_diffraction[pixel] =
        static_cast<float>(products.diffraction_sum[pixel]) /
        static_cast<float>(scans);
  }

  double mean_row = 0.0;
  double mean_column = 0.0;
  for (std::size_t scan = 0; scan < scans; ++scan) {
    const std::uint64_t total = products.total_intensity[scan];
    derived.global_total_intensity += total;
    const float row =
        total == 0 ? 0.0F
                   : static_cast<float>(products.detector_row_moment[scan]) /
                         static_cast<float>(total);
    const float column =
        total == 0 ? 0.0F
                   : static_cast<float>(products.detector_column_moment[scan]) /
                         static_cast<float>(total);
    derived.center_of_mass_row[scan] = row;
    derived.center_of_mass_column[scan] = column;
    mean_row += row;
    mean_column += column;
  }
  mean_row /= static_cast<double>(scans);
  mean_column /= static_cast<double>(scans);
  for (std::size_t scan = 0; scan < scans; ++scan) {
    derived.center_of_mass_row[scan] -= static_cast<float>(mean_row);
    derived.center_of_mass_column[scan] -= static_cast<float>(mean_column);
  }
  return derived;
}

DpcProducts derive_dpc(const DerivedProducts &products,
                       const DpcOptions &options) {
  const std::size_t rows = products.source_shape.scan_rows;
  const std::size_t columns = products.source_shape.scan_columns;
  const std::size_t scans = rows * columns;
  if (products.center_of_mass_row.size() != scans ||
      products.center_of_mass_column.size() != scans) {
    throw std::invalid_argument(
        "center-of-mass arrays do not match their declared scan shape");
  }
  if (options.automatic_rotation && options.rotation_steps < 2) {
    throw std::invalid_argument(
        "automatic DPC rotation requires at least two steps");
  }

  float angle_radians = options.fixed_rotation_degrees *
                        static_cast<float>(std::numbers::pi / 180.0);
  bool exchanged = false;
  if (options.automatic_rotation && rows >= 3 && columns >= 3) {
    double best = std::numeric_limits<double>::infinity();
    for (int order = 0; order < 2; ++order) {
      const auto &source_row = order == 0 ? products.center_of_mass_row
                                          : products.center_of_mass_column;
      const auto &source_column = order == 0 ? products.center_of_mass_column
                                             : products.center_of_mass_row;
      const CurlMoments moments =
          curl_moments(source_row, source_column, rows, columns);
      for (std::uint32_t step = 0; step < options.rotation_steps; ++step) {
        const float candidate =
            static_cast<float>(std::numbers::pi * static_cast<double>(step) /
                               static_cast<double>(options.rotation_steps - 1));
        const double score =
            curl_score(moments, std::cos(static_cast<double>(candidate)),
                       std::sin(static_cast<double>(candidate)));
        if (score < best) {
          best = score;
          angle_radians = candidate;
          exchanged = order != 0;
        }
      }
    }
  }

  const auto &source_row =
      exchanged ? products.center_of_mass_column : products.center_of_mass_row;
  const auto &source_column =
      exchanged ? products.center_of_mass_row : products.center_of_mass_column;
  const double cosine = std::cos(static_cast<double>(angle_radians));
  const double sine = std::sin(static_cast<double>(angle_radians));
  DpcProducts result;
  result.source_shape = products.source_shape;
  result.rotation_degrees = static_cast<float>(
      static_cast<double>(angle_radians) * 180.0 / std::numbers::pi);
  result.component_order_exchanged = exchanged;
  result.aligned_row.resize(scans);
  result.aligned_column.resize(scans);
  for (std::size_t index = 0; index < scans; ++index) {
    result.aligned_row[index] = static_cast<float>(cosine * source_row[index] -
                                                   sine * source_column[index]);
    result.aligned_column[index] = static_cast<float>(
        sine * source_row[index] + cosine * source_column[index]);
  }

  const auto &gradient_row =
      exchanged ? result.aligned_column : result.aligned_row;
  const auto &gradient_column =
      exchanged ? result.aligned_row : result.aligned_column;
  std::vector<std::complex<double>> row_frequency(scans);
  std::vector<std::complex<double>> column_frequency(scans);
  for (std::size_t index = 0; index < scans; ++index) {
    row_frequency[index] = gradient_row[index];
    column_frequency[index] = gradient_column[index];
  }
  fft2(row_frequency, rows, columns, false);
  fft2(column_frequency, rows, columns, false);
  std::vector<std::complex<double>> phase(scans);
  for (std::size_t row = 0; row < rows; ++row) {
    const double row_frequency_value = fft_frequency(row, rows);
    for (std::size_t column = 0; column < columns; ++column) {
      const std::size_t index = row * columns + column;
      const double column_frequency_value = fft_frequency(column, columns);
      const double squared = row_frequency_value * row_frequency_value +
                             column_frequency_value * column_frequency_value;
      phase[index] =
          squared == 0.0
              ? std::complex<double>(0.0, 0.0)
              : std::complex<double>(0.0, -0.25) *
                    (row_frequency_value * row_frequency[index] +
                     column_frequency_value * column_frequency[index]) /
                    squared;
    }
  }
  fft2(phase, rows, columns, true);
  const double mean = std::accumulate(phase.begin(), phase.end(), 0.0,
                                      [](const double total, const auto value) {
                                        return total + value.real();
                                      }) /
                      static_cast<double>(scans);
  result.integrated_phase.resize(scans);
  for (std::size_t index = 0; index < scans; ++index) {
    result.integrated_phase[index] =
        static_cast<float>(-(phase[index].real() - mean));
  }
  return result;
}

} // namespace quantem::gpu::vulkan
