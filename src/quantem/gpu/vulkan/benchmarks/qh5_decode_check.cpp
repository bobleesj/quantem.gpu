#include "quantem/gpu/vulkan/qh5_indexed_source.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using quantem::gpu::vulkan::Qh5IndexedSegment;
using quantem::gpu::vulkan::Qh5IndexedSource;
using quantem::gpu::vulkan::Shape4D;

namespace {

std::uint64_t parse_unsigned(const char *text, const char *label) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    throw std::invalid_argument(std::string(label) + " must be unsigned");
  return static_cast<std::uint64_t>(value);
}

std::uint64_t file_size(const int descriptor, const char *label) {
  struct stat status{};
  if (fstat(descriptor, &status) != 0 || status.st_size <= 0)
    throw std::runtime_error(std::string("could not inspect ") + label);
  return static_cast<std::uint64_t>(status.st_size);
}

void write_exact(const int descriptor, const std::uint8_t *bytes,
                 const std::size_t size) {
  std::size_t written = 0;
  while (written < size) {
    const ssize_t count = write(descriptor, bytes + written, size - written);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("could not write decoded frame");
    written += static_cast<std::size_t>(count);
  }
}

} // namespace

int main(const int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: qh5-decode-check SOURCE INDEX NFRAMES FRAME OUTPUT\n";
    return 2;
  }
  int source = -1;
  int index = -1;
  int output = -1;
  try {
    const std::uint64_t frame_count = parse_unsigned(argv[3], "NFRAMES");
    const std::uint64_t frame = parse_unsigned(argv[4], "FRAME");
    if (frame_count == 0 || frame >= frame_count ||
        frame_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("FRAME must be inside NFRAMES");
    }
    source = open(argv[1], O_RDONLY);
    index = open(argv[2], O_RDONLY);
    output = open(argv[5], O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (source < 0 || index < 0 || output < 0)
      throw std::runtime_error("could not open one or more paths");
    const Qh5IndexedSegment segment{source, 0, file_size(source, "source"),
                                    index,  0, file_size(index, "index")};
    auto reader = Qh5IndexedSource::open(
        {segment},
        Shape4D{1, static_cast<std::uint32_t>(frame_count), 192, 192});
    std::vector<std::uint16_t> decoded(192U * 192U);
    const auto metrics = reader->read_frames(frame, 1, decoded.data());
    write_exact(output, reinterpret_cast<const std::uint8_t *>(decoded.data()),
                decoded.size() * sizeof(std::uint16_t));
    close(output);
    close(index);
    close(source);
    std::cout << "{\"source_bytes_read\":" << metrics.source_bytes_read
              << ",\"storage_read_milliseconds\":"
              << metrics.storage_read_milliseconds
              << ",\"source_decode_milliseconds\":"
              << metrics.source_decode_milliseconds << "}\n";
    return 0;
  } catch (const std::exception &error) {
    if (output >= 0)
      close(output);
    if (index >= 0)
      close(index);
    if (source >= 0)
      close(source);
    std::cerr << "qh5-decode-check: " << error.what() << '\n';
    return 1;
  }
}
