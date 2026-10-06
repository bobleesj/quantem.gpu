#pragma once

#include "quantem/gpu/vulkan/contract.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace quantem::gpu::vulkan {

struct Qh5IndexedSegment {
  int borrowed_source_file_descriptor = -1;
  std::uint64_t source_file_offset_bytes = 0;
  std::uint64_t source_file_length_bytes = 0;
  int borrowed_index_file_descriptor = -1;
  std::uint64_t index_file_offset_bytes = 0;
  std::uint64_t index_file_length_bytes = 0;
};

struct Qh5ReadMetrics {
  std::uint64_t source_bytes_read = 0;
  std::uint64_t source_frame_count = 0;
  std::uint64_t source_block_count = 0;
  double storage_read_milliseconds = 0.0;
  double source_decode_milliseconds = 0.0;
};

/** One bounded, GPU-ready view of original HDF5 bitshuffle/LZ4 payloads.
 *
 * compressed_bytes contains only the LZ4 streams named by block_metadata;
 * it is padded to a uint32 boundary but compressed_byte_count excludes that
 * padding. block_metadata stores byte offset/length pairs in logical block
 * order. No detector value is decoded or converted by this read.
 */
struct Qh5CompressedBatch {
  std::vector<std::uint8_t> compressed_bytes;
  std::vector<std::uint32_t> block_metadata;
  std::uint32_t compressed_byte_count = 0;
  Qh5ReadMetrics metrics;
};

class Qh5SourceIoError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Validated indexed access to original HDF5 bitshuffle/LZ4 source bytes.
///
/// The small QH5IDX01 sidecars contain only byte ranges and block lengths. The
/// original HDF5 shard file descriptors remain the scientific source. Reads
/// are bounded, native uint16, and ordered by the declared logical scan.
class Qh5IndexedSource {
public:
  static std::unique_ptr<Qh5IndexedSource>
  open(const std::vector<Qh5IndexedSegment> &segments,
       const Shape4D &expected_shape);

  Qh5IndexedSource(Qh5IndexedSource &&) noexcept;
  Qh5IndexedSource &operator=(Qh5IndexedSource &&) noexcept;
  Qh5IndexedSource(const Qh5IndexedSource &) = delete;
  Qh5IndexedSource &operator=(const Qh5IndexedSource &) = delete;
  ~Qh5IndexedSource();

  [[nodiscard]] std::uint64_t frame_count() const;
  [[nodiscard]] std::uint32_t blocks_per_frame() const;
  [[nodiscard]] Qh5ReadMetrics read_frames(std::uint64_t first_frame,
                                           std::uint32_t frame_count,
                                           std::uint16_t *destination) const;

  /** Read original compressed streams for a bounded Vulkan decode batch.
   *
   * The returned storage owns all bytes and remains valid independently of
   * this source. Each decoded block is exactly 4096 uint16 values (8192 bytes)
   * and blocks remain frame-major. This method performs storage IO only.
   */
  [[nodiscard]] Qh5CompressedBatch
  read_compressed_frames(std::uint64_t first_frame,
                         std::uint32_t frame_count) const;

private:
  struct Impl;
  explicit Qh5IndexedSource(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

} // namespace quantem::gpu::vulkan
