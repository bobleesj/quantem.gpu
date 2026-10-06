#include "quantem/gpu/vulkan/c_api.h"

#include "quantem/gpu/vulkan/contract.hpp"
#include "quantem/gpu/vulkan/exact_products.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using quantem::gpu::vulkan::BenchmarkMetrics;
using quantem::gpu::vulkan::BenchmarkOptions;
using quantem::gpu::vulkan::DerivedProducts;
using quantem::gpu::vulkan::DpcProducts;
using quantem::gpu::vulkan::ExactProductExecutor;
using quantem::gpu::vulkan::ExactProducts;
using quantem::gpu::vulkan::PreparedSourceSegment;
using quantem::gpu::vulkan::Qh5IndexedSegment;
using quantem::gpu::vulkan::Qh5IndexedSource;
using quantem::gpu::vulkan::Qh5SourceIoError;
using quantem::gpu::vulkan::ScientificRequest;
using quantem::gpu::vulkan::Shape4D;
using quantem::gpu::vulkan::SourceDType;
using quantem::gpu::vulkan::SourceIoError;

using Clock = std::chrono::steady_clock;

struct OwnedFileDescriptor {
  int value = -1;

  OwnedFileDescriptor() = default;
  explicit OwnedFileDescriptor(const int source) : value(dup(source)) {
    if (value < 0)
      throw SourceIoError("failed to duplicate a file descriptor");
  }
  OwnedFileDescriptor(const OwnedFileDescriptor &) = delete;
  OwnedFileDescriptor &operator=(const OwnedFileDescriptor &) = delete;
  OwnedFileDescriptor(OwnedFileDescriptor &&other) noexcept
      : value(std::exchange(other.value, -1)) {}
  OwnedFileDescriptor &operator=(OwnedFileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value >= 0)
        close(value);
      value = std::exchange(other.value, -1);
    }
    return *this;
  }
  ~OwnedFileDescriptor() {
    if (value >= 0)
      close(value);
  }
};

struct SessionSegment {
  OwnedFileDescriptor descriptor;
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
};

std::string copy_string(const qgpu_string_view view) {
  if (view.size == 0)
    return {};
  if (view.data == nullptr)
    throw std::invalid_argument("string view has no data");
  return std::string(view.data, view.size);
}

void clear_error(qgpu_error *error) {
  if (error == nullptr || error->struct_size < sizeof(*error))
    return;
  const std::uint32_t size = error->struct_size;
  std::memset(error, 0, sizeof(*error));
  error->struct_size = size;
  error->status = QGPU_STATUS_OK;
}

qgpu_status fail(qgpu_error *error, const qgpu_status status,
                 const int native_code, const std::string_view message) {
  if (error != nullptr && error->struct_size >= sizeof(*error)) {
    const std::uint32_t size = error->struct_size;
    std::memset(error, 0, sizeof(*error));
    error->struct_size = size;
    error->status = status;
    error->native_code = native_code;
    const std::size_t count =
        std::min(message.size(), sizeof(error->message) - 1);
    std::memcpy(error->message, message.data(), count);
  }
  return status;
}

bool valid_struct(const std::uint32_t actual, const std::size_t expected) {
  return actual >= expected;
}

bool has_identity(const std::uint8_t *identity) {
  for (std::size_t index = 0; index < QGPU_SHA256_BYTES; ++index) {
    if (identity[index] != 0)
      return true;
  }
  return false;
}

SourceDType source_dtype(const qgpu_source_dtype dtype) {
  switch (dtype) {
  case QGPU_SOURCE_UINT8:
    return SourceDType::uint8;
  case QGPU_SOURCE_UINT16:
    return SourceDType::uint16;
  case QGPU_SOURCE_UINT32:
    return SourceDType::uint32;
  case QGPU_SOURCE_FLOAT32:
    break;
  }
  throw std::invalid_argument("unsupported source dtype");
}

qgpu_source_dtype c_source_dtype(const SourceDType dtype) {
  switch (dtype) {
  case SourceDType::uint8:
    return QGPU_SOURCE_UINT8;
  case SourceDType::uint16:
    return QGPU_SOURCE_UINT16;
  case SourceDType::uint32:
    return QGPU_SOURCE_UINT32;
  }
  throw std::invalid_argument("unsupported source dtype");
}

std::vector<std::uint8_t>
detector_membership(const Shape4D shape,
                    const std::array<qgpu_annular_mask, 3> &masks) {
  for (const auto &mask : masks) {
    if (!std::isfinite(mask.center_row) || !std::isfinite(mask.center_column) ||
        !std::isfinite(mask.inner_radius_exclusive) ||
        !std::isfinite(mask.outer_radius_inclusive) ||
        mask.inner_radius_exclusive < 0.0F ||
        mask.outer_radius_inclusive < mask.inner_radius_exclusive) {
      throw std::invalid_argument(
          "detector annuli must have finite ordered radii");
    }
  }
  std::vector<std::uint8_t> membership(
      static_cast<std::size_t>(shape.detector_pixel_count()), 0);
  for (std::uint32_t row = 0; row < shape.detector_rows; ++row) {
    for (std::uint32_t column = 0; column < shape.detector_columns; ++column) {
      std::uint8_t value = 0;
      for (std::size_t mask_index = 0; mask_index < masks.size();
           ++mask_index) {
        const auto &mask = masks[mask_index];
        const double delta_row = static_cast<double>(row) - mask.center_row;
        const double delta_column =
            static_cast<double>(column) - mask.center_column;
        const double radius_squared =
            delta_row * delta_row + delta_column * delta_column;
        const bool outside_inner =
            mask.inner_radius_exclusive == 0.0F ||
            radius_squared > static_cast<double>(mask.inner_radius_exclusive) *
                                 mask.inner_radius_exclusive;
        if (outside_inner &&
            radius_squared <= static_cast<double>(mask.outer_radius_inclusive) *
                                  mask.outer_radius_inclusive) {
          value |= static_cast<std::uint8_t>(1U << mask_index);
        }
      }
      membership[static_cast<std::size_t>(row) * shape.detector_columns +
                 column] = value;
    }
  }
  return membership;
}

void atomic_max(std::atomic<std::uint64_t> &target, const std::uint64_t value) {
  std::uint64_t observed = target.load(std::memory_order_relaxed);
  while (observed < value && !target.compare_exchange_weak(
                                 observed, value, std::memory_order_relaxed,
                                 std::memory_order_relaxed)) {
  }
}

qgpu_vulkan_metrics copy_metrics(const BenchmarkMetrics &source) {
  return {
      source.setup_milliseconds,
      source.storage_read_milliseconds,
      source.source_decode_milliseconds,
      source.source_staging_milliseconds,
      source.vulkan_visibility_milliseconds,
      source.first_correct_product_milliseconds,
      source.full_exact_completion_milliseconds,
      source.dpc_and_idpc_milliseconds,
      source.package_ready_milliseconds,
      source.gpu_scan_products_milliseconds,
      source.gpu_mean_diffraction_milliseconds,
      source.source_bytes_read,
      source.source_bytes_staged,
      source.explicit_copy_bytes,
      source.shader_source_bytes_read,
      source.vulkan_committed_bytes,
      source.maximum_resident_set_kibibytes,
      source.minor_page_fault_count,
      source.major_page_fault_count,
      source.queue_submit_count,
      source.fence_wait_count,
      source.device_wide_wait_count,
  };
}

} // namespace

struct qgpu_vulkan_session {
  std::uint64_t open_generation = 0;
  std::atomic<std::uint64_t> latest_generation{0};
  std::atomic<std::uint64_t> cancelled_through{0};
  std::atomic<std::uint64_t> active_generation{0};
  std::atomic<bool> active_cancel{false};
  Shape4D shape;
  SourceDType source_dtype = SourceDType::uint8;
  qgpu_source_container container = QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT8;
  std::array<std::uint8_t, QGPU_SHA256_BYTES> source_identity{};
  std::string dataset_selector;
  std::string uri_grant_identity;
  std::vector<SessionSegment> source_segments;
  std::unique_ptr<Qh5IndexedSource> indexed_source;
  OwnedFileDescriptor cache_directory;
  std::unique_ptr<ExactProductExecutor> executor;
  std::mutex execution_mutex;
  std::mutex event_mutex;
  std::deque<qgpu_vulkan_event> events;
  std::uint64_t next_event_sequence = 1;
};

struct qgpu_vulkan_result {
  std::uint64_t generation = 0;
  std::array<std::uint8_t, QGPU_SHA256_BYTES> source_identity{};
  ExactProducts exact;
  DerivedProducts derived;
  DpcProducts dpc;
  BenchmarkMetrics metrics;
};

namespace {

void push_event(qgpu_vulkan_session *session, const qgpu_event_type type,
                const qgpu_status status, const std::uint64_t generation,
                const std::uint32_t completed_scan_count = 0,
                const std::uint32_t total_scan_count = 0,
                const BenchmarkMetrics *metrics = nullptr) noexcept {
  try {
    if (session == nullptr)
      return;
    qgpu_vulkan_event event{};
    event.struct_size = sizeof(event);
    event.generation = generation;
    event.type = type;
    event.status = status;
    event.completed_scan_count = completed_scan_count;
    event.total_scan_count = total_scan_count;
    std::memcpy(event.source_identity_sha256, session->source_identity.data(),
                QGPU_SHA256_BYTES);
    if (metrics != nullptr)
      event.metrics = copy_metrics(*metrics);
    std::lock_guard lock(session->event_mutex);
    event.sequence = session->next_event_sequence++;
    session->events.push_back(event);
  } catch (...) {
    // Events are advisory; allocation failure must not unwind through Vulkan.
  }
}

void publish_progress(void *context, const std::uint32_t completed_scan_count,
                      const std::uint32_t total_scan_count,
                      const bool first_product_ready) {
  auto *session = static_cast<qgpu_vulkan_session *>(context);
  const std::uint64_t generation = session->active_generation.load();
  if (first_product_ready) {
    push_event(session, QGPU_EVENT_FIRST_PRODUCT_READY, QGPU_STATUS_OK,
               generation, completed_scan_count, total_scan_count);
  }
  push_event(session, QGPU_EVENT_PROGRESS, QGPU_STATUS_OK, generation,
             completed_scan_count, total_scan_count);
}

} // namespace

extern "C" uint32_t qgpu_vulkan_abi_version(void) {
  return QGPU_VULKAN_ABI_VERSION;
}

extern "C" qgpu_status
qgpu_vulkan_open_v1(const qgpu_vulkan_open_request *request,
                    qgpu_vulkan_session **output_session, qgpu_error *error) {
  clear_error(error);
  if (output_session != nullptr)
    *output_session = nullptr;
  try {
    if (request == nullptr || output_session == nullptr ||
        !valid_struct(request->struct_size, sizeof(*request))) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "invalid open request");
    }
    if (request->abi_version != QGPU_VULKAN_ABI_VERSION) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "unsupported ABI version");
    }
    if (request->generation == 0 ||
        !valid_struct(request->dataset.struct_size, sizeof(request->dataset))) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "generation and versioned dataset descriptor are required");
    }
    const bool prepared_uint8 =
        request->container == QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT8;
    const bool prepared_uint16 =
        request->container == QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT16;
    const bool indexed_uint16 =
        request->container == QGPU_SOURCE_QH5_INDEXED_BITSHUFFLE_LZ4;
    if (request->container == QGPU_SOURCE_HDF5_BITSHUFFLE_LZ4) {
      return fail(error, QGPU_STATUS_UNSUPPORTED_CONTAINER, 0,
                  "direct HDF5 metadata discovery is not admitted; provide "
                  "QH5IDX01 sidecars with the original HDF5 shards");
    }
    if (!prepared_uint8 && !prepared_uint16 && !indexed_uint16) {
      return fail(error, QGPU_STATUS_UNSUPPORTED_CONTAINER, 0,
                  "unsupported Android source container");
    }
    if ((prepared_uint8 &&
         request->dataset.source_dtype != QGPU_SOURCE_UINT8) ||
        ((prepared_uint16 || indexed_uint16) &&
         request->dataset.source_dtype != QGPU_SOURCE_UINT16)) {
      return fail(error, QGPU_STATUS_UNSUPPORTED_DTYPE, 0,
                  "source container and declared dtype disagree");
    }
    if (request->ordered_source_segments == nullptr ||
        request->source_segment_count == 0) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "source segments are required");
    }
    if (!has_identity(request->dataset.source_identity_sha256)) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "source identity SHA-256 is required");
    }
    auto session = std::make_unique<qgpu_vulkan_session>();
    session->open_generation = request->generation;
    session->latest_generation.store(request->generation);
    session->shape = {
        request->dataset.scan_rows,
        request->dataset.scan_columns,
        request->dataset.detector_rows,
        request->dataset.detector_columns,
    };
    session->source_dtype = source_dtype(request->dataset.source_dtype);
    session->container = request->container;
    const ScientificRequest scientific{
        session->shape,
        session->source_dtype,
        request->dataset.scan_bin,
        request->dataset.detector_bin,
        request->dataset.crop_is_none,
        true,
        true,
        true,
        true,
        true,
    };
    quantem::gpu::vulkan::validate_scientific_request(scientific);
    std::memcpy(session->source_identity.data(),
                request->dataset.source_identity_sha256, QGPU_SHA256_BYTES);
    session->dataset_selector = copy_string(request->dataset.dataset_selector);
    session->uri_grant_identity = copy_string(request->uri_grant_identity);
    session->source_segments.reserve(request->source_segment_count);
    std::uint64_t segment_bytes = 0;
    for (std::size_t index = 0; index < request->source_segment_count;
         ++index) {
      const auto &input = request->ordered_source_segments[index];
      if (!valid_struct(input.struct_size, sizeof(input)) ||
          input.borrowed_file_descriptor < 0 || input.length_bytes == 0 ||
          input.source_ordinal != index || !has_identity(input.sha256)) {
        return fail(
            error, QGPU_STATUS_INVALID_ARGUMENT, 0,
            "source segments must be valid and ordered from ordinal zero");
      }
      struct stat status{};
      if (fstat(input.borrowed_file_descriptor, &status) != 0 ||
          status.st_size < 0 ||
          input.file_offset_bytes >
              static_cast<std::uint64_t>(status.st_size) ||
          input.length_bytes > static_cast<std::uint64_t>(status.st_size) -
                                   input.file_offset_bytes) {
        return fail(error, QGPU_STATUS_SOURCE_IO, 0,
                    "source segment exceeds its file");
      }
      if (input.length_bytes >
          std::numeric_limits<std::uint64_t>::max() - segment_bytes) {
        return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                    "source byte count overflows");
      }
      SessionSegment segment;
      segment.descriptor = OwnedFileDescriptor(input.borrowed_file_descriptor);
      segment.offset = input.file_offset_bytes;
      segment.length = input.length_bytes;
      session->source_segments.push_back(std::move(segment));
      segment_bytes += input.length_bytes;
    }
    if (prepared_uint8 || prepared_uint16) {
      const std::uint64_t value_count = session->shape.value_count();
      const std::uint64_t value_bytes =
          quantem::gpu::vulkan::bytes_per_value(session->source_dtype);
      if (value_count >
          std::numeric_limits<std::uint64_t>::max() / value_bytes) {
        return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                    "prepared source byte count overflows");
      }
      const std::uint64_t expected_bytes = value_count * value_bytes;
      if (segment_bytes != expected_bytes) {
        return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                    "prepared source bytes do not match the exact declared "
                    "geometry and dtype");
      }
    } else {
      if (request->ordered_qh5_index_segments == nullptr ||
          request->qh5_index_segment_count != request->source_segment_count) {
        return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                    "indexed QH5 loading requires one ordered index for every "
                    "original HDF5 source shard");
      }
      std::vector<Qh5IndexedSegment> indexed_segments;
      indexed_segments.reserve(request->source_segment_count);
      for (std::size_t index = 0; index < request->source_segment_count;
           ++index) {
        const auto &source = request->ordered_source_segments[index];
        const auto &input = request->ordered_qh5_index_segments[index];
        if (!valid_struct(input.struct_size, sizeof(input)) ||
            input.borrowed_file_descriptor < 0 || input.length_bytes == 0 ||
            input.source_ordinal != index || !has_identity(input.sha256)) {
          return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                      "QH5 index segments must be valid and ordered from "
                      "ordinal zero");
        }
        struct stat status{};
        if (fstat(input.borrowed_file_descriptor, &status) != 0 ||
            status.st_size < 0 ||
            input.file_offset_bytes >
                static_cast<std::uint64_t>(status.st_size) ||
            input.length_bytes > static_cast<std::uint64_t>(status.st_size) -
                                     input.file_offset_bytes) {
          return fail(error, QGPU_STATUS_SOURCE_IO, 0,
                      "QH5 index segment exceeds its file");
        }
        indexed_segments.push_back({
            source.borrowed_file_descriptor,
            source.file_offset_bytes,
            source.length_bytes,
            input.borrowed_file_descriptor,
            input.file_offset_bytes,
            input.length_bytes,
        });
      }
      session->indexed_source =
          Qh5IndexedSource::open(indexed_segments, session->shape);
    }
    if (request->borrowed_cache_directory_file_descriptor >= 0) {
      session->cache_directory = OwnedFileDescriptor(
          request->borrowed_cache_directory_file_descriptor);
    }
    push_event(session.get(), QGPU_EVENT_OPENED, QGPU_STATUS_OK,
               request->generation, 0,
               static_cast<std::uint32_t>(session->shape.scan_count()));
    *output_session = session.release();
    return QGPU_STATUS_OK;
  } catch (const std::invalid_argument &exception) {
    return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0, exception.what());
  } catch (const SourceIoError &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const Qh5SourceIoError &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const std::exception &exception) {
    return fail(error, QGPU_STATUS_VULKAN, 0, exception.what());
  }
}

extern "C" qgpu_status qgpu_vulkan_request_products_v1(
    qgpu_vulkan_session *session, const qgpu_vulkan_product_request *request,
    qgpu_vulkan_result **output_result, qgpu_error *error) {
  clear_error(error);
  if (output_result != nullptr)
    *output_result = nullptr;
  try {
    if (session == nullptr || request == nullptr || output_result == nullptr ||
        !valid_struct(request->struct_size, sizeof(*request))) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "invalid product request");
    }
    std::unique_lock lock(session->execution_mutex);
    if (request->generation < session->latest_generation.load()) {
      return fail(error, QGPU_STATUS_STALE_GENERATION, 0,
                  "request generation is stale");
    }
    session->latest_generation.store(request->generation);
    if (request->generation <= session->cancelled_through.load()) {
      return fail(error, QGPU_STATUS_CANCELLED, 0,
                  "request generation is cancelled");
    }
    session->active_cancel.store(false);
    session->active_generation.store(request->generation);
    if (request->generation <= session->cancelled_through.load()) {
      session->active_cancel.store(true);
    }
    const auto package_started = Clock::now();
    const std::uint32_t total_scan_count =
        static_cast<std::uint32_t>(session->shape.scan_count());
    push_event(session, QGPU_EVENT_SOURCE_PLAN_READY, QGPU_STATUS_OK,
               request->generation, 0, total_scan_count);
    const std::array<qgpu_annular_mask, 3> masks{
        request->bright_field,
        request->annular_bright_field,
        request->annular_dark_field,
    };
    const std::vector<std::uint8_t> membership =
        detector_membership(session->shape, masks);
    BenchmarkOptions options;
    options.source_shape = session->shape;
    options.selected_scan_row = request->selected_scan_row;
    options.selected_scan_column = request->selected_scan_column;
    options.shard_scan_rows =
        request->shard_scan_rows == 0 ? 2 : request->shard_scan_rows;
    options.staging_ring_depth =
        request->staging_ring_depth == 0 ? 2 : request->staging_ring_depth;
    options.cancellation_flag = &session->active_cancel;
    options.progress_callback = publish_progress;
    options.progress_context = session;
    std::vector<PreparedSourceSegment> segments;
    segments.reserve(session->source_segments.size());
    for (const auto &source : session->source_segments) {
      segments.push_back(
          {source.descriptor.value, source.offset, source.length});
    }
    auto result = std::make_unique<qgpu_vulkan_result>();
    result->generation = request->generation;
    result->source_identity = session->source_identity;
    if (session->executor == nullptr)
      session->executor = ExactProductExecutor::create();
    switch (session->container) {
    case QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT8:
      result->metrics = session->executor->run_prepared_contiguous_u8(
          options, segments, membership, &result->exact);
      break;
    case QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT16:
      result->metrics = session->executor->run_prepared_contiguous_u16(
          options, segments, membership, &result->exact);
      break;
    case QGPU_SOURCE_QH5_INDEXED_BITSHUFFLE_LZ4:
      if (session->indexed_source == nullptr)
        throw std::logic_error("indexed QH5 session has no source reader");
      result->metrics = session->executor->run_indexed_qh5_u16(
          options, *session->indexed_source, membership, &result->exact);
      break;
    case QGPU_SOURCE_HDF5_BITSHUFFLE_LZ4:
      throw std::logic_error("direct HDF5 source reached execution");
    }
    result->derived = quantem::gpu::vulkan::derive_products(result->exact);
    const auto dpc_started = Clock::now();
    result->dpc = quantem::gpu::vulkan::derive_dpc(result->derived);
    result->metrics.dpc_and_idpc_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - dpc_started)
            .count();
    result->metrics.package_ready_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  package_started)
            .count();
    push_event(session, QGPU_EVENT_PRODUCTS_READY, QGPU_STATUS_OK,
               request->generation, total_scan_count, total_scan_count,
               &result->metrics);
    session->active_generation.store(0);
    *output_result = result.release();
    return QGPU_STATUS_OK;
  } catch (const quantem::gpu::vulkan::OperationCancelled &exception) {
    push_event(session, QGPU_EVENT_CANCELLED, QGPU_STATUS_CANCELLED,
               request == nullptr ? 0 : request->generation);
    session->active_generation.store(0);
    return fail(error, QGPU_STATUS_CANCELLED, 0, exception.what());
  } catch (const std::invalid_argument &exception) {
    push_event(session, QGPU_EVENT_ERROR, QGPU_STATUS_INVALID_ARGUMENT,
               request == nullptr ? 0 : request->generation);
    session->active_generation.store(0);
    return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0, exception.what());
  } catch (const SourceIoError &exception) {
    push_event(session, QGPU_EVENT_ERROR, QGPU_STATUS_SOURCE_IO,
               request == nullptr ? 0 : request->generation);
    session->active_generation.store(0);
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const Qh5SourceIoError &exception) {
    push_event(session, QGPU_EVENT_ERROR, QGPU_STATUS_SOURCE_IO,
               request == nullptr ? 0 : request->generation);
    session->active_generation.store(0);
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const std::runtime_error &exception) {
    push_event(session, QGPU_EVENT_ERROR, QGPU_STATUS_VULKAN,
               request == nullptr ? 0 : request->generation);
    session->active_generation.store(0);
    return fail(error, QGPU_STATUS_VULKAN, 0, exception.what());
  } catch (const std::exception &exception) {
    push_event(session, QGPU_EVENT_ERROR, QGPU_STATUS_INTERNAL,
               request == nullptr ? 0 : request->generation);
    session->active_generation.store(0);
    return fail(error, QGPU_STATUS_INTERNAL, 0, exception.what());
  }
}

extern "C" qgpu_status qgpu_vulkan_read_selected_diffraction_v1(
    qgpu_vulkan_session *session,
    const qgpu_vulkan_selected_diffraction_request *request,
    qgpu_vulkan_selected_diffraction_result *output_result, qgpu_error *error) {
  const auto total_started = Clock::now();
  clear_error(error);
  try {
    if (session == nullptr || request == nullptr || output_result == nullptr ||
        !valid_struct(request->struct_size, sizeof(*request)) ||
        !valid_struct(output_result->struct_size, sizeof(*output_result))) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "invalid selected diffraction request");
    }
    const std::uint32_t output_struct_size = output_result->struct_size;
    std::memset(output_result, 0, sizeof(*output_result));
    output_result->struct_size = output_struct_size;
    if (session->container != QGPU_SOURCE_QH5_INDEXED_BITSHUFFLE_LZ4 ||
        session->source_dtype != SourceDType::uint16 ||
        session->indexed_source == nullptr) {
      return fail(error, QGPU_STATUS_UNSUPPORTED_CONTAINER, 0,
                  "selected diffraction reads require an indexed uint16 QH5 "
                  "session");
    }
    if (request->generation == 0 || request->destination_uint16 == nullptr) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "generation and caller-owned uint16 destination are "
                  "required");
    }
    if (request->scan_row >= session->shape.scan_rows ||
        request->scan_column >= session->shape.scan_columns) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "selected scan coordinate is outside the logical scan");
    }
    const std::uint64_t detector_value_count =
        session->shape.detector_pixel_count();
    if (detector_value_count > std::numeric_limits<std::size_t>::max() ||
        request->destination_value_capacity < detector_value_count) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "selected diffraction destination is smaller than one "
                  "detector frame");
    }

    std::unique_lock lock(session->execution_mutex);
    if (request->generation < session->latest_generation.load()) {
      return fail(error, QGPU_STATUS_STALE_GENERATION, 0,
                  "request generation is stale");
    }
    session->latest_generation.store(request->generation);
    if (request->generation <= session->cancelled_through.load()) {
      return fail(error, QGPU_STATUS_CANCELLED, 0,
                  "request generation is cancelled");
    }
    const std::uint64_t frame = static_cast<std::uint64_t>(request->scan_row) *
                                    session->shape.scan_columns +
                                request->scan_column;
    const auto metrics = session->indexed_source->read_frames(
        frame, 1U, request->destination_uint16);

    output_result->generation = request->generation;
    std::memcpy(output_result->source_identity_sha256,
                session->source_identity.data(), QGPU_SHA256_BYTES);
    output_result->scan_row = request->scan_row;
    output_result->scan_column = request->scan_column;
    output_result->detector_rows = session->shape.detector_rows;
    output_result->detector_columns = session->shape.detector_columns;
    output_result->source_dtype = QGPU_SOURCE_UINT16;
    output_result->destination_value_count =
        static_cast<std::size_t>(detector_value_count);
    output_result->metrics.storage_read_milliseconds =
        metrics.storage_read_milliseconds;
    output_result->metrics.source_decode_milliseconds =
        metrics.source_decode_milliseconds;
    output_result->metrics.source_bytes_read = metrics.source_bytes_read;
    output_result->metrics.source_frame_count = metrics.source_frame_count;
    output_result->metrics.source_block_count =
        static_cast<std::uint32_t>(metrics.source_block_count);
    output_result->metrics.full_product_source_frames_read = 0U;
    output_result->metrics.vulkan_queue_submit_count = 0U;
    output_result->metrics.total_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - total_started)
            .count();
    return QGPU_STATUS_OK;
  } catch (const Qh5SourceIoError &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const SourceIoError &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const std::invalid_argument &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const std::exception &exception) {
    return fail(error, QGPU_STATUS_INTERNAL, 0, exception.what());
  }
}

extern "C" qgpu_status qgpu_vulkan_read_selected_diffraction_gpu_v1(
    qgpu_vulkan_session *session,
    const qgpu_vulkan_selected_diffraction_request *request,
    qgpu_vulkan_selected_diffraction_result *output_result, qgpu_error *error) {
  clear_error(error);
  try {
    if (session == nullptr || request == nullptr || output_result == nullptr ||
        !valid_struct(request->struct_size, sizeof(*request)) ||
        !valid_struct(output_result->struct_size, sizeof(*output_result))) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "invalid GPU selected diffraction request");
    }
    const std::uint32_t output_struct_size = output_result->struct_size;
    std::memset(output_result, 0, sizeof(*output_result));
    output_result->struct_size = output_struct_size;
    if (session->container != QGPU_SOURCE_QH5_INDEXED_BITSHUFFLE_LZ4 ||
        session->source_dtype != SourceDType::uint16 ||
        session->indexed_source == nullptr) {
      return fail(error, QGPU_STATUS_UNSUPPORTED_CONTAINER, 0,
                  "GPU selected diffraction reads require an indexed uint16 "
                  "QH5 session");
    }
    if (request->generation == 0 || request->destination_uint16 == nullptr) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "generation and caller-owned uint16 destination are required");
    }
    if (request->scan_row >= session->shape.scan_rows ||
        request->scan_column >= session->shape.scan_columns) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "selected scan coordinate is outside the logical scan");
    }
    const std::uint64_t detector_value_count =
        session->shape.detector_pixel_count();
    if (detector_value_count > std::numeric_limits<std::size_t>::max() ||
        request->destination_value_capacity < detector_value_count) {
      return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                  "selected diffraction destination is smaller than one frame");
    }

    std::unique_lock lock(session->execution_mutex);
    if (request->generation < session->latest_generation.load()) {
      return fail(error, QGPU_STATUS_STALE_GENERATION, 0,
                  "request generation is stale");
    }
    session->latest_generation.store(request->generation);
    if (request->generation <= session->cancelled_through.load()) {
      return fail(error, QGPU_STATUS_CANCELLED, 0,
                  "request generation is cancelled");
    }
    if (session->executor == nullptr)
      session->executor = ExactProductExecutor::create();
    const std::uint64_t frame = static_cast<std::uint64_t>(request->scan_row) *
                                    session->shape.scan_columns +
                                request->scan_column;
    const auto metrics = session->executor->read_indexed_qh5_frame_u16(
        *session->indexed_source, frame, request->destination_uint16,
        request->destination_value_capacity);

    output_result->generation = request->generation;
    std::memcpy(output_result->source_identity_sha256,
                session->source_identity.data(), QGPU_SHA256_BYTES);
    output_result->scan_row = request->scan_row;
    output_result->scan_column = request->scan_column;
    output_result->detector_rows = session->shape.detector_rows;
    output_result->detector_columns = session->shape.detector_columns;
    output_result->source_dtype = QGPU_SOURCE_UINT16;
    output_result->destination_value_count =
        static_cast<std::size_t>(detector_value_count);
    output_result->metrics.total_milliseconds = metrics.total_milliseconds;
    output_result->metrics.storage_read_milliseconds =
        metrics.storage_read_milliseconds;
    output_result->metrics.source_decode_milliseconds =
        metrics.gpu_decode_milliseconds;
    output_result->metrics.source_bytes_read = metrics.source_bytes_read;
    output_result->metrics.source_frame_count = metrics.source_frame_count;
    output_result->metrics.source_block_count =
        static_cast<std::uint32_t>(metrics.source_block_count);
    output_result->metrics.full_product_source_frames_read = 0U;
    output_result->metrics.vulkan_queue_submit_count =
        metrics.queue_submit_count;
    return QGPU_STATUS_OK;
  } catch (const Qh5SourceIoError &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const SourceIoError &exception) {
    return fail(error, QGPU_STATUS_SOURCE_IO, 0, exception.what());
  } catch (const std::invalid_argument &exception) {
    return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0, exception.what());
  } catch (const std::runtime_error &exception) {
    return fail(error, QGPU_STATUS_VULKAN, 0, exception.what());
  } catch (const std::exception &exception) {
    return fail(error, QGPU_STATUS_INTERNAL, 0, exception.what());
  }
}

extern "C" qgpu_status
qgpu_vulkan_poll_event_v1(qgpu_vulkan_session *session,
                          qgpu_vulkan_event *output_event, qgpu_error *error) {
  clear_error(error);
  if (session == nullptr || output_event == nullptr ||
      !valid_struct(output_event->struct_size, sizeof(*output_event))) {
    return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                "invalid event poll request");
  }
  std::lock_guard lock(session->event_mutex);
  if (session->events.empty())
    return QGPU_STATUS_NO_EVENT;
  const std::uint32_t size = output_event->struct_size;
  *output_event = session->events.front();
  output_event->struct_size = size;
  session->events.pop_front();
  return QGPU_STATUS_OK;
}

extern "C" qgpu_status
qgpu_vulkan_result_view_v1(const qgpu_vulkan_result *result,
                           qgpu_vulkan_result_view *output_view,
                           qgpu_error *error) {
  clear_error(error);
  if (result == nullptr || output_view == nullptr ||
      !valid_struct(output_view->struct_size, sizeof(*output_view))) {
    return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0,
                "invalid result view request");
  }
  const std::uint32_t size = output_view->struct_size;
  std::memset(output_view, 0, sizeof(*output_view));
  output_view->struct_size = size;
  output_view->generation = result->generation;
  std::memcpy(output_view->source_identity_sha256,
              result->source_identity.data(), QGPU_SHA256_BYTES);
  const Shape4D shape = result->exact.source_shape;
  output_view->scan_rows = shape.scan_rows;
  output_view->scan_columns = shape.scan_columns;
  output_view->detector_rows = shape.detector_rows;
  output_view->detector_columns = shape.detector_columns;
  output_view->source_dtype = c_source_dtype(result->exact.source_dtype);
  output_view->selected_diffraction_uint8 =
      result->exact.selected_diffraction_uint8.empty()
          ? nullptr
          : result->exact.selected_diffraction_uint8.data();
  output_view->selected_diffraction_uint16 =
      result->exact.selected_diffraction_uint16.empty()
          ? nullptr
          : result->exact.selected_diffraction_uint16.data();
  output_view->selected_diffraction_count =
      result->exact.source_dtype == SourceDType::uint8
          ? result->exact.selected_diffraction_uint8.size()
          : result->exact.selected_diffraction_uint16.size();
  output_view->bright_field_uint64 = result->exact.band1.data();
  output_view->annular_bright_field_uint64 = result->exact.band2.data();
  output_view->annular_dark_field_uint64 = result->exact.band4.data();
  output_view->total_intensity_uint64 = result->exact.total_intensity.data();
  output_view->diffraction_sum_uint64 = result->exact.diffraction_sum.data();
  output_view->mean_diffraction_float32 =
      result->derived.mean_diffraction.data();
  output_view->center_of_mass_row_float32 =
      result->derived.center_of_mass_row.data();
  output_view->center_of_mass_column_float32 =
      result->derived.center_of_mass_column.data();
  output_view->dpc_row_float32 = result->dpc.aligned_row.data();
  output_view->dpc_column_float32 = result->dpc.aligned_column.data();
  output_view->idpc_float32 = result->dpc.integrated_phase.data();
  output_view->scan_product_count = result->exact.total_intensity.size();
  output_view->detector_product_count = result->exact.diffraction_sum.size();
  output_view->global_total_intensity_uint64 =
      result->derived.global_total_intensity;
  output_view->dpc_rotation_degrees = result->dpc.rotation_degrees;
  output_view->dpc_component_order_exchanged =
      result->dpc.component_order_exchanged;
  output_view->parity_checked = result->metrics.parity_checked;
  output_view->mismatch_count = result->metrics.mismatch_count;
  output_view->metrics = copy_metrics(result->metrics);
  return QGPU_STATUS_OK;
}

extern "C" void qgpu_vulkan_result_release_v1(qgpu_vulkan_result **result) {
  if (result == nullptr || *result == nullptr)
    return;
  delete *result;
  *result = nullptr;
}

extern "C" qgpu_status
qgpu_vulkan_cancel_through_generation_v1(qgpu_vulkan_session *session,
                                         const uint64_t generation,
                                         qgpu_error *error) {
  clear_error(error);
  if (session == nullptr) {
    return fail(error, QGPU_STATUS_INVALID_ARGUMENT, 0, "session is null");
  }
  atomic_max(session->cancelled_through, generation);
  const std::uint64_t active = session->active_generation.load();
  if (active != 0 && active <= generation)
    session->active_cancel.store(true);
  push_event(session, QGPU_EVENT_CANCEL_REQUESTED, QGPU_STATUS_CANCELLED,
             generation);
  return QGPU_STATUS_OK;
}

extern "C" void qgpu_vulkan_close_v1(qgpu_vulkan_session **session) {
  if (session == nullptr || *session == nullptr)
    return;
  qgpu_vulkan_session *value = *session;
  value->active_cancel.store(true);
  {
    std::unique_lock lock(value->execution_mutex);
  }
  delete value;
  *session = nullptr;
}
