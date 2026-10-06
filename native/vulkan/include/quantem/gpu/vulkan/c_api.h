#ifndef QUANTEM_GPU_VULKAN_C_API_H
#define QUANTEM_GPU_VULKAN_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QGPU_VULKAN_ABI_VERSION 2u
#define QGPU_SHA256_BYTES 32u

typedef struct qgpu_vulkan_session qgpu_vulkan_session;
typedef struct qgpu_vulkan_result qgpu_vulkan_result;

typedef enum qgpu_status {
  QGPU_STATUS_OK = 0,
  QGPU_STATUS_INVALID_ARGUMENT = 1,
  QGPU_STATUS_UNSUPPORTED_CONTAINER = 2,
  QGPU_STATUS_UNSUPPORTED_DTYPE = 3,
  QGPU_STATUS_SOURCE_IO = 4,
  QGPU_STATUS_VULKAN = 5,
  QGPU_STATUS_CANCELLED = 6,
  QGPU_STATUS_STALE_GENERATION = 7,
  QGPU_STATUS_INTERNAL = 8,
  QGPU_STATUS_NO_EVENT = 9
} qgpu_status;

typedef enum qgpu_event_type {
  QGPU_EVENT_OPENED = 1,
  QGPU_EVENT_SOURCE_PLAN_READY = 2,
  QGPU_EVENT_PROGRESS = 3,
  QGPU_EVENT_FIRST_PRODUCT_READY = 4,
  QGPU_EVENT_PRODUCTS_READY = 5,
  QGPU_EVENT_CANCEL_REQUESTED = 6,
  QGPU_EVENT_CANCELLED = 7,
  QGPU_EVENT_ERROR = 8
} qgpu_event_type;

typedef enum qgpu_source_container {
  QGPU_SOURCE_HDF5_BITSHUFFLE_LZ4 = 1,
  QGPU_SOURCE_QH5_INDEXED_BITSHUFFLE_LZ4 = 2,
  QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT8 = 3,
  QGPU_SOURCE_PREPARED_CONTIGUOUS_UINT16 = 4
} qgpu_source_container;

typedef enum qgpu_source_dtype {
  QGPU_SOURCE_UINT8 = 1,
  QGPU_SOURCE_UINT16 = 2,
  QGPU_SOURCE_UINT32 = 3,
  QGPU_SOURCE_FLOAT32 = 4
} qgpu_source_dtype;

typedef struct qgpu_string_view {
  const char *data;
  size_t size;
} qgpu_string_view;

typedef struct qgpu_error {
  uint32_t struct_size;
  qgpu_status status;
  int32_t native_code;
  char message[256];
} qgpu_error;

typedef struct qgpu_source_segment {
  uint32_t struct_size;
  int32_t borrowed_file_descriptor;
  uint64_t file_offset_bytes;
  uint64_t length_bytes;
  uint32_t source_ordinal;
  uint8_t sha256[QGPU_SHA256_BYTES];
} qgpu_source_segment;

typedef struct qgpu_qh5_index_segment {
  uint32_t struct_size;
  int32_t borrowed_file_descriptor;
  uint64_t file_offset_bytes;
  uint64_t length_bytes;
  uint32_t source_ordinal;
  uint8_t sha256[QGPU_SHA256_BYTES];
} qgpu_qh5_index_segment;

typedef struct qgpu_dataset_descriptor {
  uint32_t struct_size;
  uint32_t scan_rows;
  uint32_t scan_columns;
  uint32_t detector_rows;
  uint32_t detector_columns;
  qgpu_source_dtype source_dtype;
  uint32_t scan_bin;
  uint32_t detector_bin;
  bool crop_is_none;
  qgpu_string_view dataset_selector;
  uint8_t source_identity_sha256[QGPU_SHA256_BYTES];
  uint8_t calibration_sha256[QGPU_SHA256_BYTES];
  uint8_t pixel_mask_sha256[QGPU_SHA256_BYTES];
} qgpu_dataset_descriptor;

typedef struct qgpu_vulkan_open_request {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t generation;
  qgpu_source_container container;
  qgpu_dataset_descriptor dataset;
  const qgpu_source_segment *ordered_source_segments;
  size_t source_segment_count;
  int32_t borrowed_cache_directory_file_descriptor;
  qgpu_string_view uri_grant_identity;
  const qgpu_qh5_index_segment *ordered_qh5_index_segments;
  size_t qh5_index_segment_count;
} qgpu_vulkan_open_request;

typedef struct qgpu_annular_mask {
  float center_row;
  float center_column;
  float inner_radius_exclusive;
  float outer_radius_inclusive;
} qgpu_annular_mask;

typedef struct qgpu_vulkan_product_request {
  uint32_t struct_size;
  uint64_t generation;
  uint32_t selected_scan_row;
  uint32_t selected_scan_column;
  qgpu_annular_mask bright_field;
  qgpu_annular_mask annular_bright_field;
  qgpu_annular_mask annular_dark_field;
  uint32_t shard_scan_rows;
  uint32_t staging_ring_depth;
  uint32_t priority;
} qgpu_vulkan_product_request;

typedef struct qgpu_vulkan_selected_diffraction_request {
  uint32_t struct_size;
  uint64_t generation;
  uint32_t scan_row;
  uint32_t scan_column;
  uint16_t *destination_uint16;
  size_t destination_value_capacity;
} qgpu_vulkan_selected_diffraction_request;

typedef struct qgpu_vulkan_selected_diffraction_metrics {
  double total_milliseconds;
  double storage_read_milliseconds;
  double source_decode_milliseconds;
  uint64_t source_bytes_read;
  uint64_t source_frame_count;
  uint32_t source_block_count;
  uint64_t full_product_source_frames_read;
  uint32_t vulkan_queue_submit_count;
} qgpu_vulkan_selected_diffraction_metrics;

typedef struct qgpu_vulkan_selected_diffraction_result {
  uint32_t struct_size;
  uint64_t generation;
  uint8_t source_identity_sha256[QGPU_SHA256_BYTES];
  uint32_t scan_row;
  uint32_t scan_column;
  uint32_t detector_rows;
  uint32_t detector_columns;
  qgpu_source_dtype source_dtype;
  size_t destination_value_count;
  qgpu_vulkan_selected_diffraction_metrics metrics;
} qgpu_vulkan_selected_diffraction_result;

typedef struct qgpu_vulkan_metrics {
  double setup_milliseconds;
  double storage_read_milliseconds;
  double source_decode_milliseconds;
  double staging_milliseconds;
  double vulkan_visibility_milliseconds;
  double first_product_milliseconds;
  double full_exact_milliseconds;
  double dpc_and_idpc_milliseconds;
  double package_ready_milliseconds;
  double gpu_scan_products_milliseconds;
  double gpu_mean_diffraction_milliseconds;
  uint64_t source_bytes_read;
  uint64_t source_bytes_staged;
  uint64_t explicit_copy_bytes;
  uint64_t shader_source_bytes_read;
  uint64_t vulkan_committed_bytes;
  uint64_t maximum_resident_set_kibibytes;
  uint64_t minor_page_fault_count;
  uint64_t major_page_fault_count;
  uint32_t queue_submit_count;
  uint32_t fence_wait_count;
  uint32_t device_wide_wait_count;
} qgpu_vulkan_metrics;

typedef struct qgpu_vulkan_event {
  uint32_t struct_size;
  uint64_t sequence;
  uint64_t generation;
  qgpu_event_type type;
  qgpu_status status;
  uint8_t source_identity_sha256[QGPU_SHA256_BYTES];
  uint32_t completed_scan_count;
  uint32_t total_scan_count;
  qgpu_vulkan_metrics metrics;
} qgpu_vulkan_event;

typedef struct qgpu_vulkan_result_view {
  uint32_t struct_size;
  uint64_t generation;
  uint8_t source_identity_sha256[QGPU_SHA256_BYTES];
  uint32_t scan_rows;
  uint32_t scan_columns;
  uint32_t detector_rows;
  uint32_t detector_columns;
  qgpu_source_dtype source_dtype;
  const uint8_t *selected_diffraction_uint8;
  const uint16_t *selected_diffraction_uint16;
  size_t selected_diffraction_count;
  const uint64_t *bright_field_uint64;
  const uint64_t *annular_bright_field_uint64;
  const uint64_t *annular_dark_field_uint64;
  const uint64_t *total_intensity_uint64;
  const uint64_t *diffraction_sum_uint64;
  const float *mean_diffraction_float32;
  const float *center_of_mass_row_float32;
  const float *center_of_mass_column_float32;
  const float *dpc_row_float32;
  const float *dpc_column_float32;
  const float *idpc_float32;
  size_t scan_product_count;
  size_t detector_product_count;
  uint64_t global_total_intensity_uint64;
  float dpc_rotation_degrees;
  bool dpc_component_order_exchanged;
  bool parity_checked;
  uint64_t mismatch_count;
  qgpu_vulkan_metrics metrics;
} qgpu_vulkan_result_view;

uint32_t qgpu_vulkan_abi_version(void);

qgpu_status qgpu_vulkan_open_v1(const qgpu_vulkan_open_request *request,
                                qgpu_vulkan_session **output_session,
                                qgpu_error *error);

qgpu_status qgpu_vulkan_request_products_v1(
    qgpu_vulkan_session *session, const qgpu_vulkan_product_request *request,
    qgpu_vulkan_result **output_result, qgpu_error *error);

/**
 * Read one native-uint16 diffraction frame from an already-open indexed QH5
 * session without creating or submitting Vulkan work.
 *
 * This call is synchronous. The destination is caller-owned and must remain
 * valid for the duration of the call; it may be reused immediately after the
 * call returns. On success, exactly destination_value_count values have been
 * written. Validation failures leave the destination untouched. Destination
 * contents are unspecified after a source or decode failure.
 */
qgpu_status qgpu_vulkan_read_selected_diffraction_v1(
    qgpu_vulkan_session *session,
    const qgpu_vulkan_selected_diffraction_request *request,
    qgpu_vulkan_selected_diffraction_result *output_result, qgpu_error *error);

/**
 * Read one native-uint16 diffraction frame using Vulkan LZ4 decode and
 * bit-unshuffle. Storage IO and command submission remain host-orchestrated;
 * detector-value decompression never executes on the CPU.
 */
qgpu_status qgpu_vulkan_read_selected_diffraction_gpu_v1(
    qgpu_vulkan_session *session,
    const qgpu_vulkan_selected_diffraction_request *request,
    qgpu_vulkan_selected_diffraction_result *output_result, qgpu_error *error);

qgpu_status qgpu_vulkan_result_view_v1(const qgpu_vulkan_result *result,
                                       qgpu_vulkan_result_view *output_view,
                                       qgpu_error *error);

qgpu_status qgpu_vulkan_poll_event_v1(qgpu_vulkan_session *session,
                                      qgpu_vulkan_event *output_event,
                                      qgpu_error *error);

void qgpu_vulkan_result_release_v1(qgpu_vulkan_result **result);

qgpu_status qgpu_vulkan_cancel_through_generation_v1(
    qgpu_vulkan_session *session, uint64_t generation, qgpu_error *error);

void qgpu_vulkan_close_v1(qgpu_vulkan_session **session);

#ifdef __cplusplus
}
#endif

#endif
