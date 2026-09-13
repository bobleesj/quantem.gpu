#ifndef C_NATIVE_HDF5_H
#define C_NATIVE_HDF5_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t frame_count;
  uint64_t detector_rows;
  uint64_t detector_columns;
  uint32_t source_bytes;
  uint64_t chunk_frames;
  uint64_t chunk_rows;
  uint64_t chunk_columns;
} qh5_stack_info;

typedef struct {
  uint64_t offset;
  uint64_t size;
} qh5_chunk_info;

typedef struct {
  char *key;
  char *value;
} qh5_metadata_item;

typedef struct {
  uint64_t expected_frames;
  int has_expected_frames;
  uint64_t scan_rows;
  uint64_t scan_columns;
  int has_scan_shape;
  uint64_t *bad_pixel_indices;
  size_t bad_pixel_count;
  uint32_t *detector_mask_values;
  size_t detector_mask_count;
  double scan_pixel_row_nm;
  double scan_pixel_column_nm;
  int has_scan_pixel_size;
  double reciprocal_row_mrad;
  double reciprocal_column_mrad;
  int has_reciprocal_sampling;
  char *acquisition_date;
  qh5_metadata_item *metadata;
  size_t metadata_count;
  char **external_files;
  size_t external_file_count;
} qh5_master_info;

typedef struct {
  uint64_t rows;
  uint64_t columns;
  uint32_t source_bytes;
  char *metadata_json;
  char *metadata_path;
} qh5_velox_image_info;

typedef struct qh5_lossless_pack_v1_writer qh5_lossless_pack_v1_writer;

/* Standard uint16 HDF5 storage for chunks already compressed on an accelerator. */
typedef struct qh5_chunk_writer qh5_chunk_writer;
int qh5_chunk_writer_open_typed(const char *path, const uint64_t shape[4],
  uint32_t item_bytes, qh5_chunk_writer **output, char **error_message);
int qh5_chunk_writer_open(const char *path, const uint64_t shape[4],
  qh5_chunk_writer **writer, char **error_message);
int qh5_chunk_writer_append(qh5_chunk_writer *writer, uint64_t first_frame,
  uint64_t frame_count, const uint8_t *chunks, uint64_t stride,
  const uint32_t *sizes, char **error_message);
int qh5_chunk_writer_attribute(qh5_chunk_writer *writer, const char *name,
  const char *value, char **error_message);
int qh5_chunk_writer_close(qh5_chunk_writer *writer, char **error_message);
void qh5_chunk_writer_abort(qh5_chunk_writer *writer);
char *qh5_read_root_attribute(const char *path, const char *name);

/* EMD 1.0 float32 datacube: physical contiguous storage, in recorded axis order. */
typedef struct {
  uint64_t rows, columns, offset, bytes;
  double scan_angstrom, angle_mrad, voltage, semiangle_mrad, camera_meters;
} qh5_emd_float_info;
int qh5_inspect_emd_float(const char *path, qh5_emd_float_info *info, char **error_message);

int qh5_export_scientific_image(const char *path, const char *name,
  const void *values, uint64_t rows, uint64_t columns, uint32_t scalar_type,
  const char *metadata_json, int create, char **error_message);
char *qh5_read_scientific_metadata(const char *path);

typedef struct {
  uint64_t payload_offset;
  uint64_t payload_bytes;
  uint64_t headers_offset;
  uint64_t headers_bytes;
} qh5_lossless_pack_v1_shard_layout;

int qh5_inspect_stack(
  const char *path,
  int include_chunks,
  qh5_stack_info *stack,
  qh5_chunk_info **chunks,
  size_t *chunk_count,
  char **error_message
);

int qh5_inspect_master(
  const char *path,
  uint64_t detector_rows,
  uint64_t detector_columns,
  qh5_master_info *info,
  char **error_message
);

int qh5_prepare_velox_image(
  const char *source_path,
  const char *raw_output_path,
  qh5_velox_image_info *info,
  char **error_message
);

/* Create a new exclusive temporary HDF5 path and return its owned writer. */
int qh5_lossless_pack_v1_writer_open(
  const char *path,
  uint64_t user_block_bytes,
  qh5_lossless_pack_v1_writer **writer,
  char **error_message
);

/* Append one non-empty payload/header pair in consecutive ordinal order. */
int qh5_lossless_pack_v1_writer_append_shard(
  qh5_lossless_pack_v1_writer *writer,
  uint32_t ordinal,
  const uint32_t *payload,
  uint64_t payload_words,
  const uint32_t *headers,
  uint64_t header_words,
  qh5_lossless_pack_v1_shard_layout *layout,
  char **error_message
);

/* Flush and close a complete temporary file. The writer is consumed. */
int qh5_lossless_pack_v1_writer_close(
  qh5_lossless_pack_v1_writer *writer,
  char **error_message
);

/* Close an open writer and remove only the path created by writer_open. */
void qh5_lossless_pack_v1_writer_abort(qh5_lossless_pack_v1_writer *writer);

void qh5_free_chunks(qh5_chunk_info *chunks);
void qh5_free_master_info(qh5_master_info *info);
void qh5_free_velox_image_info(qh5_velox_image_info *info);
void qh5_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
