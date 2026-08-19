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

void qh5_free_chunks(qh5_chunk_info *chunks);
void qh5_free_master_info(qh5_master_info *info);
void qh5_free_velox_image_info(qh5_velox_image_info *info);
void qh5_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
