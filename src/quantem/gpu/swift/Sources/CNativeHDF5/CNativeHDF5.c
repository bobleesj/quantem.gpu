#include "CNativeHDF5.h"

#include <hdf5.h>

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QH5_METADATA_LIMIT 100

static pthread_mutex_t qh5_hdf5_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
  char *path;
} qh5_stack_search;

typedef struct {
  qh5_chunk_info *chunks;
  size_t count;
  int invalid;
} qh5_chunk_context;

static int qh5_fail(char **message, const char *format, ...) {
  if (message != NULL) {
    va_list arguments;
    va_start(arguments, format);
    int length = vsnprintf(NULL, 0, format, arguments);
    va_end(arguments);
    if (length >= 0) {
      *message = malloc((size_t)length + 1);
      if (*message != NULL) {
        va_start(arguments, format);
        vsnprintf(*message, (size_t)length + 1, format, arguments);
        va_end(arguments);
      }
    }
  }
  return -1;
}

static char *qh5_copy_string(const char *value) {
  if (value == NULL) return NULL;
  size_t length = strlen(value);
  char *copy = malloc(length + 1);
  if (copy != NULL) memcpy(copy, value, length + 1);
  return copy;
}

static hid_t qh5_open_known_stack(hid_t file) {
  const char *paths[] = {"/entry/data/data", "entry/data/data", "/data", "data"};
  for (size_t index = 0; index < sizeof(paths) / sizeof(paths[0]); index++) {
    hid_t dataset = H5Dopen2(file, paths[index], H5P_DEFAULT);
    if (dataset < 0) continue;
    hid_t space = H5Dget_space(dataset);
    int rank = space >= 0 ? H5Sget_simple_extent_ndims(space) : -1;
    if (space >= 0) H5Sclose(space);
    if (rank == 3) return dataset;
    H5Dclose(dataset);
  }
  return -1;
}

static herr_t qh5_find_stack_callback(
  hid_t object,
  const char *name,
  const H5O_info2_t *object_info,
  void *context_pointer
) {
  if (object_info->type != H5O_TYPE_DATASET) return 0;
  qh5_stack_search *context = context_pointer;
  hid_t dataset = H5Dopen2(object, name, H5P_DEFAULT);
  if (dataset < 0) return 0;
  hid_t space = H5Dget_space(dataset);
  int rank = space >= 0 ? H5Sget_simple_extent_ndims(space) : -1;
  if (space >= 0) H5Sclose(space);
  H5Dclose(dataset);
  if (rank != 3) return 0;
  context->path = qh5_copy_string(name);
  return context->path == NULL ? -1 : 1;
}

static hid_t qh5_open_stack(hid_t file) {
  hid_t dataset = qh5_open_known_stack(file);
  if (dataset >= 0) return dataset;
  qh5_stack_search search = {.path = NULL};
  herr_t result = H5Ovisit3(
    file,
    H5_INDEX_NAME,
    H5_ITER_NATIVE,
    qh5_find_stack_callback,
    &search,
    H5O_INFO_BASIC
  );
  if (result < 0 || search.path == NULL) {
    free(search.path);
    return -1;
  }
  dataset = H5Dopen2(file, search.path, H5P_DEFAULT);
  free(search.path);
  return dataset;
}

static int qh5_read_stack_geometry(
  hid_t dataset,
  qh5_stack_info *stack,
  char **error_message
) {
  hid_t space = H5Dget_space(dataset);
  if (space < 0) return qh5_fail(error_message, "Could not inspect detector-stack dimensions");
  hsize_t dimensions[3] = {0, 0, 0};
  int rank = H5Sget_simple_extent_dims(space, dimensions, NULL);
  H5Sclose(space);
  if (rank != 3 || dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] == 0) {
    return qh5_fail(error_message, "The detector stack must have three non-empty dimensions");
  }

  hid_t type = H5Dget_type(dataset);
  if (type < 0) return qh5_fail(error_message, "Could not inspect detector-stack dtype");
  H5T_class_t type_class = H5Tget_class(type);
  H5T_sign_t sign = H5Tget_sign(type);
  size_t source_bytes = H5Tget_size(type);
  H5Tclose(type);
  if (type_class != H5T_INTEGER || sign != H5T_SGN_NONE || (source_bytes != 1 && source_bytes != 2)) {
    return qh5_fail(
      error_message,
      "QuantEM.GPU native HDF5 supports uint8/uint16 detector counts; this stack uses an unsupported dtype"
    );
  }

  hid_t creation = H5Dget_create_plist(dataset);
  if (creation < 0 || H5Pget_layout(creation) != H5D_CHUNKED) {
    if (creation >= 0) H5Pclose(creation);
    return qh5_fail(error_message, "The detector stack is not chunked HDF5 data");
  }
  hsize_t chunk_dimensions[3] = {0, 0, 0};
  int chunk_rank = H5Pget_chunk(creation, 3, chunk_dimensions);
  H5Pclose(creation);
  if (chunk_rank != 3) {
    return qh5_fail(error_message, "The detector stack has unsupported HDF5 chunk geometry");
  }

  stack->frame_count = dimensions[0];
  stack->detector_rows = dimensions[1];
  stack->detector_columns = dimensions[2];
  stack->source_bytes = (uint32_t)source_bytes;
  stack->chunk_frames = chunk_dimensions[0];
  stack->chunk_rows = chunk_dimensions[1];
  stack->chunk_columns = chunk_dimensions[2];
  return 0;
}

static herr_t qh5_chunk_callback(
  const hsize_t *offset,
  unsigned filter_mask,
  haddr_t address,
  hsize_t size,
  void *context_pointer
) {
  (void)filter_mask;
  qh5_chunk_context *context = context_pointer;
  if (offset[1] != 0 || offset[2] != 0 || offset[0] >= context->count || size == 0
      || address == HADDR_UNDEF || context->chunks[offset[0]].size != 0) {
    context->invalid = 1;
    return -1;
  }
  context->chunks[offset[0]].offset = address;
  context->chunks[offset[0]].size = size;
  return 0;
}

static int qh5_inspect_stack_unlocked(
  const char *path,
  int include_chunks,
  qh5_stack_info *stack,
  qh5_chunk_info **chunks,
  size_t *chunk_count,
  char **error_message
) {
  if (path == NULL || stack == NULL || chunks == NULL || chunk_count == NULL) {
    return qh5_fail(error_message, "Invalid native HDF5 stack request");
  }
  *chunks = NULL;
  *chunk_count = 0;
  if (error_message != NULL) *error_message = NULL;
  memset(stack, 0, sizeof(*stack));
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) return qh5_fail(error_message, "Could not open HDF5 file %s", path);
  hid_t dataset = qh5_open_stack(file);
  if (dataset < 0) {
    H5Fclose(file);
    return qh5_fail(error_message, "No 3-D detector stack was found in %s", path);
  }
  int status = qh5_read_stack_geometry(dataset, stack, error_message);
  if (status == 0 && include_chunks) {
    if (stack->chunk_frames != 1 || stack->chunk_rows != stack->detector_rows
        || stack->chunk_columns != stack->detector_columns) {
      status = qh5_fail(
        error_message,
        "%s is not one full detector frame per HDF5 chunk",
        path
      );
    } else if (stack->frame_count > SIZE_MAX / sizeof(qh5_chunk_info)) {
      status = qh5_fail(error_message, "%s contains too many detector frames", path);
    } else {
      qh5_chunk_info *table = calloc((size_t)stack->frame_count, sizeof(*table));
      if (table == NULL) {
        status = qh5_fail(error_message, "Could not allocate the HDF5 chunk index");
      } else {
        qh5_chunk_context context = {
          .chunks = table,
          .count = (size_t)stack->frame_count,
          .invalid = 0,
        };
        herr_t iteration = H5Dchunk_iter(dataset, H5P_DEFAULT, qh5_chunk_callback, &context);
        if (iteration < 0 || context.invalid) {
          free(table);
          status = qh5_fail(error_message, "%s has invalid detector chunk coordinates", path);
        } else {
          for (size_t index = 0; index < context.count; index++) {
            if (table[index].size == 0) {
              free(table);
              status = qh5_fail(error_message, "%s is missing detector frame %zu", path, index);
              break;
            }
          }
          if (status == 0) {
            *chunks = table;
            *chunk_count = context.count;
          }
        }
      }
    }
  }
  H5Dclose(dataset);
  H5Fclose(file);
  return status;
}

int qh5_inspect_stack(
  const char *path,
  int include_chunks,
  qh5_stack_info *stack,
  qh5_chunk_info **chunks,
  size_t *chunk_count,
  char **error_message
) {
  pthread_mutex_lock(&qh5_hdf5_lock);
  int status = qh5_inspect_stack_unlocked(
    path,
    include_chunks,
    stack,
    chunks,
    chunk_count,
    error_message
  );
  pthread_mutex_unlock(&qh5_hdf5_lock);
  return status;
}

static int qh5_append_string(char ***values, size_t *count, const char *value) {
  if (*count >= SIZE_MAX / sizeof(**values)) return -1;
  char *copy = qh5_copy_string(value);
  if (copy == NULL) return -1;
  char **grown = realloc(*values, (*count + 1) * sizeof(**values));
  if (grown == NULL) {
    free(copy);
    return -1;
  }
  grown[*count] = copy;
  *values = grown;
  (*count)++;
  return 0;
}

static herr_t qh5_external_link_callback(
  hid_t group,
  const char *name,
  const H5L_info2_t *link_info,
  void *context_pointer
) {
  qh5_master_info *info = context_pointer;
  if (link_info->type != H5L_TYPE_EXTERNAL || link_info->u.val_size == 0) return 0;
  void *value = malloc(link_info->u.val_size);
  if (value == NULL) return -1;
  if (H5Lget_val(group, name, value, link_info->u.val_size, H5P_DEFAULT) < 0) {
    free(value);
    return -1;
  }
  unsigned flags = 0;
  const char *filename = NULL;
  const char *object_name = NULL;
  herr_t result = H5Lunpack_elink_val(
    value,
    link_info->u.val_size,
    &flags,
    &filename,
    &object_name
  );
  (void)flags;
  (void)object_name;
  if (result >= 0 && filename != NULL) {
    result = qh5_append_string(&info->external_files, &info->external_file_count, filename);
  }
  free(value);
  return result;
}

static int qh5_read_external_files(hid_t file, qh5_master_info *info) {
  hid_t group = H5Gopen2(file, "/entry/data", H5P_DEFAULT);
  if (group < 0) return 0;
  hsize_t index = 0;
  herr_t result = H5Literate2(
    group,
    H5_INDEX_NAME,
    H5_ITER_INC,
    &index,
    qh5_external_link_callback,
    info
  );
  H5Gclose(group);
  return result < 0 ? -1 : 0;
}

static int qh5_read_integer_dataset(hid_t file, const char *path, uint64_t *value) {
  hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  if (dataset < 0) return 0;
  hid_t space = H5Dget_space(dataset);
  hssize_t points = space >= 0 ? H5Sget_simple_extent_npoints(space) : -1;
  if (space >= 0) H5Sclose(space);
  unsigned long long raw = 0;
  int found = points == 1 && H5Dread(dataset, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &raw) >= 0;
  H5Dclose(dataset);
  if (found) *value = (uint64_t)raw;
  return found;
}

static int qh5_read_scan_shape_at(hid_t file, const char *path, qh5_master_info *info) {
  hid_t object = H5Oopen(file, path, H5P_DEFAULT);
  if (object < 0) return 0;
  hid_t attribute = H5Aopen(object, "scan_shape", H5P_DEFAULT);
  H5Oclose(object);
  if (attribute < 0) return 0;
  hid_t space = H5Aget_space(attribute);
  hssize_t points = space >= 0 ? H5Sget_simple_extent_npoints(space) : -1;
  if (space >= 0) H5Sclose(space);
  long long values[2] = {0, 0};
  int found = points == 2
    && H5Aread(attribute, H5T_NATIVE_LLONG, values) >= 0
    && values[0] > 0 && values[1] > 0;
  H5Aclose(attribute);
  if (found) {
    info->scan_rows = (uint64_t)values[0];
    info->scan_columns = (uint64_t)values[1];
    info->has_scan_shape = 1;
  }
  return found;
}

static int qh5_read_bad_pixels(
  hid_t file,
  uint64_t detector_rows,
  uint64_t detector_columns,
  qh5_master_info *info
) {
  hid_t dataset = H5Dopen2(
    file,
    "/entry/instrument/detector/detectorSpecific/pixel_mask",
    H5P_DEFAULT
  );
  if (dataset < 0) return 0;
  hid_t space = H5Dget_space(dataset);
  hsize_t dimensions[2] = {0, 0};
  int rank = space >= 0 ? H5Sget_simple_extent_dims(space, dimensions, NULL) : -1;
  if (space >= 0) H5Sclose(space);
  if (rank != 2 || dimensions[0] != detector_rows || dimensions[1] != detector_columns) {
    H5Dclose(dataset);
    return -1;
  }
  if (detector_rows != 0 && detector_columns > SIZE_MAX / detector_rows) {
    H5Dclose(dataset);
    return -1;
  }
  size_t count = (size_t)(detector_rows * detector_columns);
  if (count > SIZE_MAX / sizeof(unsigned long long)) {
    H5Dclose(dataset);
    return -1;
  }
  unsigned long long *values = malloc(count * sizeof(*values));
  if (values == NULL || H5Dread(dataset, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) < 0) {
    free(values);
    H5Dclose(dataset);
    return -1;
  }
  H5Dclose(dataset);
  size_t bad_count = 0;
  for (size_t index = 0; index < count; index++) bad_count += values[index] != 0;
  if (bad_count != 0) {
    info->bad_pixel_indices = malloc(bad_count * sizeof(*info->bad_pixel_indices));
    if (info->bad_pixel_indices == NULL) {
      free(values);
      return -1;
    }
    size_t output = 0;
    for (size_t index = 0; index < count; index++) {
      if (values[index] != 0) info->bad_pixel_indices[output++] = index;
    }
  }
  info->bad_pixel_count = bad_count;
  free(values);
  return 1;
}

static char *qh5_read_attribute_string(hid_t object, const char *name);

static char *qh5_read_string_value(hid_t container, int is_attribute) {
  hid_t type = is_attribute ? H5Aget_type(container) : H5Dget_type(container);
  hid_t space = is_attribute ? H5Aget_space(container) : H5Dget_space(container);
  if (type < 0 || space < 0 || H5Tget_class(type) != H5T_STRING
      || H5Sget_simple_extent_npoints(space) != 1) {
    if (type >= 0) H5Tclose(type);
    if (space >= 0) H5Sclose(space);
    return NULL;
  }
  char *result = NULL;
  if (H5Tis_variable_str(type)) {
    char *value = NULL;
    herr_t status = is_attribute
      ? H5Aread(container, type, &value)
      : H5Dread(container, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    if (status >= 0 && value != NULL) result = qh5_copy_string(value);
    if (value != NULL) H5free_memory(value);
  } else {
    size_t length = H5Tget_size(type);
    char *value = calloc(length + 1, 1);
    if (value != NULL) {
      herr_t status = is_attribute
        ? H5Aread(container, type, value)
        : H5Dread(container, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value);
      if (status >= 0) {
        value[length] = '\0';
        while (length > 0 && (value[length - 1] == '\0' || value[length - 1] == ' ')) {
          value[--length] = '\0';
        }
        result = qh5_copy_string(value);
      }
      free(value);
    }
  }
  H5Tclose(type);
  H5Sclose(space);
  return result;
}

static char *qh5_read_attribute_string(hid_t object, const char *name) {
  hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0) return NULL;
  char *value = qh5_read_string_value(attribute, 1);
  H5Aclose(attribute);
  return value;
}

static char *qh5_read_dataset_string(hid_t file, const char *path) {
  hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  if (dataset < 0) return NULL;
  char *value = qh5_read_string_value(dataset, 0);
  H5Dclose(dataset);
  return value;
}

static int qh5_read_length_meters_with_policy(
    hid_t file,
    const char *path,
    double *value_meters,
    int require_explicit_units) {
  hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  if (dataset < 0) return 0;
  hid_t space = H5Dget_space(dataset);
  hssize_t points = space >= 0 ? H5Sget_simple_extent_npoints(space) : -1;
  if (space >= 0) H5Sclose(space);
  double value = 0;
  int found = points == 1
    && H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) >= 0
    && isfinite(value) && value > 0;
  char *units = found ? qh5_read_attribute_string(dataset, "units") : NULL;
  H5Dclose(dataset);
  if (!found) {
    free(units);
    return 0;
  }
  if (require_explicit_units && (units == NULL || units[0] == '\0')) {
    free(units);
    return 0;
  }
  double factor = 0;
  const char *unit = units == NULL || units[0] == '\0' ? "m" : units;
  if (strcasecmp(unit, "m") == 0) factor = 1;
  else if (strcasecmp(unit, "mm") == 0) factor = 1e-3;
  else if (strcasecmp(unit, "um") == 0 || strcmp(unit, "µm") == 0) factor = 1e-6;
  else if (strcasecmp(unit, "nm") == 0) factor = 1e-9;
  free(units);
  if (factor == 0) return 0;
  *value_meters = value * factor;
  return 1;
}

static int qh5_read_length_meters(hid_t file, const char *path, double *value_meters) {
  return qh5_read_length_meters_with_policy(file, path, value_meters, 0);
}

static void qh5_read_reciprocal_sampling(hid_t file, qh5_master_info *info) {
  double distance = 0;
  double row_pitch = 0;
  double column_pitch = 0;
  if (qh5_read_length_meters(file, "/entry/instrument/detector/detector_distance", &distance)
      && qh5_read_length_meters(file, "/entry/instrument/detector/y_pixel_size", &row_pitch)
      && qh5_read_length_meters(file, "/entry/instrument/detector/x_pixel_size", &column_pitch)) {
    info->reciprocal_row_mrad = atan(row_pitch / distance) * 1000;
    info->reciprocal_column_mrad = atan(column_pitch / distance) * 1000;
    info->has_reciprocal_sampling = 1;
  }
}

static void qh5_read_scan_pixel_size(hid_t file, qh5_master_info *info) {
  const char *row_paths[] = {
    "/entry/instrument/scan/y_pixel_size",
    "/entry/instrument/scan/step_y",
    "/entry/measurement/scan_step_y",
    "/entry/scan/y_pixel_size",
  };
  const char *column_paths[] = {
    "/entry/instrument/scan/x_pixel_size",
    "/entry/instrument/scan/step_x",
    "/entry/measurement/scan_step_x",
    "/entry/scan/x_pixel_size",
  };
  double row_meters = 0;
  double column_meters = 0;
  int found_row = 0;
  int found_column = 0;
  for (size_t index = 0; index < sizeof(row_paths) / sizeof(row_paths[0]); index++) {
    if (qh5_read_length_meters_with_policy(file, row_paths[index], &row_meters, 1)) {
      found_row = 1;
      break;
    }
  }
  for (size_t index = 0; index < sizeof(column_paths) / sizeof(column_paths[0]); index++) {
    if (qh5_read_length_meters_with_policy(file, column_paths[index], &column_meters, 1)) {
      found_column = 1;
      break;
    }
  }
  if (found_row && found_column) {
    info->scan_pixel_row_nm = row_meters * 1e9;
    info->scan_pixel_column_nm = column_meters * 1e9;
    info->has_scan_pixel_size = 1;
  }
}

static char *qh5_format_numeric(hid_t container, int is_attribute) {
  hid_t type = is_attribute ? H5Aget_type(container) : H5Dget_type(container);
  hid_t space = is_attribute ? H5Aget_space(container) : H5Dget_space(container);
  if (type < 0 || space < 0) {
    if (type >= 0) H5Tclose(type);
    if (space >= 0) H5Sclose(space);
    return NULL;
  }
  hssize_t points_value = H5Sget_simple_extent_npoints(space);
  H5T_class_t type_class = H5Tget_class(type);
  if (points_value < 1 || points_value > 16
      || (type_class != H5T_INTEGER && type_class != H5T_FLOAT)) {
    H5Tclose(type);
    H5Sclose(space);
    return NULL;
  }
  size_t points = (size_t)points_value;
  char *result = calloc(points * 32 + 1, 1);
  if (result == NULL) {
    H5Tclose(type);
    H5Sclose(space);
    return NULL;
  }
  size_t used = 0;
  if (type_class == H5T_FLOAT) {
    double values[16] = {0};
    herr_t status = is_attribute
      ? H5Aread(container, H5T_NATIVE_DOUBLE, values)
      : H5Dread(container, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
    if (status < 0) used = SIZE_MAX;
    for (size_t index = 0; used != SIZE_MAX && index < points; index++) {
      int count = snprintf(result + used, points * 32 + 1 - used, "%s%.8g", index ? ", " : "", values[index]);
      if (count < 0) used = SIZE_MAX;
      else used += (size_t)count;
    }
  } else if (H5Tget_sign(type) == H5T_SGN_NONE) {
    unsigned long long values[16] = {0};
    herr_t status = is_attribute
      ? H5Aread(container, H5T_NATIVE_ULLONG, values)
      : H5Dread(container, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
    if (status < 0) used = SIZE_MAX;
    for (size_t index = 0; used != SIZE_MAX && index < points; index++) {
      int count = snprintf(result + used, points * 32 + 1 - used, "%s%llu", index ? ", " : "", values[index]);
      if (count < 0) used = SIZE_MAX;
      else used += (size_t)count;
    }
  } else {
    long long values[16] = {0};
    herr_t status = is_attribute
      ? H5Aread(container, H5T_NATIVE_LLONG, values)
      : H5Dread(container, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
    if (status < 0) used = SIZE_MAX;
    for (size_t index = 0; used != SIZE_MAX && index < points; index++) {
      int count = snprintf(result + used, points * 32 + 1 - used, "%s%lld", index ? ", " : "", values[index]);
      if (count < 0) used = SIZE_MAX;
      else used += (size_t)count;
    }
  }
  H5Tclose(type);
  H5Sclose(space);
  if (used == SIZE_MAX) {
    free(result);
    return NULL;
  }
  return result;
}

static char *qh5_format_value(hid_t container, int is_attribute) {
  char *string_value = qh5_read_string_value(container, is_attribute);
  return string_value != NULL ? string_value : qh5_format_numeric(container, is_attribute);
}

static int qh5_append_metadata(qh5_master_info *info, const char *key, char *value) {
  if (value == NULL || value[0] == '\0') {
    free(value);
    return 0;
  }
  if (info->metadata_count >= QH5_METADATA_LIMIT) {
    free(value);
    return 0;
  }
  qh5_metadata_item *grown = realloc(
    info->metadata,
    (info->metadata_count + 1) * sizeof(*info->metadata)
  );
  if (grown == NULL) {
    free(value);
    return -1;
  }
  info->metadata = grown;
  qh5_metadata_item *item = &info->metadata[info->metadata_count];
  item->key = qh5_copy_string(key);
  item->value = value;
  if (item->key == NULL) {
    free(value);
    return -1;
  }
  info->metadata_count++;
  return 0;
}

typedef struct {
  qh5_master_info *info;
  const char *object_name;
  int failed;
} qh5_attribute_context;

static herr_t qh5_attribute_callback(
  hid_t object,
  const char *attribute_name,
  const H5A_info_t *attribute_info,
  void *context_pointer
) {
  (void)attribute_info;
  qh5_attribute_context *context = context_pointer;
  if (strcmp(attribute_name, "units") == 0
      || context->info->metadata_count >= QH5_METADATA_LIMIT) return 0;
  hid_t attribute = H5Aopen(object, attribute_name, H5P_DEFAULT);
  if (attribute < 0) return 0;
  char *value = qh5_format_value(attribute, 1);
  H5Aclose(attribute);
  size_t length = strlen(context->object_name) + strlen(attribute_name) + 2;
  char *key = malloc(length);
  if (key == NULL) {
    free(value);
    context->failed = 1;
    return -1;
  }
  snprintf(key, length, "%s@%s", context->object_name, attribute_name);
  int result = qh5_append_metadata(context->info, key, value);
  free(key);
  if (result < 0) context->failed = 1;
  return result;
}

typedef struct {
  qh5_master_info *info;
  int failed;
} qh5_metadata_context;

static herr_t qh5_metadata_callback(
  hid_t root,
  const char *name,
  const H5O_info2_t *object_info,
  void *context_pointer
) {
  qh5_metadata_context *context = context_pointer;
  if (context->info->metadata_count >= QH5_METADATA_LIMIT) return 1;
  if (strcmp(name, ".") == 0) return 0;
  if (strcmp(name, "entry/data") == 0 || strncmp(name, "entry/data/", 11) == 0) return 0;
  hid_t object = H5Oopen(root, name, H5P_DEFAULT);
  if (object < 0) return 0;
  size_t name_length = strlen(name);
  int is_pixel_mask = name_length >= strlen("pixel_mask")
    && strcmp(name + name_length - strlen("pixel_mask"), "pixel_mask") == 0;
  if (object_info->type == H5O_TYPE_DATASET && !is_pixel_mask) {
    char *value = qh5_format_value(object, 0);
    char *units = qh5_read_attribute_string(object, "units");
    if (value != NULL && units != NULL && units[0] != '\0') {
      size_t length = strlen(value) + strlen(units) + 2;
      char *combined = malloc(length);
      if (combined == NULL) {
        free(value);
        value = NULL;
        context->failed = 1;
      } else {
        snprintf(combined, length, "%s %s", value, units);
        free(value);
        value = combined;
      }
    }
    free(units);
    if (qh5_append_metadata(context->info, name, value) < 0) context->failed = 1;
  }
  if (!context->failed && context->info->metadata_count < QH5_METADATA_LIMIT) {
    qh5_attribute_context attributes = {
      .info = context->info,
      .object_name = name,
      .failed = 0,
    };
    hsize_t index = 0;
    H5Aiterate2(
      object,
      H5_INDEX_NAME,
      H5_ITER_INC,
      &index,
      qh5_attribute_callback,
      &attributes
    );
    context->failed = attributes.failed;
  }
  H5Oclose(object);
  return context->failed ? -1 : 0;
}

static int qh5_read_display_metadata(hid_t file, qh5_master_info *info) {
  qh5_metadata_context context = {.info = info, .failed = 0};
  herr_t result = H5Ovisit3(
    file,
    H5_INDEX_NAME,
    H5_ITER_NATIVE,
    qh5_metadata_callback,
    &context,
    H5O_INFO_BASIC
  );
  return result < 0 || context.failed ? -1 : 0;
}

static int qh5_inspect_master_unlocked(
  const char *path,
  uint64_t detector_rows,
  uint64_t detector_columns,
  qh5_master_info *info,
  char **error_message
) {
  if (path == NULL || info == NULL) return qh5_fail(error_message, "Invalid native HDF5 metadata request");
  memset(info, 0, sizeof(*info));
  if (error_message != NULL) *error_message = NULL;
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) return qh5_fail(error_message, "Could not open HDF5 metadata file %s", path);

  int status = 0;
  if (qh5_read_external_files(file, info) < 0) {
    status = qh5_fail(error_message, "Could not inspect external HDF5 data links in %s", path);
  }
  if (status == 0 && !qh5_read_scan_shape_at(file, "/entry/data/data", info)) {
    qh5_read_scan_shape_at(file, "/entry/data", info);
  }
  uint64_t expected = 0;
  if (status == 0
      && (qh5_read_integer_dataset(file, "/entry/instrument/detector/detectorSpecific/ntrigger", &expected)
          || qh5_read_integer_dataset(file, "/entry/instrument/detector/detectorSpecific/nimages", &expected))) {
    info->expected_frames = expected;
    info->has_expected_frames = 1;
  }
  if (status == 0 && detector_rows > 0 && detector_columns > 0) {
    int mask = qh5_read_bad_pixels(file, detector_rows, detector_columns, info);
    if (mask < 0) status = qh5_fail(error_message, "The HDF5 pixel mask does not match the detector shape");
  }
  if (status == 0) qh5_read_reciprocal_sampling(file, info);
  if (status == 0) qh5_read_scan_pixel_size(file, info);
  const char *date_paths[] = {
    "/entry/instrument/detector/detectorSpecific/data_collection_date",
    "/entry/start_time",
    "/entry/instrument/start_time",
  };
  if (status == 0) {
    for (size_t index = 0; index < sizeof(date_paths) / sizeof(date_paths[0]); index++) {
      info->acquisition_date = qh5_read_dataset_string(file, date_paths[index]);
      if (info->acquisition_date != NULL && info->acquisition_date[0] != '\0') break;
      free(info->acquisition_date);
      info->acquisition_date = NULL;
    }
  }
  if (status == 0 && qh5_read_display_metadata(file, info) < 0) {
    status = qh5_fail(error_message, "Could not collect HDF5 display metadata from %s", path);
  }
  H5Fclose(file);
  if (status != 0) qh5_free_master_info(info);
  return status;
}

int qh5_inspect_master(
  const char *path,
  uint64_t detector_rows,
  uint64_t detector_columns,
  qh5_master_info *info,
  char **error_message
) {
  pthread_mutex_lock(&qh5_hdf5_lock);
  int status = qh5_inspect_master_unlocked(
    path,
    detector_rows,
    detector_columns,
    info,
    error_message
  );
  pthread_mutex_unlock(&qh5_hdf5_lock);
  return status;
}

typedef struct {
  char *group_name;
  int allocation_failed;
} qh5_velox_search;

static herr_t qh5_velox_image_callback(
  hid_t image_root,
  const char *name,
  const H5L_info2_t *link_info,
  void *context_pointer
) {
  (void)link_info;
  qh5_velox_search *search = context_pointer;
  hid_t group = H5Gopen2(image_root, name, H5P_DEFAULT);
  if (group < 0) return 0;
  hid_t data = H5Dopen2(group, "Data", H5P_DEFAULT);
  hid_t metadata = H5Dopen2(group, "Metadata", H5P_DEFAULT);
  hid_t space = data >= 0 ? H5Dget_space(data) : -1;
  hsize_t dimensions[3] = {0, 0, 0};
  int rank = space >= 0 ? H5Sget_simple_extent_dims(space, dimensions, NULL) : -1;
  if (space >= 0) H5Sclose(space);
  if (data >= 0) H5Dclose(data);
  if (metadata >= 0) H5Dclose(metadata);
  H5Gclose(group);
  if (rank != 3 || dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] != 1
      || metadata < 0) return 0;
  search->group_name = qh5_copy_string(name);
  if (search->group_name == NULL) {
    search->allocation_failed = 1;
    return -1;
  }
  return 1;
}

static int qh5_prepare_velox_image_unlocked(
  const char *source_path,
  const char *raw_output_path,
  qh5_velox_image_info *info,
  char **error_message
) {
  if (source_path == NULL || info == NULL) {
    return qh5_fail(error_message, "Invalid native Velox EMD request");
  }
  memset(info, 0, sizeof(*info));
  if (error_message != NULL) *error_message = NULL;
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  hid_t file = H5Fopen(source_path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) return qh5_fail(error_message, "Could not open Velox EMD file %s", source_path);
  hid_t image_root = H5Gopen2(file, "/Data/Image", H5P_DEFAULT);
  if (image_root < 0) {
    H5Fclose(file);
    return qh5_fail(error_message, "The EMD file has no Velox Data/Image group");
  }

  qh5_velox_search search = {.group_name = NULL, .allocation_failed = 0};
  hsize_t index = 0;
  herr_t iteration = H5Literate2(
    image_root,
    H5_INDEX_NAME,
    H5_ITER_INC,
    &index,
    qh5_velox_image_callback,
    &search
  );
  if (iteration < 0 || search.allocation_failed) {
    H5Gclose(image_root);
    H5Fclose(file);
    free(search.group_name);
    return qh5_fail(error_message, "Could not inspect Velox Data/Image entries");
  }
  if (search.group_name == NULL) {
    H5Gclose(image_root);
    H5Fclose(file);
    return qh5_fail(
      error_message,
      "The EMD file has no supported 2-D Velox image and Metadata JSON pair"
    );
  }

  hid_t group = H5Gopen2(image_root, search.group_name, H5P_DEFAULT);
  hid_t data = group >= 0 ? H5Dopen2(group, "Data", H5P_DEFAULT) : -1;
  hid_t metadata = group >= 0 ? H5Dopen2(group, "Metadata", H5P_DEFAULT) : -1;
  int status = 0;
  hsize_t dimensions[3] = {0, 0, 0};
  hid_t data_space = data >= 0 ? H5Dget_space(data) : -1;
  int rank = data_space >= 0
    ? H5Sget_simple_extent_dims(data_space, dimensions, NULL) : -1;
  if (data_space >= 0) H5Sclose(data_space);
  if (rank != 3 || dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] != 1) {
    status = qh5_fail(error_message, "Velox scalar image must have shape (row, column, 1)");
  }

  hid_t data_type = status == 0 ? H5Dget_type(data) : -1;
  size_t source_bytes = data_type >= 0 ? H5Tget_size(data_type) : 0;
  if (status == 0
      && (data_type < 0 || H5Tget_class(data_type) != H5T_INTEGER
          || H5Tget_sign(data_type) != H5T_SGN_NONE
          || (source_bytes != 1 && source_bytes != 2))) {
    status = qh5_fail(
      error_message,
      "QuantEM.GPU native EMD supports uint8/uint16 scalar images; this image uses an unsupported dtype"
    );
  }
  if (data_type >= 0) H5Tclose(data_type);

  hid_t metadata_space = status == 0 ? H5Dget_space(metadata) : -1;
  hssize_t metadata_points = metadata_space >= 0
    ? H5Sget_simple_extent_npoints(metadata_space) : -1;
  if (metadata_space >= 0) H5Sclose(metadata_space);
  unsigned char *metadata_bytes = NULL;
  if (status == 0 && (metadata_points <= 0 || (uint64_t)metadata_points >= SIZE_MAX)) {
    status = qh5_fail(error_message, "Velox Metadata JSON is empty or too large");
  }
  if (status == 0) {
    metadata_bytes = calloc((size_t)metadata_points + 1, 1);
    if (metadata_bytes == NULL
        || H5Dread(
          metadata,
          H5T_NATIVE_UCHAR,
          H5S_ALL,
          H5S_ALL,
          H5P_DEFAULT,
          metadata_bytes
        ) < 0) {
      status = qh5_fail(error_message, "Could not read Velox Metadata JSON");
    }
  }

  if (status == 0 && raw_output_path != NULL) {
    if (dimensions[0] > SIZE_MAX / dimensions[1]
        || dimensions[0] * dimensions[1] > SIZE_MAX / source_bytes) {
      status = qh5_fail(error_message, "The Velox scalar image is too large");
    } else {
      size_t byte_count = (size_t)(dimensions[0] * dimensions[1] * source_bytes);
      void *values = malloc(byte_count);
      hid_t memory_type = source_bytes == 1 ? H5T_STD_U8LE : H5T_STD_U16LE;
      if (values == NULL
          || H5Dread(data, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) < 0) {
        free(values);
        status = qh5_fail(error_message, "Could not read the Velox scalar image");
      } else {
        FILE *output = fopen(raw_output_path, "wb");
        int write_failed = output == NULL;
        if (!write_failed && fwrite(values, 1, byte_count, output) != byte_count) {
          write_failed = 1;
        }
        if (!write_failed && fflush(output) != 0) write_failed = 1;
        if (output != NULL && fclose(output) != 0) write_failed = 1;
        if (write_failed) {
          remove(raw_output_path);
          status = qh5_fail(error_message, "Could not write the native Velox image cache");
        }
        free(values);
      }
    }
  }

  if (status == 0) {
    size_t path_length = strlen("Data/Image//Metadata") + strlen(search.group_name) + 1;
    info->metadata_path = malloc(path_length);
    if (info->metadata_path == NULL) {
      status = qh5_fail(error_message, "Could not allocate Velox metadata provenance");
    } else {
      snprintf(
        info->metadata_path,
        path_length,
        "Data/Image/%s/Metadata",
        search.group_name
      );
      info->rows = dimensions[0];
      info->columns = dimensions[1];
      info->source_bytes = (uint32_t)source_bytes;
      info->metadata_json = (char *)metadata_bytes;
      metadata_bytes = NULL;
    }
  }

  free(metadata_bytes);
  if (metadata >= 0) H5Dclose(metadata);
  if (data >= 0) H5Dclose(data);
  if (group >= 0) H5Gclose(group);
  H5Gclose(image_root);
  H5Fclose(file);
  free(search.group_name);
  if (status != 0) qh5_free_velox_image_info(info);
  return status;
}

int qh5_prepare_velox_image(
  const char *source_path,
  const char *raw_output_path,
  qh5_velox_image_info *info,
  char **error_message
) {
  pthread_mutex_lock(&qh5_hdf5_lock);
  int status = qh5_prepare_velox_image_unlocked(
    source_path,
    raw_output_path,
    info,
    error_message
  );
  pthread_mutex_unlock(&qh5_hdf5_lock);
  return status;
}

void qh5_free_chunks(qh5_chunk_info *chunks) {
  free(chunks);
}

void qh5_free_master_info(qh5_master_info *info) {
  if (info == NULL) return;
  free(info->bad_pixel_indices);
  free(info->acquisition_date);
  for (size_t index = 0; index < info->metadata_count; index++) {
    free(info->metadata[index].key);
    free(info->metadata[index].value);
  }
  free(info->metadata);
  for (size_t index = 0; index < info->external_file_count; index++) free(info->external_files[index]);
  free(info->external_files);
  memset(info, 0, sizeof(*info));
}

void qh5_free_velox_image_info(qh5_velox_image_info *info) {
  if (info == NULL) return;
  free(info->metadata_json);
  free(info->metadata_path);
  memset(info, 0, sizeof(*info));
}

void qh5_free_error(char *error_message) {
  free(error_message);
}
