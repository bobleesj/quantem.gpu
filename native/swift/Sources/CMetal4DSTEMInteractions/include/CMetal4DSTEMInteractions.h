#ifndef C_METAL_4DSTEM_INTERACTIONS_H
#define C_METAL_4DSTEM_INTERACTIONS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint32_t word;
  uint32_t coefficients;
} QDetectorWordEntry;

/// Apply exact packed-uint16 detector deltas to one detector-word-major shard.
void q_update_virtual_detector_u16(
    const uint32_t *data,
    uint32_t *output,
    size_t scan_count,
    const QDetectorWordEntry *entries,
    size_t entry_count);

/// Apply exact packed-uint8 detector deltas to a contiguous scan range.
void q_update_virtual_detector_u8_range(
    const uint32_t *data,
    uint32_t *output,
    size_t data_scan_count,
    size_t scan_offset,
    size_t scan_count,
    const QDetectorWordEntry *entries,
    size_t entry_count);

#endif
