"""Adjudicate every cycle against the frozen full-image baseline, then time it.

Usage: python compare.py REFERENCE_JSONL CONTROL_A CANDIDATE CONTROL_B
The reference was independently checked against all original HDF5 counts.
GPU timings do not imply drawable presentation or cold-source IO.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def fingerprint(row):
    return row['source_identity'], row['case'], row['step'], row['trial']


def summarize(reference, rows):
    expected = {fingerprint(r): r['sha256_u32_le'] for r in reference
                if r.get('phase') == 'detector'}
    expected_bytes = {r['source_identity']: r['resident_bytes'] for r in reference
                      if r.get('phase') == 'resident'}
    expected_frames = {r['source_identity']: r['sample_hashes'] for r in reference
                       if r.get('phase') == 'resident'}
    assert rows[-1]['phase'] == 'complete', 'Incomplete benchmark'
    cycles = sorted({r['cycle'] for r in rows if r.get('phase') == 'resident'})
    assert cycles == [0, 1, 2], 'Missing load cycle'
    result = {}
    for cycle in cycles:
        maps = [r for r in rows if r.get('phase') == 'detector' and r['cycle'] == cycle]
        actual = {fingerprint(r): r['sha256_u32_le'] for r in maps}
        assert len(maps) == len(expected), 'Duplicate or missing detector record'
        assert actual == expected, 'Exact full-image hash mismatch'
        loads = [r for r in rows if r.get('phase') == 'resident' and r['cycle'] == cycle]
        assert {r['source_identity']: r['resident_bytes'] for r in loads} == expected_bytes
        assert {r['source_identity']: r['sample_hashes'] for r in loads} == expected_frames
        groups = defaultdict(list)
        for row in maps:
            if row['trial']:
                groups[f"{row['case']}-{row['step']}"].append(row['gpu_ms'])
        result[str(cycle)] = {
            'exact_maps': len(maps), 'resident_bytes_unchanged': True,
            'selected_diffraction_exact': True,
            'load_p50_seconds': statistics.median(r['seconds'] for r in loads),
            'load_max_seconds': max(r['seconds'] for r in loads),
            'seven_sequential_load_seconds': sum(r['seconds'] for r in loads),
            'combined_decode_packing_gpu_p50_seconds': statistics.median(
                r['gpu_decode_packing_seconds'] for r in loads)
                if all(r.get('gpu_decode_packing_seconds') is not None for r in loads) else None,
            'detector_p50_gpu_ms': {k: statistics.median(v) for k, v in sorted(groups.items())},
        }
    return result


if __name__ == '__main__':
    reference = read(sys.argv[1])
    assert len(sys.argv) == 5, 'Expected reference and three experiment arms'
    result = {arm: summarize(reference, read(path))
              for arm, path in zip(['control_a', 'candidate', 'control_b'], sys.argv[2:])}
    result['cold_io_claim'] = False
    result['native_ui_claim'] = False
    print(json.dumps(result, indent=2))
