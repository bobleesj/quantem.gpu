"""Compare exact original-HDF5 kernel variants with a fixed executable.

Every case runs in a fresh process and releases its residents. Frozen full-map
hashes authenticate each mask at every cycle. GPU/API timing is not native FPS.
The JSON case file contains a name and explicit environment overrides only.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    for name in ['exe', 'input', 'index', 'plans', 'out', 'reference', 'cases']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--trials', type=int, default=4)
    args = parser.parse_args()
    assert 2 <= args.repeats <= 10 and 2 <= args.trials <= 8
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    executable = Path(args.exe).resolve()
    binary_hash = digest(executable)
    def resource_hashes():
        return {str(path.relative_to(executable.parent)): digest(path)
                for path in sorted(executable.parent.glob('*.bundle/Resources/*.metal'))}
    resources = resource_hashes()
    assert resources, 'Missing runtime Metal resource bundle'
    reference = read(Path(args.reference))
    expected = {(r['source_identity'], r['case'], r['step'], r['trial']): r['sha256_u32_le']
                for r in reference if r.get('phase') == 'detector' and r['trial'] < args.trials}
    expected_loads = {r['source_identity']: r for r in reference if r.get('phase') == 'resident'}
    cases = json.loads(Path(args.cases).read_text())
    assert len({c['name'] for c in cases}) == len(cases)
    reports = []
    for case in cases:
        name = case['name']
        assert name and all(c.isalnum() or c in '-_' for c in name)
        stdout, stderr = output / (name + '.jsonl'), output / (name + '.stderr')
        env = {**os.environ, 'COMPACT_ORIGINAL_FUSED_PLANES': '1',
               'COMPACT_ORIGINAL_PLANES': '0', 'COMPACT_RESIDENCY_SETS': '0',
               'QGPU_ORIGINAL_PROFILE': '1', 'QGPU_ORIGINAL_STAGE_PROFILE': '0',
               'QGPU_ORIGINAL_DECODE_THREADS': '32', 'QGPU_ORIGINAL_PACK_THREADS': '32',
               'COMPACT_RAW_PLANE_ILP': '0', 'COMPACT_RAW_SCAN_COOPERATIVE': '0',
               'COMPACT_RAW_QUAD_SCAN': '0', 'COMPACT_DETECTOR_WIDTH_BUCKETS': '0',
               'COMPACT_RAW_QUAD_VECTOR': '0', 'COMPACT_RAW_OCTO_SCAN': '0',
               'COMPACT_RAW_VECTOR_THREADS': '128',
               'QGPU_ORIGINAL_PLANE_VECTOR4': '0', 'QGPU_ORIGINAL_PLANE_DEFERRED_VERIFY': '0',
               'QGPU_ORIGINAL_PLANE_VECTOR2': '0', 'QGPU_ORIGINAL_STANDARD_PLANES': '0',
               'QGPU_ORIGINAL_DIRECT_DPC': '0',
               'QGPU_ORIGINAL_ZERO_SCRATCH': '0',
               'QGPU_ORIGINAL_SORT_BLOCKS': '0', 'QGPU_ORIGINAL_SORT_BLOCK_BUCKETS': '0',
               'QGPU_ORIGINAL_COOPERATIVE_BLOCK': '0',
               'QGPU_ORIGINAL_LAZY_PLANES': '0',
               'QGPU_ORIGINAL_ZERO_TAIL': '0',
               'QGPU_ORIGINAL_FIXED_DECODE_PIPELINE': '0',
               'QGPU_ORIGINAL_FIXED_PACK_PIPELINE': '0',
               'COMPACT_PLANAR_WIDTH_BUCKETS': '0', 'COMPACT_PLANAR_WIDE_ONLY': '0',
               'QGPU_ORIGINAL_BOUND_ZERO_PLANES': '0',
               'QGPU_LIBRARY_REUSE': '0',
               'QGPU_ORIGINAL_FUSED_BLOCK_PLANES': '0',
               **case.get('environment', {})}
        assert digest(executable) == binary_hash, 'Executable changed during matrix'
        assert resource_hashes() == resources, 'Metal resources changed during matrix'
        report = {'case': name, 'environment': case.get('environment', {}),
                  'started_unix': time.time(), 'binary_sha256': binary_hash,
                  'metal_resource_sha256': resources, 'accepted': False}
        plan_directory = Path(args.plans)
        if 'plan_subdirectory' in case:
            suffix = case['plan_subdirectory']
            assert suffix and all(c.isalnum() or c in '-_' for c in suffix)
            plan_directory = plan_directory / suffix
        with stdout.open('w') as out, stderr.open('w') as errors:
            try:
                process = subprocess.run([str(executable), args.input, args.index,
                    '--repeats', str(args.repeats), '--reuse-products',
                    '--detector-trials', str(args.trials), '--budget-bytes', '8000000000',
                    '--plan-directory', str(plan_directory)], env=env, stdout=out, stderr=errors, timeout=900)
                report['exit'] = process.returncode
            except subprocess.TimeoutExpired:
                report['exit'] = 'timeout'
        report.update(finished_unix=time.time(), stdout_sha256=digest(stdout), stderr_sha256=digest(stderr))
        try:
            rows = read(stdout)
            assert report['exit'] == 0 and rows[-1]['phase'] == 'complete', 'Incomplete run'
            loads = [r for r in rows if r.get('phase') == 'resident']
            maps = [r for r in rows if r.get('phase') == 'detector']
            identities = {r['source_identity'] for r in loads}
            assert identities == set(expected_loads), 'Missing acquisition'
            assert len(loads) == args.repeats * len(identities), 'Missing load cycle'
            for cycle in range(args.repeats):
                selected = [r for r in maps if r['cycle'] == cycle]
                actual = {(r['source_identity'], r['case'], r['step'], r['trial']): r['sha256_u32_le']
                          for r in selected}
                assert len(selected) == len(expected) and actual == expected, 'Exact detector mismatch or missing map'
                cycle_loads = [r for r in loads if r['cycle'] == cycle]
                assert {r['source_identity'] for r in cycle_loads} == identities
            assert all(r['resident_bytes'] == expected_loads[r['source_identity']]['resident_bytes']
                       and r['sample_hashes'] == expected_loads[r['source_identity']]['sample_hashes'] for r in loads)
            profiles = [json.loads(line.split(' ', 1)[1]) for line in stderr.read_text().splitlines()
                        if line.startswith('ORIGINAL_PACK_PROFILE ')]
            library_hits = sum(line.startswith('QGPU_LIBRARY_CACHE ') and line.endswith('hit=1')
                               for line in stderr.read_text().splitlines())
            assert (library_hits > 0) == (env['QGPU_LIBRARY_REUSE'] == '1'), 'Library reuse setting was not executed'
            assert len(profiles) == len(loads), 'Missing or extra load-stage profile'
            for load, profile in zip(loads, profiles):
                assert profile['packing_plan_fallbacks'] == 0, 'Experimental path fell back'
                if env.get('QGPU_ORIGINAL_FIXED_DECODE_PIPELINE') == '1':
                    assert profile['decode_pipeline_thread_limit'] == 32, 'Decoder compiler limit was not applied'
                if env.get('QGPU_ORIGINAL_FIXED_PACK_PIPELINE') == '1':
                    assert profile['packing_pipeline_thread_limit'] == 32, 'Packing compiler limit was not applied'
                if 'QGPU_ORIGINAL_WINDOW_FRAMES' in env:
                    assert profile['decode_window_frames'] == int(env['QGPU_ORIGINAL_WINDOW_FRAMES']), 'Wrong processing window'
                direct = load['cycle'] > 0 or env['QGPU_ORIGINAL_DIRECT_DPC'] == '1'
                expected_layout = int(env['COMPACT_ORIGINAL_FUSED_PLANES'] == '1') if direct else int(env['QGPU_ORIGINAL_STANDARD_PLANES'] == '1')
                assert profile['packed_payload_layout'] == expected_layout, 'Wrong resident layout'
                if direct:
                    assert profile['direct_bitshuffle_windows'] > 0, 'Did not execute direct source packing'
                expected_zero = profile['direct_bitshuffle_windows'] if env['QGPU_ORIGINAL_ZERO_SCRATCH'] == '1' else 0
                assert profile.get('zero_initialized_scratch_windows', 0) == expected_zero, 'Wrong zero-scratch decode path'
                expected_sorted = profile['scalar_decode_slices'] if direct and env['QGPU_ORIGINAL_SORT_BLOCKS'] == '1' else 0
                assert profile.get('sorted_block_slices', 0) == expected_sorted, 'Wrong sorted-block decode path'
                cooperative = profile.get('cooperative_block_slices', 0)
                assert (cooperative > 0) == (direct and env['QGPU_ORIGINAL_COOPERATIVE_BLOCK'] == '1'), 'Wrong cooperative decode path'
                lazy = profile.get('lazy_plane_slices', 0)
                assert (profile.get('zero_tail_slices', 0) > 0) == (load['cycle'] > 0 and env['QGPU_ORIGINAL_ZERO_TAIL'] == '1'), 'Wrong zero-tail decode path'
                assert (profile.get('bounded_zero_plane_windows', 0) > 0) == (load['cycle'] > 0 and env['QGPU_ORIGINAL_ZERO_TAIL'] == '1' and env['QGPU_ORIGINAL_BOUND_ZERO_PLANES'] == '1'), 'Wrong bounded-plane path'
                expected_lazy = load['cycle'] > 0 and env['QGPU_ORIGINAL_LAZY_PLANES'] == '1'
                assert (lazy > 0) == expected_lazy, 'Wrong lazy-plane decode path'
                fused = profile.get('fused_block_plane_windows', 0)
                assert (fused > 0) == (direct and env['QGPU_ORIGINAL_FUSED_BLOCK_PLANES'] == '1'), 'Wrong fused-block decode path'
            released = [r['device_allocated_bytes'] for r in rows if r.get('phase') == 'released']
            assert len(released) == len(loads) and max(released) <= released[0] + (64 << 20), 'Release memory budget failed'
            grouped, per_source, wall = defaultdict(list), defaultdict(list), defaultdict(list)
            for row in maps:
                if row['cycle'] > 0 and row['trial'] > 0:
                    key = f"{row['case']}-{row['step']}"
                    grouped[key].append(row['gpu_ms'])
                    per_source[row['source_identity'] + ':' + key].append(row['gpu_ms'])
                    wall[key].append(row['call_ms_including_mask_preparation'])
            report.update(accepted=True, exact_maps=len(maps), full_loads=len(loads),
                release_max_bytes=max(released), resident_bytes_unchanged=True,
                load_stage_profiles=len(profiles), experimental_fallbacks=0,
                library_cache_hits=library_hits,
                reopen_p50_seconds=statistics.median(r['seconds'] for r in loads if r['cycle'] > 0),
                first_p50_seconds=statistics.median(r['seconds'] for r in loads if r['cycle'] == 0),
                gpu_p50_ms={k: statistics.median(v) for k, v in sorted(grouped.items())},
                gpu_per_source_p50_ms={k: statistics.median(v) for k, v in sorted(per_source.items())},
                call_p50_ms={k: statistics.median(v) for k, v in sorted(wall.items())},
                device_allocation_excess_max_bytes=max(r['device_allocated_after_bytes']
                    - expected_loads[r['source_identity']]['device_allocated_after_bytes'] for r in loads),
                native_ui_claim=False, cold_io_claim=False)
        except (AssertionError, ValueError, KeyError, IndexError) as error:
            report['failure'] = str(error)
        reports.append(report)
        (output / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if 'per_source' not in k}), flush=True)
        if not report['accepted']:
            raise RuntimeError(f"{name}: {report.get('failure', 'acceptance failed')}; evidence retained")
    assert digest(executable) == binary_hash
    assert resource_hashes() == resources


if __name__ == '__main__':
    main()
