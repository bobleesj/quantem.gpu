"""Run a bounded, shuffled threadgroup matrix on one unchanged original source.

Each fresh process loads all counts three times. Only exact DPC metadata and
packing plans are reused, never a4D count cache. Repeated control cases reveal
drift; selected winners still require the full seven-source acceptance matrix.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ['exe', 'input', 'index', 'plans', 'out', 'reference']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    executable = Path(args.exe).resolve()
    executable_hash = digest(executable)
    reference = [json.loads(line) for line in Path(args.reference).read_text().splitlines()]
    expected = {(r['source_identity'], r['case'], r['step']): r['sha256_u32_le']
                for r in reference if r.get('phase') == 'detector'}
    resident_bytes = {r['source_identity']: r['resident_bytes'] for r in reference
                      if r.get('phase') == 'resident'}
    frames = {r['source_identity']: r['sample_hashes'] for r in reference
              if r.get('phase') == 'resident'}
    cases = [(layout, decode, pack) for layout in [0, 1]
             for decode in [32, 64, 128, 256, 512] for pack in [32, 64, 128, 256, 512]]
    random.Random(20260908).shuffle(cases)
    schedule = []
    for index, case in enumerate(cases):
        if index % 5 == 0:
            schedule.append((0, 128, 128, True))
        schedule.append((*case, False))
    summaries = []
    for ordinal, (layout, decode, pack, control) in enumerate(schedule):
        name = f'{ordinal:02d}-plane{layout}-d{decode}-p{pack}'
        stdout, stderr = out / (name + '.jsonl'), out / (name + '.stderr')
        env = {**os.environ, 'COMPACT_ORIGINAL_FUSED_PLANES': str(layout),
               'COMPACT_ORIGINAL_PLANES': '0', 'COMPACT_RESIDENCY_SETS': '0',
               'QGPU_ORIGINAL_PROFILE': '1', 'QGPU_ORIGINAL_STAGE_PROFILE': '0',
               'QGPU_ORIGINAL_DECODE_THREADS': str(decode), 'QGPU_ORIGINAL_PACK_THREADS': str(pack)}
        assert digest(executable) == executable_hash, 'Benchmark binary changed mid-matrix'
        start = time.time()
        with stdout.open('w') as output, stderr.open('w') as errors:
            try:
                process = subprocess.run([str(executable), args.input, args.index,
                    '--repeats', '3', '--reuse-products', '--detector-trials', '1',
                    '--budget-bytes', '8000000000', '--plan-directory', args.plans],
                    env=env, stdout=output, stderr=errors, timeout=120)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code = 'timeout'
        result = {'case': name, 'layout': layout, 'decode_threads': decode,
                  'pack_threads': pack, 'control': control, 'exit': exit_code,
                  'started_unix': start, 'finished_unix': time.time(),
                  'executable_sha256': executable_hash, 'stdout_sha256': digest(stdout),
                  'stderr_sha256': digest(stderr), 'accepted': False}
        try:
            rows = [json.loads(line) for line in stdout.read_text().splitlines()]
            assert exit_code == 0 and rows[-1]['phase'] == 'complete', 'Incomplete run'
            loads = [r for r in rows if r.get('phase') == 'resident']
            maps = [r for r in rows if r.get('phase') == 'detector']
            assert len(loads) == 3 and len(maps) == 45, 'Incomplete source/cycle coverage'
            assert {r['cycle'] for r in loads} == {0, 1, 2}
            assert len({(r['cycle'], r['case'], r['step']) for r in maps}) == 45
            assert all(r['resident_bytes'] == resident_bytes[r['source_identity']]
                       and r['sample_hashes'] == frames[r['source_identity']] for r in loads)
            assert all(r['sha256_u32_le'] == expected[(r['source_identity'], r['case'], r['step'])]
                       for r in maps), 'Exact detector parity failed'
            repeated = [r for r in loads if r['cycle'] > 0]
            result.update(accepted=True, exact_maps=len(maps), resident_bytes=loads[0]['resident_bytes'],
                repeated_p50_seconds=statistics.median(r['seconds'] for r in repeated),
                repeated_gpu_p50_seconds=statistics.median(r['gpu_decode_packing_seconds'] for r in repeated),
                maximum_released_bytes=max(r['device_allocated_bytes'] for r in rows
                                           if r.get('phase') == 'released'))
        except (AssertionError, KeyError, ValueError) as error:
            result['failure'] = str(error)
        summaries.append(result)
        # Generated measurement artifact, updated after every independent case.
        (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    assert digest(executable) == executable_hash


if __name__ == '__main__':
    main()
