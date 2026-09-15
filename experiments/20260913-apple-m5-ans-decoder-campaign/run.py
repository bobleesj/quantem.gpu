"""Run interleaved exact decoder probes in one retained-resident process."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess


def main():
    """Retain seven sources and collect one raw response per planned trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--phased-readers", action="store_true")
    variants.add_argument("--pair-unroll", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = [
        ("allocation-control", {"reuse_scratch": False}),
        ("reuse-control", {"reuse_scratch": True}),
        ("branchless-pop", {"reuse_scratch": True, "branchless_pop": True}),
        ("refill-16", {"reuse_scratch": True, "refill_threshold": 16}),
        ("refill-24", {"reuse_scratch": True, "refill_threshold": 24}),
    ]
    if args.phased_readers:
        arms = [("reuse-control", {"reuse_scratch": True}),
                ("phased-readers", {"reuse_scratch": True, "phased_readers": True})]
    if args.pair_unroll:
        arms = [("reuse-control", {"reuse_scratch": True})] + [
            (f"unroll-{factor}", {"reuse_scratch": True, "pair_unroll": factor})
            for factor in [2, 4, 8]]
    schedule = [(name, options, -1) for name, options in arms]
    generator = random.Random(1709)
    for repetition in range(20):
        order = list(arms)
        generator.shuffle(order)
        schedule.extend((name, options, repetition) for name, options in order)
    environment = os.environ | {
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_PREPARE_BRANCHLESS_POP": "1",
        "QGPU_PREPARE_REFILL_THRESHOLDS": "1",
        "QGPU_PREPARE_PHASED_READERS": "1" if args.phased_readers else "0",
        "QGPU_PREPARE_PAIR_UNROLL": "1" if args.pair_unroll else "0",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
    }
    with (args.out / "raw.jsonl").open("x") as raw, \
            (args.out / "trials.jsonl").open("x") as trials, \
            (args.out / "stderr.log").open("x") as stderr:
        process = subprocess.Popen(
            [str(args.exe.resolve()), str(args.folder), str(args.cache)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, bufsize=1, env=environment,
        )

        def receive(event):
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                record = json.loads(line)
                if record.get("event") == "ans_resident_loop_error":
                    raise RuntimeError(record)
                if record.get("event") == event:
                    return record
            raise RuntimeError(f"Benchmark exited before {event}: {process.poll()}")

        try:
            receive("ans_resident_loop_ready")
            print("Seven residents ready; starting interleaved probes", flush=True)
            for name, options, repetition in schedule:
                command = {"op": "stage_isolation", **options}
                process.stdin.write(json.dumps(command) + "\n")
                process.stdin.flush()
                record = receive("stage_isolation")
                assert record["exact"], record
                trials.write(json.dumps({"arm": name, "repetition": repetition,
                                         "response": record}) + "\n")
                trials.flush()
                if name == "reuse-control":
                    print(f"Repetition {repetition}: parity passed", flush=True)
            if args.pair_unroll:
                process.stdin.write(json.dumps({"op": "run", "mode": "indexed",
                                                "batch": False, "cycles": 2,
                                                "arm": "A1"}) + "\n")
                process.stdin.flush()
                regression = receive("ans_resident_loop_result")
                assert regression["fullmap_parity"]
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            receive("ans_resident_loop_end")
            if process.wait(timeout=30) != 0:
                raise RuntimeError("Benchmark failed on teardown")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
