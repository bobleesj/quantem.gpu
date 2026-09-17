#!/usr/bin/env python3
"""Compare a sequential control fit against a batched fit trial by trial."""
import json
import sys


def main(control_path, batched_path):
    control = json.load(open(control_path))
    batched = json.load(open(batched_path))
    print(f"control: k={control['k']} seconds={control['fit_seconds']:.2f} "
          f"loss={control['fit_loss']:.16f} best={control['fit_best']}")
    print(f"batched: k={batched['k']} seconds={batched['fit_seconds']:.2f} "
          f"loss={batched['fit_loss']:.16f} best={batched['fit_best']}")
    ct, bt = control["trials"], batched["trials"]
    print(f"trial counts: control={len(ct)} batched={len(bt)}")
    first_divergence = None
    shared = 0
    for index, (c, b) in enumerate(zip(ct, bt)):
        same_point = (
            c["c10"] == b["c10"] and c["c12"] == b["c12"] and c["phi12"] == b["phi12"]
        )
        same_loss = c["loss"] == b["loss"]
        if same_point and same_loss:
            shared += 1
        elif first_divergence is None:
            first_divergence = index
    print(f"identical trial entries: {shared}, first divergence index: {first_divergence}")
    if first_divergence is not None:
        index = first_divergence
        print(f"  control[{index}]: {ct[index]}")
        print(f"  batched[{index}]: {bt[index]}")
    # Loss agreement at shared indices
    deltas = [
        abs(c["loss"] - b["loss"])
        for c, b in zip(ct, bt)
        if c["c10"] == b["c10"] and c["c12"] == b["c12"] and c["phi12"] == b["phi12"]
    ]
    if deltas:
        print(f"shared-index loss deltas: max={max(deltas):.3e} "
              f"nonzero={sum(1 for d in deltas if d)}")
    print(f"final loss delta: {batched['fit_loss'] - control['fit_loss']:.3e}")
    for key in ("stages", "refinement_evaluations", "sequential_verification"):
        print(f"{key}: control={control.get(key)} batched={batched.get(key)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
