"""Compact changed columns across all 264 model contexts in one batch."""

from pathlib import Path

import cupy as cp
import numpy as np

_KERNELS = Path(__file__).with_name("kernels")


class Groups:
    """Immutable context membership with reusable request controls."""

    def __init__(self, ids: np.ndarray, models: int) -> None:
        self.ids = ids
        self.models = models
        detectors, starts = [], [0]
        # Model ordering is format metadata; this runs only during preparation.
        for row in ids:
            for model in range(models):
                group = np.flatnonzero(row == model).astype(np.int32)
                detectors.append(group)
                starts.append(starts[-1] + len(group))
        self.sorted_detectors = cp.asarray(np.concatenate(detectors))
        self.starts = cp.asarray(starts, cp.int32)
        self.segments = len(ids) * models * 2
        capacity = len(ids) * (36864 + 64 * models)
        self.counts = cp.empty(self.segments, cp.int32)
        self.group_counts = cp.empty_like(self.counts)
        self.group_offsets = cp.empty(self.segments + 1, cp.int32)
        self.selected = cp.empty(capacity, cp.int32)
        self.coefficients = cp.empty(capacity, cp.int8)
        self.context = cp.empty((capacity + 31) // 32, cp.int32)
        self.model = cp.empty_like(self.context)
        self.difference = cp.empty(36864, cp.int8)
        self.module = cp.RawModule(
            code=(_KERNELS / "groups.cu").read_text(),
            options=("--std=c++17",),
        )


class PaletteGroups:
    """Share tables across changed columns without an acquisition loop."""

    def __init__(self, base: Groups) -> None:
        self.base = base
        self.palette = 5
        capacity = len(base.selected)
        groups = (capacity + 31) // 32
        self.selected = cp.empty(capacity, cp.int32)
        self.slots = cp.empty(capacity, cp.uint8)
        self.context = cp.empty(groups, cp.int32)
        self.signs = cp.empty(groups, cp.int8)
        self.lengths = cp.empty(groups, cp.uint8)
        self.model_counts = cp.empty(groups, cp.uint8)
        self.models = cp.empty(groups * self.palette, cp.int32)
        self.counts = cp.empty(528, cp.int32)
        self.offsets = cp.empty(529, cp.int32)
        self.module = cp.RawModule(
            code=(_KERNELS / "palette.cu").read_text(),
            options=("--std=c++17",),
        )

    def build(self, difference: np.ndarray) -> int:
        """Return all-context group count after one compaction sequence."""
        base = self.base
        base.difference.set(np.asarray(difference, np.int8))
        base.module.get_function("count_groups")(
            (base.segments,),
            (256,),
            (
                base.sorted_detectors,
                base.starts,
                base.difference,
                base.counts,
                base.group_counts,
            ),
        )
        base.group_offsets[0] = 0
        cp.cumsum(base.group_counts, out=base.group_offsets[1:])
        base.module.get_function("scatter_groups")(
            (base.segments,),
            (256,),
            (
                base.sorted_detectors,
                base.starts,
                base.difference,
                base.counts,
                base.group_offsets,
                np.int32(base.models),
                base.selected,
                base.coefficients,
                base.context,
                base.model,
            ),
        )
        self.module.get_function("count_palette")(
            (528,),
            (32,),
            (base.counts, self.counts),
        )
        self.offsets[0] = 0
        cp.cumsum(self.counts, out=self.offsets[1:])
        self.module.get_function("scatter_palette")(
            (528,),
            (32,),
            (
                base.counts,
                base.group_offsets,
                base.selected,
                self.offsets,
                self.selected,
                self.slots,
                self.context,
                self.signs,
                self.lengths,
                self.model_counts,
                self.models,
            ),
        )
        return int(self.offsets[-1].get())
