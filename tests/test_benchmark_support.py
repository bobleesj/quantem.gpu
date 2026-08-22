from __future__ import annotations

import numpy as np

from scripts._benchmark_support import MemorySampler, _release_array


class _ManagedArray:
    def __init__(self) -> None:
        self.free_count = 0

    def free(self) -> None:
        self.free_count += 1


def test_release_array_frees_explicitly_managed_output() -> None:
    array = _ManagedArray()

    method = _release_array(array)

    assert method == "free"
    assert array.free_count == 1


def test_release_array_leaves_allocator_managed_output_alone() -> None:
    array = np.zeros((2, 3), dtype=np.uint16)

    method = _release_array(array)

    assert method is None


def test_memory_sampler_reports_numeric_peaks() -> None:
    sampler = MemorySampler("cpu", interval_ms=1)

    sampler.start()
    sample = sampler.stop()

    assert sample["sample_count"] >= 2
    assert sample["interval_ms"] == 1
    assert sample["peak"]["process_peak_rss_bytes"] > 0
