from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import _benchmark_support as support
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


def test_darwin_vm_stat_normalizes_counters_to_bytes(monkeypatch) -> None:
    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPageouts: 3.\nSwapins: 5.\nSwapouts: 7.\nPages occupied by compressor: 11.\n"""
    monkeypatch.setattr(support.sys, "platform", "darwin")
    monkeypatch.setattr(
        support.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    snapshot = support._darwin_vm_stat()

    assert snapshot == {
        "page_size_bytes": 16_384,
        "pageout_bytes": 3 * 16_384,
        "swapin_bytes": 5 * 16_384,
        "swapout_bytes": 7 * 16_384,
        "compressor_bytes": 11 * 16_384,
    }
