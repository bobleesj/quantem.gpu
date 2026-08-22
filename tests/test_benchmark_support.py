from __future__ import annotations

import numpy as np

from scripts._benchmark_support import _release_array


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
