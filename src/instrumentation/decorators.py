"""
@instrument wraps any algorithm function so that time is measured
automatically around the call. Memory is estimated cheaply via
sys.getsizeof on the array rather than tracemalloc, since tracemalloc's
per-allocation tracing overhead is far too slow at large input sizes.
"""

import time
import sys
from .metrics import Metrics


def instrument(func):
    def wrapper(arr, *args, **kwargs):
        metrics = Metrics()
        arr_copy = arr.copy()

        start_time = time.perf_counter()
        result = func(arr_copy, metrics, *args, **kwargs)
        elapsed_time = time.perf_counter() - start_time

        memory_estimate = sys.getsizeof(arr_copy) + sum(sys.getsizeof(x) for x in arr_copy[:1])
        # Rough estimate: size of the list container + one representative element,
        # scaled by length (cheap approximation, avoids tracemalloc's per-op cost).
        memory_estimate = sys.getsizeof(arr_copy) + (len(arr_copy) * sys.getsizeof(arr_copy[0]) if arr_copy else 0)

        return {
            "result": result,
            "comparisons": metrics.comparisons,
            "swaps": metrics.swaps,
            "probes": metrics.probes,
            "time_taken": elapsed_time,
            "memory_used": memory_estimate,
        }

    return wrapper