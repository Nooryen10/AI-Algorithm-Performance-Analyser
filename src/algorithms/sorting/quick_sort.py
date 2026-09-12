import random
from ...instrumentation.decorators import instrument


def _partition(arr, low, high, counters):
    # Randomize pivot choice to avoid worst-case behavior on sorted/
    # reverse-sorted input (which would otherwise always pick an extreme
    # element as pivot and degrade to O(n^2) time / O(n) recursion depth).
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]

    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        counters["comparisons"] += 1
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            counters["swaps"] += 1
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    counters["swaps"] += 1
    return i + 1


def _quick_sort_helper(arr, low, high, counters):
    # Iterative on the larger partition, recursive only on the smaller
    # one -> guarantees O(log n) recursion depth regardless of pivot luck,
    # instead of risking O(n) depth on adversarial/sorted input.
    while low < high:
        pivot_idx = _partition(arr, low, high, counters)

        if pivot_idx - low < high - pivot_idx:
            _quick_sort_helper(arr, low, pivot_idx - 1, counters)
            low = pivot_idx + 1
        else:
            _quick_sort_helper(arr, pivot_idx + 1, high, counters)
            high = pivot_idx - 1


@instrument
def quick_sort(arr, metrics):
    counters = {"comparisons": 0, "swaps": 0}
    _quick_sort_helper(arr, 0, len(arr) - 1, counters)
    metrics.compare(counters["comparisons"])
    metrics.swap(counters["swaps"])
    return arr