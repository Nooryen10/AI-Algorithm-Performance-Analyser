from ...instrumentation.decorators import instrument


def _heapify(arr, n, i, counters):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n:
        counters["comparisons"] += 1
        if arr[left] > arr[largest]:
            largest = left

    if right < n:
        counters["comparisons"] += 1
        if arr[right] > arr[largest]:
            largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        counters["swaps"] += 1
        _heapify(arr, n, largest, counters)


@instrument
def heap_sort(arr, metrics):
    counters = {"comparisons": 0, "swaps": 0}
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i, counters)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        counters["swaps"] += 1
        _heapify(arr, i, 0, counters)

    metrics.compare(counters["comparisons"])
    metrics.swap(counters["swaps"])
    return arr