from ...instrumentation.decorators import instrument


@instrument
def selection_sort(arr, metrics):
    n = len(arr)
    comparisons = 0
    swaps = 0

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1

    metrics.compare(comparisons)
    metrics.swap(swaps)
    return arr