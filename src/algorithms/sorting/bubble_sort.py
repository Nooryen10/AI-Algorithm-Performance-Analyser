from ...instrumentation.decorators import instrument


@instrument
def bubble_sort(arr, metrics):
    n = len(arr)
    comparisons = 0
    swaps = 0

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                swapped = True
        if not swapped:
            break

    metrics.compare(comparisons)
    metrics.swap(swaps)
    return arr