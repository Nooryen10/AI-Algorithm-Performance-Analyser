from ...instrumentation.decorators import instrument


def _merge(arr, left, mid, right, counters):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        counters["comparisons"] += 1
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        counters["swaps"] += 1
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        counters["swaps"] += 1
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        counters["swaps"] += 1
        j += 1
        k += 1


def _merge_sort_helper(arr, left, right, counters):
    if left < right:
        mid = (left + right) // 2
        _merge_sort_helper(arr, left, mid, counters)
        _merge_sort_helper(arr, mid + 1, right, counters)
        _merge(arr, left, mid, right, counters)


@instrument
def merge_sort(arr, metrics):
    counters = {"comparisons": 0, "swaps": 0}
    _merge_sort_helper(arr, 0, len(arr) - 1, counters)
    metrics.compare(counters["comparisons"])
    metrics.swap(counters["swaps"])
    return arr