from ...instrumentation.decorators import instrument


def _binary_search_range(arr, metrics, target, low, high):
    while low <= high:
        mid = (low + high) // 2
        metrics.probe()
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


@instrument
def exponential_search(arr, metrics, target):
    arr = sorted(arr)
    n = len(arr)

    if n == 0:
        return {"found": False, "index": -1}

    metrics.probe()
    if arr[0] == target:
        return {"found": True, "index": 0}

    i = 1
    while i < n:
        metrics.probe()
        if arr[i] >= target:
            break
        i *= 2

    low = i // 2
    high = min(i, n - 1)
    idx = _binary_search_range(arr, metrics, target, low, high)

    if idx != -1:
        return {"found": True, "index": idx}
    return {"found": False, "index": -1}