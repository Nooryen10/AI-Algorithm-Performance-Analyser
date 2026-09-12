from ...instrumentation.decorators import instrument


@instrument
def binary_search(arr, metrics, target):
    arr = sorted(arr)
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        metrics.probe()
        if arr[mid] == target:
            return {"found": True, "index": mid}
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return {"found": False, "index": -1}