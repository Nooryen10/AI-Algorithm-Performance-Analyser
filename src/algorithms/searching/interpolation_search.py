from ...instrumentation.decorators import instrument


@instrument
def interpolation_search(arr, metrics, target):
    arr = sorted(arr)
    low, high = 0, len(arr) - 1

    while low <= high and arr[low] <= target <= arr[high]:
        if arr[high] == arr[low]:
            metrics.probe()
            if arr[low] == target:
                return {"found": True, "index": low}
            return {"found": False, "index": -1}

        pos = low + int(
            ((target - arr[low]) * (high - low)) / (arr[high] - arr[low])
        )
        pos = max(low, min(pos, high))

        metrics.probe()
        if arr[pos] == target:
            return {"found": True, "index": pos}
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1

    return {"found": False, "index": -1}