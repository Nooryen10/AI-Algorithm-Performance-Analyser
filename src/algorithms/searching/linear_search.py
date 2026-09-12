from ...instrumentation.decorators import instrument


@instrument
def linear_search(arr, metrics, target):
    for i, val in enumerate(arr):
        metrics.probe()
        if val == target:
            return {"found": True, "index": i}
    return {"found": False, "index": -1}