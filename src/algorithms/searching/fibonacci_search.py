from ...instrumentation.decorators import instrument


@instrument
def fibonacci_search(arr, metrics, target):
    arr = sorted(arr)
    n = len(arr)

    fib2 = 0  # (m-2)'th Fibonacci number
    fib1 = 1  # (m-1)'th Fibonacci number
    fib = fib1 + fib2  # m'th Fibonacci number

    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2

    offset = -1

    while fib > 1:
        i = min(offset + fib2, n - 1)

        metrics.probe()
        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            return {"found": True, "index": i}

    if fib1 and offset + 1 < n:
        metrics.probe()
        if arr[offset + 1] == target:
            return {"found": True, "index": offset + 1}

    return {"found": False, "index": -1}