from ...instrumentation.decorators import instrument


def _counting_sort_by_digit(arr, exp, counters):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
        counters["swaps"] += 1

    for i in range(n):
        arr[i] = output[i]


@instrument
def radix_sort(arr, metrics):
    counters = {"comparisons": 0, "swaps": 0}
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        _counting_sort_by_digit(arr, exp, counters)
        exp *= 10

    metrics.compare(counters["comparisons"])
    metrics.swap(counters["swaps"])
    return arr