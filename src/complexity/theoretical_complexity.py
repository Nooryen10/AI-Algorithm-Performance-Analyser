"""
Theoretical Big-O complexity reference for all 13 algorithms, used in the
Complexity Analysis section of the report and for comparing against the
empirical (measured) results from the datasets.
"""

SORTING_COMPLEXITY = {
    "bubble_sort": {
        "best": "O(n)",
        "average": "O(n^2)",
        "worst": "O(n^2)",
        "space": "O(1)",
        "stable": True,
        "notes": "Best case O(n) achieved via early-exit when no swaps occur in a pass (already sorted).",
    },
    "selection_sort": {
        "best": "O(n^2)",
        "average": "O(n^2)",
        "worst": "O(n^2)",
        "space": "O(1)",
        "stable": False,
        "notes": "Always scans the remaining unsorted portion fully, regardless of input order.",
    },
    "insertion_sort": {
        "best": "O(n)",
        "average": "O(n^2)",
        "worst": "O(n^2)",
        "space": "O(1)",
        "stable": True,
        "notes": "Best case O(n) on already-sorted input, since the inner while loop exits immediately.",
    },
    "merge_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
        "stable": True,
        "notes": "Consistent performance regardless of input order; extra O(n) space for the merge step.",
    },
    "quick_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n^2)",
        "space": "O(log n)",
        "stable": False,
        "notes": "Worst case occurs with poor pivot choices (e.g., always picking an extreme element on sorted input). Mitigated here via randomized pivot selection and recursing on the smaller partition.",
    },
    "heap_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(1)",
        "stable": False,
        "notes": "Consistent performance; in-place sorting using a binary heap structure.",
    },
    "radix_sort": {
        "best": "O(d*(n+k))",
        "average": "O(d*(n+k))",
        "worst": "O(d*(n+k))",
        "space": "O(n+k)",
        "stable": True,
        "notes": "d = number of digits, k = base (10 here). Non-comparison sort; only works on non-negative integers in this implementation.",
    },
}

SEARCHING_COMPLEXITY = {
    "linear_search": {
        "best": "O(1)",
        "average": "O(n)",
        "worst": "O(n)",
        "space": "O(1)",
        "requires_sorted": False,
        "notes": "Best case when target is the first element checked.",
    },
    "binary_search": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
        "requires_sorted": True,
        "notes": "Best case when target is exactly the middle element.",
    },
    "interpolation_search": {
        "best": "O(1)",
        "average": "O(log log n)",
        "worst": "O(n)",
        "space": "O(1)",
        "requires_sorted": True,
        "notes": "Worst case occurs with non-uniformly distributed data, where the position estimate is consistently poor.",
    },
    "exponential_search": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
        "requires_sorted": True,
        "notes": "Combines an exponential range-finding phase with binary search within that range.",
    },
    "fibonacci_search": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
        "requires_sorted": True,
        "notes": "Similar to binary search but splits the search range using Fibonacci numbers instead of simple bisection.",
    },
    "hashing_search": {
        "best": "O(1)",
        "average": "O(1)",
        "worst": "O(n)",
        "space": "O(n)",
        "requires_sorted": False,
        "notes": "Worst case occurs when many keys collide into the same bucket (high load factor). This implementation uses chaining and deliberately undersizes the table to surface measurable collisions.",
    },
}


def get_complexity(algorithm_name):
    if algorithm_name in SORTING_COMPLEXITY:
        return SORTING_COMPLEXITY[algorithm_name]
    if algorithm_name in SEARCHING_COMPLEXITY:
        return SEARCHING_COMPLEXITY[algorithm_name]
    raise ValueError(f"Unknown algorithm '{algorithm_name}'")


def print_complexity_table():
    print(f"{'Algorithm':<22}{'Best':<15}{'Average':<15}{'Worst':<15}{'Space':<10}")
    print("-" * 77)
    for name, data in {**SORTING_COMPLEXITY, **SEARCHING_COMPLEXITY}.items():
        print(f"{name:<22}{data['best']:<15}{data['average']:<15}{data['worst']:<15}{data['space']:<10}")


if __name__ == "__main__":
    print_complexity_table()