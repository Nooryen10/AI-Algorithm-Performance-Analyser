"""
Generates a target value for a given array to produce a specific search
case (best-case, average-case, worst-case, not-found, duplicates-present).
The array should already be sorted where the case logic assumes ordering
(binary/interpolation/exponential/fibonacci search all sort internally
anyway, but we reason about position here on the sorted version).
"""

import random


def best_case_target(arr):
    """
    For most searches, the 'best case' is finding the target in the
    fewest possible probes. For sorted-array searches, that's usually the
    middle element (binary search finds it in 1 probe). For linear search,
    best case is the first element. We use the middle element as a
    reasonable general best-case target across algorithms.
    """
    sorted_arr = sorted(arr)
    mid = len(sorted_arr) // 2
    return sorted_arr[mid]


def average_case_target(arr):
    """A random existing element, simulating a 'typical' search."""
    return random.choice(arr)


def worst_case_target(arr):
    """
    Worst case for linear search is the last element (or absent).
    For sorted-array searches, elements near the boundaries tend to need
    more probes. We use the last element of the sorted array.
    """
    sorted_arr = sorted(arr)
    return sorted_arr[-1]


def not_found_target(arr):
    """A value guaranteed to be outside the array's value range."""
    return max(arr) + 1000


def duplicates_present_target(arr):
    """
    Picks a value that appears more than once in the array, if any exist;
    otherwise falls back to an average-case target.
    """
    from collections import Counter
    counts = Counter(arr)
    duplicated_values = [val for val, count in counts.items() if count > 1]
    if duplicated_values:
        return random.choice(duplicated_values)
    return average_case_target(arr)


CASE_GENERATORS = {
    "best_case": best_case_target,
    "average_case": average_case_target,
    "worst_case": worst_case_target,
    "not_found": not_found_target,
    "duplicates_present": duplicates_present_target,
}


def generate_search_target(case_type, arr):
    if case_type not in CASE_GENERATORS:
        raise ValueError(f"Unknown case_type '{case_type}'. Valid options: {list(CASE_GENERATORS.keys())}")
    return CASE_GENERATORS[case_type](arr)