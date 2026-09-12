"""
Generates arrays of a given size, in 5 distribution types, for benchmarking
the sorting algorithms.
"""

import random


def generate_random_array(size, low=0, high=100000):
    return [random.randint(low, high) for _ in range(size)]


def generate_sorted_array(size, low=0, high=100000):
    arr = generate_random_array(size, low, high)
    arr.sort()
    return arr


def generate_reverse_sorted_array(size, low=0, high=100000):
    arr = generate_sorted_array(size, low, high)
    arr.reverse()
    return arr


def generate_nearly_sorted_array(size, low=0, high=100000, disturb_fraction=0.05):
    """
    Starts from a sorted array, then swaps a small fraction of random
    element pairs to introduce controlled disorder (~5% by default).
    """
    arr = generate_sorted_array(size, low, high)
    num_swaps = max(1, int(size * disturb_fraction))

    for _ in range(num_swaps):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]

    return arr


def generate_duplicates_array(size, num_unique_values=10, low=0, high=100000):
    """
    Generates an array with many repeated values by sampling from a small
    pool of unique values (default: only 10 distinct values in the array).
    """
    pool = [random.randint(low, high) for _ in range(num_unique_values)]
    return [random.choice(pool) for _ in range(size)]


GENERATORS = {
    "random": generate_random_array,
    "sorted": generate_sorted_array,
    "reverse_sorted": generate_reverse_sorted_array,
    "nearly_sorted": generate_nearly_sorted_array,
    "duplicates": generate_duplicates_array,
}


def generate_array(data_type, size, **kwargs):
    if data_type not in GENERATORS:
        raise ValueError(f"Unknown data_type '{data_type}'. Valid options: {list(GENERATORS.keys())}")
    return GENERATORS[data_type](size, **kwargs)