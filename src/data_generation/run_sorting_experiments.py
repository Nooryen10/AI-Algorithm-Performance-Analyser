"""
Runs all 7 sorting algorithms across 200 input sizes (10-2000, step 10),
5 data types, and 5 trials each -> 35,000 rows written to
data/raw/sorting_dataset.csv
"""

import csv
import os

from .array_generator import generate_array
from .feature_extractor import extract_sorting_features

from ..algorithms.sorting.bubble_sort import bubble_sort
from ..algorithms.sorting.selection_sort import selection_sort
from ..algorithms.sorting.insertion_sort import insertion_sort
from ..algorithms.sorting.merge_sort import merge_sort
from ..algorithms.sorting.quick_sort import quick_sort
from ..algorithms.sorting.heap_sort import heap_sort
from ..algorithms.sorting.radix_sort import radix_sort

ALGORITHMS = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "merge_sort": merge_sort,
    "quick_sort": quick_sort,
    "heap_sort": heap_sort,
    "radix_sort": radix_sort,
}

DATA_TYPES = ["random", "sorted", "reverse_sorted", "nearly_sorted", "duplicates"]
SIZES = list(range(10, 2001, 10))   # 200 sizes
TRIALS = 5

OUTPUT_PATH = os.path.join("data", "raw", "sorting_dataset.csv")

FIELDNAMES = [
    "algorithm", "input_size", "data_type", "trial_id",
    "comparisons", "swaps", "time_taken", "memory_used",
    "presortedness_score", "duplicate_ratio", "value_range",
]


def run():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    total_runs = len(ALGORITHMS) * len(SIZES) * len(DATA_TYPES) * TRIALS
    completed = 0

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for data_type in DATA_TYPES:
            for size in SIZES:
                for trial in range(1, TRIALS + 1):
                    # Same base array reused across all 7 algorithms for
                    # this (data_type, size, trial) combination, so their
                    # results are directly comparable.
                    base_array = generate_array(data_type, size)
                    features = extract_sorting_features(base_array)

                    for algo_name, algo_func in ALGORITHMS.items():
                        try:
                            output = algo_func(base_array)
                        except Exception as e:
                            print(f"FAILED: {algo_name} | {data_type} | size={size} | trial={trial} | {e}")
                            continue

                        row = {
                            "algorithm": algo_name,
                            "input_size": size,
                            "data_type": data_type,
                            "trial_id": trial,
                            "comparisons": output["comparisons"],
                            "swaps": output["swaps"],
                            "time_taken": output["time_taken"],
                            "memory_used": output["memory_used"],
                            **features,
                        }
                        writer.writerow(row)
                        completed += 1

                    if completed % 500 == 0:
                        print(f"Progress: {completed}/{total_runs} rows written")

    print(f"Done. {completed} rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()