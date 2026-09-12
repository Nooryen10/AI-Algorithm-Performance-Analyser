"""
Runs all 6 searching algorithms across 200 input sizes (10-2000, step 10),
5 case types, and 5 trials each -> 30,000 rows written to
data/raw/searching_dataset.csv
"""

import csv
import os

from .array_generator import generate_array
from .search_case_generator import generate_search_target

from ..algorithms.searching.linear_search import linear_search
from ..algorithms.searching.binary_search import binary_search
from ..algorithms.searching.interpolation_search import interpolation_search
from ..algorithms.searching.exponential_search import exponential_search
from ..algorithms.searching.fibonacci_search import fibonacci_search
from ..algorithms.searching.hashing_search import hashing_search


ALGORITHMS = {
    "linear_search": linear_search,
    "binary_search": binary_search,
    "interpolation_search": interpolation_search,
    "exponential_search": exponential_search,
    "fibonacci_search": fibonacci_search,
    "hashing_search": hashing_search,
}

CASE_TYPES = ["best_case", "average_case", "worst_case", "not_found", "duplicates_present"]
SIZES = list(range(10, 2001, 10))   # 200 sizes  
TRIALS = 5

OUTPUT_PATH = os.path.join("data", "raw", "searching_dataset.csv")

FIELDNAMES = [
    "algorithm", "input_size", "case_type", "trial_id",
    "probes", "time_taken", "memory_used", "found",
    "load_factor", "collision_count",
]


def run():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    total_runs = len(ALGORITHMS) * len(SIZES) * len(CASE_TYPES) * TRIALS
    completed = 0

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for case_type in CASE_TYPES:
            for size in SIZES:
                for trial in range(1, TRIALS + 1):
                    base_array = generate_array("random", size)
                    target = generate_search_target(case_type, base_array)

                    for algo_name, algo_func in ALGORITHMS.items():
                        print(f"Running {algo_name} | {case_type} | size={size} | trial={trial}")
                        try:
                            output = algo_func(base_array, target)
                        except Exception as e:
                            print(f"FAILED: {algo_name} | {case_type} | size={size} | trial={trial} | {e}")
                            continue

                        result = output["result"]
                        row = {
                            "algorithm": algo_name,
                            "input_size": size,
                            "case_type": case_type,
                            "trial_id": trial,
                            "probes": output["probes"],
                            "time_taken": output["time_taken"],
                            "memory_used": output["memory_used"],
                            "found": result.get("found"),
                            "load_factor": result.get("load_factor", ""),
                            "collision_count": result.get("collision_count", ""),
                        }
                        writer.writerow(row)
                        completed += 1

                    if completed % 500 == 0:
                        print(f"Progress: {completed}/{total_runs} rows written")

    print(f"Done. {completed} rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()