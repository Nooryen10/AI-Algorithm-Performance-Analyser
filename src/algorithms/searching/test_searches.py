from .linear_search import linear_search
from .binary_search import binary_search
from .interpolation_search import interpolation_search
from .exponential_search import exponential_search
from .fibonacci_search import fibonacci_search
from .hashing_search import hashing_search

test_arr = [5, 2, 9, 1, 5, 6, 3, 8, 7, 4]
target_found = 7
target_missing = 100

for name, func in [
    ("Linear", linear_search), ("Binary", binary_search),
    ("Interpolation", interpolation_search), ("Exponential", exponential_search),
    ("Fibonacci", fibonacci_search), ("Hashing", hashing_search),
]:
    out_found = func(test_arr, target_found)
    out_missing = func(test_arr, target_missing)
    print(f"{name}: found={out_found['result']} (probes={out_found['probes']}) | "
          f"missing={out_missing['result']} (probes={out_missing['probes']})")