from .bubble_sort import bubble_sort
from .selection_sort import selection_sort
from .insertion_sort import insertion_sort
from .merge_sort import merge_sort
from .quick_sort import quick_sort
from .heap_sort import heap_sort
from .radix_sort import radix_sort

test_arr = [5, 2, 9, 1, 5, 6, 3, 8, 7, 4]

for name, func in [
    ("Bubble", bubble_sort), ("Selection", selection_sort),
    ("Insertion", insertion_sort), ("Merge", merge_sort),
    ("Quick", quick_sort), ("Heap", heap_sort), ("Radix", radix_sort),
]:
    output = func(test_arr)
    print(f"{name}: {output['result']} | comparisons={output['comparisons']}, swaps={output['swaps']}, time={output['time_taken']:.6f}")