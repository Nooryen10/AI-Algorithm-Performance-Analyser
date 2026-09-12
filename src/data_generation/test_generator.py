from .array_generator import generate_array

for dtype in ["random", "sorted", "reverse_sorted", "nearly_sorted", "duplicates"]:
    arr = generate_array(dtype, size=20)
    print(f"{dtype:>15}: {arr}")