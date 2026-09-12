from .array_generator import generate_array
from .feature_extractor import extract_sorting_features

for dtype in ["random", "sorted", "reverse_sorted", "nearly_sorted", "duplicates"]:
    arr = generate_array(dtype, size=500)
    features = extract_sorting_features(arr)
    print(f"{dtype:>15}: {features}")