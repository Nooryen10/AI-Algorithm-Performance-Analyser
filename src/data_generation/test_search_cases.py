from .array_generator import generate_array
from .search_case_generator import generate_search_target

arr = generate_array("duplicates", size=20)
print(f"Array: {arr}\n")

for case in ["best_case", "average_case", "worst_case", "not_found", "duplicates_present"]:
    target = generate_search_target(case, arr)
    print(f"{case:>20}: target = {target}")