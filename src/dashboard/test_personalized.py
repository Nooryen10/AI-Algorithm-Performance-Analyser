"""
Headless tests for the Personalized Prediction feature (and a regression check
that every pre-existing dashboard page still renders).

Run from the repository root:
    python -m src.dashboard.test_personalized
"""

import json
import os
import re
import sys

from streamlit.testing.v1 import AppTest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.dashboard.personalized import (  # noqa: E402
    parse_user_numbers, validate_user_data,
    derive_sorting_condition, derive_search_case)

APP = os.path.join(os.path.dirname(__file__), "app.py")
SORT_ALGOS = ["bubble_sort", "quick_sort", "radix_sort"]
SORT_CONDS = ["random", "sorted"]
EXISTING_PAGES = ["Overview", "Algorithm Explorer", "Performance Comparison",
                  "AI Prediction", "Algorithm Recommendation", "Model Performance",
                  "Predictability Gap", "About / Methodology"]

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def new_app(page):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    return at


print("Test 1 - existing pages unchanged (all render, no exception)")
for p in EXISTING_PAGES:
    at = new_app(p)
    check(f"page renders: {p}", not at.exception, [e.value for e in at.exception])
at = new_app("AI Prediction")
at.button[0].click().run()
check("existing AI Prediction still predicts", any("Predicted Execution Time" in m.value for m in at.markdown))
at = new_app("Algorithm Recommendation")
at.button[0].click().run()
check("existing Algorithm Recommendation still recommends", not at.exception and any("Model recommendation" in m.value for m in at.markdown))

print("Test 1b - Personalized page has ONLY the own-data mode")
at = new_app("Personalized Prediction")
check("page renders", not at.exception, [e.value for e in at.exception])
check("old Mode selector is gone", not any("Mode" == r.label for r in at.radio))
check("old profile/default inputs are gone", not at.get("number_input") or all(
    n.key not in ("pz_size", "pz_presort", "pz_dup", "pz_vr") for n in at.get("number_input")))
check("own-data input is shown", any(t.key == "pz_data_text" for t in at.text_area))

print("Test 7 - own data: parsing, validation, scenario detection")
check("parses commas/spaces/newlines", parse_user_numbers("3, 1 2\n9;4")[0] == [3, 1, 2, 9, 4])
check("rejects text", len(parse_user_numbers("1, two, 3")[1]) == 1)
check("rejects decimals", len(parse_user_numbers("1.5, 2")[1]) == 1)
check("rejects empty", len(parse_user_numbers("   ")[1]) == 1)
check("accepts short lists (1-9 numbers)", validate_user_data([1, 2, 3], "Sort") == [] and validate_user_data([7], "Sort") == [])
check("accepts up to 5000 numbers", validate_user_data(list(range(5000)), "Sort") == [])
check("rejects more than 5000 numbers", len(validate_user_data(list(range(5001)), "Sort")) > 0)
check("rejects negatives", len(validate_user_data([-1] + list(range(20)), "Sort")) > 0)
check("search requires target", len(validate_user_data(list(range(20)), "Search", None)) > 0)
check("sorted detected", derive_sorting_condition(1.0, 0.0) == "sorted")
check("reverse detected", derive_sorting_condition(0.0, 0.0) == "reverse_sorted")
check("duplicates detected", derive_sorting_condition(0.5, 0.9) == "duplicates")
check("nearly sorted detected", derive_sorting_condition(0.93, 0.0) == "nearly_sorted")
check("random detected", derive_sorting_condition(0.5, 0.0) == "random")
check("not_found case", derive_search_case([1, 2, 3, 4, 5], 99) == "not_found")
check("best_case = middle element", derive_search_case([1, 2, 3, 4, 5], 3) == "best_case")
check("worst_case = last element", derive_search_case([1, 2, 3, 4, 5], 5) == "worst_case")
check("duplicates_present case", derive_search_case([1, 2, 2, 4, 5, 6, 7], 2) == "duplicates_present")


def own_data(task, text, target=None):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Personalized Prediction").run()
    at.radio(key="pz_data_task").set_value(task).run()
    at.text_area(key="pz_data_text").set_value(text)
    if target is not None:
        at.number_input(key="pz_data_target").set_value(target)
    at.run()
    at.button(key="pz_data_run").click().run()
    return at


import random
random.seed(7)
SORTING = {"bubble_sort", "selection_sort", "insertion_sort", "merge_sort", "quick_sort",
           "heap_sort", "radix_sort"}
SEARCHING = {"linear_search", "binary_search", "interpolation_search", "exponential_search",
             "fibonacci_search", "hashing_search"}

print("Test 8 - own data: sort really sorts, ML recommendation depends on the data")
rnd = [random.randint(0, 99999) for _ in range(800)]
at = own_data("Sort", ", ".join(map(str, rnd)))
a = at.session_state["pz_data_last"] if "pz_data_last" in at.session_state else None
check("no exception", not at.exception, [e.value for e in at.exception])
check("output equals sorted(input)", a and a["chosen_run"]["result"] == sorted(rnd))
check("recommended algorithm is a sorting algorithm", a and a["clf_pick"] in SORTING, a and a["clf_pick"])
check("random data detected as random", a and a["condition"] == "random", a and a["condition"])
asc = sorted(rnd)
at2 = own_data("Sort", ", ".join(map(str, asc)))
a2 = at2.session_state["pz_data_last"]
check("sorted input detected as sorted (presortedness 1.0)", a2["condition"] == "sorted" and a2["presortedness"] == 1.0)
check("all 7 algorithms benchmarked on user's data", set(a2["benchmark"]["algorithm"]) == SORTING)
check("every algorithm produced a real measured time", (a2["benchmark"]["measured_ms"] > 0).all())
rev = ", ".join(map(str, asc[::-1]))
a3 = own_data("Sort", rev).session_state["pz_data_last"]
check("reverse input detected as reverse_sorted", a3["condition"] == "reverse_sorted")
check("different data -> different ML features/prediction",
      a["presortedness"] != a2["presortedness"] and a["reg_ranking"] != a2["reg_ranking"])

print("Test 9 - own data: search finds / does not find")
base = ", ".join(map(str, rnd[:300]))
f = own_data("Search", base, rnd[17]).session_state["pz_data_last"]
check("found target", f["chosen_run"]["result"]["found"] is True and f["verified"])
check("recommended algorithm is a searching algorithm", f["clf_pick"] in SEARCHING, f["clf_pick"])
nf = own_data("Search", base, 123456).session_state["pz_data_last"]
check("missing target -> not_found, found False", nf["condition"] == "not_found" and nf["chosen_run"]["result"]["found"] is False)
check("all 6 search algorithms run", set(nf["benchmark"]["algorithm"]) == SEARCHING)

print("Test 10 - own data: invalid input is rejected without crashing")
for label, text, task, tgt in [("text in list", "1, 2, abc", "Sort", None),
                                                              ("negative number", "-5, " + ", ".join(map(str, range(20))), "Sort", None),
                               ("empty", "", "Sort", None),
                               ("search without target", base, "Search", None)]:
    at = own_data(task, text, tgt)
    check(f"rejected: {label}", len(at.error) > 0 and not at.exception and "pz_data_last" not in at.session_state)

print("Test 10b - own data: ANY list works, including the exact input from the bug report")
cases = [("1,2,3,4,4,5,6,7,8", "Sort"), ("5", "Sort"), ("9,3", "Sort"), ("7,7,7,7", "Sort"),
         ("0,0,0", "Sort"), ("100000,5,99,5", "Sort"), ("1,2,3,4,4,5,6,7,8", "Search")]
for text, task in cases:
    at = own_data(task, text, 4 if task == "Search" else None)
    a = at.session_state["pz_data_last"] if "pz_data_last" in at.session_state else None
    nums = [int(v) for v in re.split(r"[,\s]+", text)]
    ok = a is not None and not at.exception and a["verified"] and (
        a["chosen_run"]["result"] == sorted(nums) if task == "Sort" else a["chosen_run"]["result"]["found"])
    check(f"{task} works on '{text}'", ok, [e.value for e in at.exception] or [e.value for e in at.error])
big = own_data("Sort", ", ".join(str((i * 7919) % 100003) for i in range(3000)))
check("3000 numbers works with an informational note", "pz_data_last" in big.session_state and len(big.info) > 0)

print("Test 11 - own data: reset")
at = own_data("Sort", base)
at.button(key="pz_data_reset").click().run()
check("reset clears own-data result", "pz_data_last" not in at.session_state)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)