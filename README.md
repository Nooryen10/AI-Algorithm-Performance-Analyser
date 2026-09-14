# AI-Based Algorithm Performance Analyser

### Using Sorting, Searching & Complexity Analysis

An AI-based system that benchmarks classical sorting and searching algorithms at scale, then uses machine learning to go beyond static theoretical complexity analysis — predicting execution time and recommending the best algorithm for a given input profile.

---

## Project Overview

Instead of only stating that, for example, Quick Sort is `O(n log n)` on average, this system **empirically measures** how each of 13 algorithms (7 sorting, 6 searching) actually behaves across thousands of controlled test cases, and trains ML models that:

1. **Predict** an algorithm's execution time for a given input profile (regression)
2. **Recommend** the best algorithm to use for that profile (classification)

The project also investigates a specific research gap: most algorithm-selection literature reports only classification accuracy for "best algorithm" recommendations, without examining _why_ it's harder to predict than raw execution time. This project explicitly compares regression vs. classification performance and documents the resulting **predictability gap** as a discussed finding.

---

## Algorithm Scope (13 Total)

| Sorting (7)    | Searching (6)                             |
| -------------- | ----------------------------------------- |
| Bubble Sort    | Linear Search                             |
| Selection Sort | Binary Search                             |
| Insertion Sort | Interpolation Search                      |
| Merge Sort     | Exponential Search                        |
| Quick Sort     | Fibonacci Search                          |
| Heap Sort      | Hashing (custom, with collision tracking) |
| Radix Sort     |                                           |

---

## Dataset

| Dataset   | Size Range        | Variations                                                                        | Trials | Raw Rows | Cleaned Rows |
| --------- | ----------------- | --------------------------------------------------------------------------------- | ------ | -------- | ------------ |
| Sorting   | 10–2000 (step 10) | 5 data types (random, sorted, reverse-sorted, nearly sorted, duplicates)          | 5      | 35,000   | 34,992       |
| Searching | 10–2000 (step 10) | 5 case types (best-case, average-case, worst-case, not-found, duplicates-present) | 5      | 30,000   | 29,788       |

**Combined: ~64,780 cleaned rows.**

Each sorting row includes measured metrics (comparisons, swaps, time, memory) plus dataset features (presortedness score, duplicate ratio, value range). Each searching row includes probes, time, memory, and (for Hashing) load factor and collision count.

---

## Project Structure

```
AI-Algorithm-Performance-Analyser/
├── data/
│   ├── raw/                  # Unprocessed experiment output
│   └── processed/            # Cleaned datasets used for ML
├── src/
│   ├── algorithms/
│   │   ├── sorting/          # 7 instrumented sorting algorithms
│   │   └── searching/        # 6 instrumented searching algorithms
│   ├── instrumentation/      # Metrics tracking + @instrument decorator
│   ├── data_generation/      # Generators, feature extraction, experiment drivers, cleaning
│   ├── complexity/           # Theoretical Big-O reference table
│   ├── ml/                   # (Member B) Regression + classification models
│   └── dashboard/            # (Member B) Streamlit/Flask interactive dashboard
├── models/                   # Saved trained models
├── reports/                  # Final report, figures, literature review
├── presentation/             # Slide deck
├── tests/                    # Unit tests
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

## Running the Data Generation Pipeline

All commands are run as modules from the project root (required for internal relative imports to work):

```bash
# Generate raw datasets
python -m src.data_generation.run_sorting_experiments
python -m src.data_generation.run_searching_experiments

# Clean the datasets
python -m src.data_generation.data_cleaning

# View theoretical complexity table
python -m src.complexity.theoretical_complexity
```

> **Note:** Bubble/Selection/Insertion Sort are O(n²), so the full sorting sweep (sizes up to 2000) can take several minutes depending on your machine.

---

## Key Engineering Decisions

- **Randomized pivot in Quick Sort** — the initial last-element-pivot implementation caused stack overflow (`maximum recursion depth exceeded`) on already-sorted input, since it degrades to O(n) recursion depth. Fixed via randomized pivot selection + recursing on the smaller partition (guarantees O(log n) depth).
- **Custom hash table with chaining** (not Python's built-in `dict`) for Hashing Search, deliberately undersized relative to input, to surface genuine `load_factor` and `collision_count` features for the ML model — rather than hiding collision behavior inside `dict`'s internals.
- **O(n log n) inversion counting** (merge-sort-based) for the presortedness feature, instead of a naive O(n²) pairwise check — keeps feature extraction efficient even at input size 2000.
- **Median-ratio outlier filtering** instead of IQR for data cleaning, since each (algorithm, size, type) group only has 5 trials — IQR's quartile calculations are unreliable at that sample size and over-remove valid variance.
- **`sys.getsizeof`-based memory estimation** instead of `tracemalloc`, since per-allocation tracing overhead made full-scale data generation prohibitively slow (original estimate: ~10 hours; optimized: well under an hour).

---

## Team & Work Allocation

| Member      | Responsibilities                                                                                                                       |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Nooryen** | Full algorithm engine (13 algorithms), data generation pipeline, dataset cleaning, theoretical complexity documentation                |
| **Sudipta** | Feature engineering, regression + classification models, evaluation & predictability-gap analysis, interactive dashboard, final report |

---

## Status

- [x] Algorithm engine (13 algorithms, instrumented, tested)
- [x] Dataset generation (~65,000 rows)
- [x] Data cleaning
- [x] Theoretical complexity table
- [x] ML models (regression + classification)
- [x] Predictability-gap analysis
- [x] Interactive dashboard
- [x] Final report
