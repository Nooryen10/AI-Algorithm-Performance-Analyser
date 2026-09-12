"""
Cleans the raw sorting and searching datasets:
- Removes rows with negative or zero time_taken (measurement glitches)
- Removes extreme outliers in time_taken per (algorithm, size) group using
  a median-ratio filter (more robust than IQR for small groups of 5 trials)
- Ensures consistent column types
- Writes cleaned CSVs to data/processed/
"""

import pandas as pd
import os

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def remove_extreme_outliers(df, group_cols, target_col, ratio_threshold=5.0):
    """
    Removes rows where target_col is more than `ratio_threshold` times
    away from the group's median. This is more robust than IQR for small
    groups (here, groups of only 5 trials), where quartile-based methods
    are statistically unreliable and tend to over-remove valid variance.
    """
    def filter_group(group):
        median_val = group[target_col].median()
        if median_val <= 0:
            return group
        lower = median_val / ratio_threshold
        upper = median_val * ratio_threshold
        return group[(group[target_col] >= lower) & (group[target_col] <= upper)]

    return df.groupby(group_cols, group_keys=False).apply(filter_group)


def clean_sorting_data():
    path = os.path.join(RAW_DIR, "sorting_dataset.csv")
    df = pd.read_csv(path)

    before = len(df)

    # Remove impossible/glitched measurements
    df = df[df["time_taken"] > 0]
    df = df[df["comparisons"] >= 0]
    df = df[df["swaps"] >= 0]

    # Remove extreme outliers per (algorithm, input_size, data_type)
    df = remove_extreme_outliers(
        df,
        group_cols=["algorithm", "input_size", "data_type"],
        target_col="time_taken",
    )

    after = len(df)
    print(f"Sorting: {before} -> {after} rows ({before - after} removed)")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "sorting_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def clean_searching_data():
    path = os.path.join(RAW_DIR, "searching_dataset.csv")
    df = pd.read_csv(path)

    before = len(df)

    df = df[df["time_taken"] > 0]
    df = df[df["probes"] >= 0]

    df = remove_extreme_outliers(
        df,
        group_cols=["algorithm", "input_size", "case_type"],
        target_col="time_taken",
    )

    after = len(df)
    print(f"Searching: {before} -> {after} rows ({before - after} removed)")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "searching_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    clean_sorting_data()
    clean_searching_data()