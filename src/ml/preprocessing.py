"""
Reusable feature engineering and preprocessing logic for the ML pipeline.

This module is the single source of truth for:
- Merging the sorting and searching datasets into a unified schema
- Handling domain-specific (structural) missing values
- Building the sklearn ColumnTransformer used for both regression and
  classification inputs
- Producing a leakage-safe, grouped train/test split

Both the training notebooks and the dashboard (src/dashboard/app.py) should
import from here rather than redefining this logic, to avoid the notebook
and the dashboard silently drifting out of sync.
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

# Columns present in both domains after renaming (see build_unified_dataset)
COMMON_CATEGORICAL = ["domain", "algorithm", "input_condition"]
COMMON_NUMERICAL = ["input_size"]

# Sorting-only measured features (NaN for searching rows before fillna)
SORTING_ONLY_NUMERICAL = [
    "comparisons",
    "swaps_or_shifts",
    "presortedness_score",
    "duplicate_ratio",
    "value_range",
]

# Searching-only measured features.
# NOTE: load_factor is deliberately excluded -- it is constant (2.0) across
# every hashing_search row in the current dataset (zero variance, verified in
# notebooks/01_data_exploration.ipynb), so it carries no predictive signal.
# It is still written into the unified dataset for documentation purposes,
# just not used as a model input feature.
SEARCHING_ONLY_NUMERICAL = ["probes", "collision_count"]

REGRESSION_TARGET = "time_taken"

CLASSIFICATION_TARGET = "best_algorithm"
CLASSIFICATION_NUMERICAL = ["input_size", "presortedness_score", "duplicate_ratio", "value_range"]
CLASSIFICATION_CATEGORICAL = ["domain", "input_condition"]

# Threshold (data-driven, see notebooks/04_classification_training.ipynb) below
# which the margin between the best and second-best algorithm is treated as a
# "close call" -- i.e. plausibly unstable to trial-level timing noise.
CLOSE_CALL_MARGIN_THRESHOLD = 0.10

NUMERICAL_FEATURES = COMMON_NUMERICAL + SORTING_ONLY_NUMERICAL + SEARCHING_ONLY_NUMERICAL
CATEGORICAL_FEATURES = COMMON_CATEGORICAL
STRUCTURAL_NUMERICAL_COLS = SORTING_ONLY_NUMERICAL + SEARCHING_ONLY_NUMERICAL


# ---------------------------------------------------------------------------
# Dataset loading and merging
# ---------------------------------------------------------------------------

def load_cleaned_datasets(data_dir="data/processed"):
    """Load the two per-domain cleaned CSVs produced by data_cleaning.py."""
    sorting_df = pd.read_csv(os.path.join(data_dir, "sorting_clean.csv"))
    searching_df = pd.read_csv(os.path.join(data_dir, "searching_clean.csv"))
    return sorting_df, searching_df


def build_unified_dataset(sorting_df: pd.DataFrame, searching_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the sorting and searching datasets into one unified schema.

    - Renames domain-specific columns to shared unified names
      (data_type/case_type -> input_condition, swaps -> swaps_or_shifts).
    - Adds a `domain` column.
    - Adds an explicit `is_hashing` flag, since load_factor/collision_count
      are only meaningful for hashing_search.
    - Fills structural NaNs (features not applicable to a given domain) with
      0 -- this is safe because `domain` and `algorithm` are themselves
      features, so the model can learn that a 0 here is structural, not a
      real measured zero.
    """
    sorting = sorting_df.copy()
    sorting["domain"] = "sorting"
    sorting = sorting.rename(columns={
        "data_type": "input_condition",
        "swaps": "swaps_or_shifts",
    })

    searching = searching_df.copy()
    searching["domain"] = "searching"
    searching = searching.rename(columns={
        "case_type": "input_condition",
    })

    unified_df = pd.concat([sorting, searching], ignore_index=True, sort=False)

    unified_df["is_hashing"] = (unified_df["algorithm"] == "hashing_search").astype(int)

    unified_df[STRUCTURAL_NUMERICAL_COLS] = unified_df[STRUCTURAL_NUMERICAL_COLS].fillna(0)

    unified_df["config_id"] = (
        unified_df["domain"] + "_" +
        unified_df["algorithm"] + "_" +
        unified_df["input_size"].astype(str) + "_" +
        unified_df["input_condition"]
    )

    return unified_df


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used to prepare features for both the
    regression and classification models: StandardScaler on numeric
    features, OneHotEncoder on categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_feature_target_split(unified_df: pd.DataFrame, target_col: str = REGRESSION_TARGET):
    """Return (X, y) for the given target column using the standard feature set."""
    X = unified_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = unified_df[target_col]
    return X, y


# ---------------------------------------------------------------------------
# Leakage-safe train/test split
# ---------------------------------------------------------------------------

def grouped_train_test_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                              test_size: float = 0.2, random_state: int = 42):
    """
    Split into train/test grouped by `groups` (config_id), so that repeated
    trials of the same (domain, algorithm, input_size, input_condition)
    configuration never end up split across both train and test -- which
    would let the model partially memorize a configuration's typical timing
    rather than generalize to genuinely unseen configurations.

    Returns X_train, X_test, y_train, y_test, and raises if any config_id
    leaks across the split.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = len(train_groups & test_groups)
    if overlap != 0:
        raise ValueError(
            f"Data leakage detected: {overlap} config_id(s) appear in both "
            "train and test sets."
        )

    return X_train, X_test, y_train, y_test
# ---------------------------------------------------------------------------
# Classification target derivation
# ---------------------------------------------------------------------------

PROFILE_COLS = ["domain", "input_size", "input_condition"]


def derive_profile_labels(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the best/second-best algorithm per unique input profile
    (domain, input_size, input_condition), along with the absolute and
    relative timing margin between them.

    This is the classification target derivation logic: `best_algorithm`
    does not exist as a raw column, it is computed here from measured
    execution times, averaged across trials.
    """
    profile_algo_times = (
        unified_df
        .groupby(PROFILE_COLS + ["algorithm"])["time_taken"]
        .mean()
        .reset_index()
    )

    def get_best_and_second(group):
        sorted_group = group.sort_values("time_taken")
        best_algo = sorted_group.iloc[0]["algorithm"]
        best_time = sorted_group.iloc[0]["time_taken"]
        second_algo = sorted_group.iloc[1]["algorithm"] if len(sorted_group) > 1 else None
        second_time = sorted_group.iloc[1]["time_taken"] if len(sorted_group) > 1 else None
        return pd.Series({
            "best_algorithm": best_algo,
            "best_time": best_time,
            "second_algorithm": second_algo,
            "second_time": second_time,
        })

    profile_labels = (
        profile_algo_times
        .groupby(PROFILE_COLS)
        .apply(get_best_and_second)
        .reset_index()
    )

    profile_labels["absolute_margin"] = profile_labels["second_time"] - profile_labels["best_time"]
    profile_labels["relative_margin"] = profile_labels["absolute_margin"] / profile_labels["best_time"]
    profile_labels["is_close_call"] = profile_labels["relative_margin"] <= CLOSE_CALL_MARGIN_THRESHOLD

    return profile_labels


def build_classification_dataset(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the profile-level classification dataset: one row per unique
    input profile, with the derived `best_algorithm` label and profile-level
    (array-property) numerical features.

    Note: `algorithm` is intentionally excluded as a feature here (it would
    leak the answer), and `collision_count`/`load_factor` are excluded since
    they are algorithm-specific outcomes (only meaningful for hashing_search),
    not properties of the input profile itself.
    """
    profile_labels = derive_profile_labels(unified_df)

    profile_features = (
        unified_df
        .groupby(PROFILE_COLS)[["presortedness_score", "duplicate_ratio", "value_range"]]
        .mean()
        .reset_index()
    )

    classification_df = profile_labels[PROFILE_COLS + ["best_algorithm", "is_close_call",
                                                          "relative_margin"]].merge(
        profile_features, on=PROFILE_COLS, how="left"
    )

    return classification_df


def build_classification_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the classification feature set."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CLASSIFICATION_NUMERICAL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CLASSIFICATION_CATEGORICAL),
        ]
    )


def get_classification_feature_target(classification_df: pd.DataFrame):
    """Return (X, y) for the classification task."""
    X = classification_df[CLASSIFICATION_NUMERICAL + CLASSIFICATION_CATEGORICAL]
    y = classification_df[CLASSIFICATION_TARGET]
    return X, y