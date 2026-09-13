"""
AI-Based Algorithm Performance Analyser — Interactive Dashboard

Loads pre-trained models and pre-computed datasets/metrics; does NOT retrain
anything at runtime. See notebooks/03-06 for training and evaluation.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ml.preprocessing import (
    load_cleaned_datasets,
    build_unified_dataset,
    build_classification_dataset,
    derive_profile_labels,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSIFICATION_NUMERICAL,
    CLASSIFICATION_CATEGORICAL,
)
from src.complexity.theoretical_complexity import (
    SORTING_COMPLEXITY,
    SEARCHING_COMPLEXITY,
    get_complexity,
)

st.set_page_config(
    page_title="AI-Based Algorithm Performance Analyser",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "visualizations")

TEAL = "#4FD1C5"
AMBER = "#F5A623"

SORTING_ALGOS = list(SORTING_COMPLEXITY.keys())
SEARCHING_ALGOS = list(SEARCHING_COMPLEXITY.keys())


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #12151C; color: #E6E8EB; }

        section[data-testid="stSidebar"] {
            background-color: #0D0F15;
            border-right: 1px solid #2A2F3A;
        }
        section[data-testid="stSidebar"] h1 {
            font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; color: #E6E8EB;
        }
        h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; color: #F2F3F5; }
        p, li, label, .stMarkdown { color: #C4C9D1; }
        code, .algo-mono { font-family: 'JetBrains Mono', monospace; }
        hr { border-color: #2A2F3A; }

        .stat-block { border-top: 1px solid #2A2F3A; padding-top: 0.75rem; }
        .stat-label { font-size: 0.8rem; color: #8B93A1; margin-bottom: 0.25rem; }
        .stat-value {
            font-size: 2.1rem; font-weight: 700; color: #F2F3F5;
            font-family: 'JetBrains Mono', monospace;
        }
        .stat-value.small { font-size: 1.4rem; }
        .stat-value.teal { color: #4FD1C5; }
        .stat-value.amber { color: #F5A623; }

        .callout {
            border-left: 3px solid #4FD1C5;
            background-color: #171B24;
            padding: 0.9rem 1.1rem;
            font-size: 0.95rem;
            color: #C4C9D1;
            margin-top: 1rem;
        }
        .callout.warn { border-left-color: #F5A623; }

        .rec-card {
            border: 1px solid #2A2F3A;
            background-color: #171B24;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.6rem;
        }
        .rec-rank { color: #8B93A1; font-size: 0.8rem; }
        .rec-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem; font-weight: 700; color: #F2F3F5;
        }
        .rec-name.primary { color: #4FD1C5; }
        .rec-time { color: #8B93A1; font-size: 0.9rem; font-family: 'JetBrains Mono', monospace; }

        .stButton > button {
            background-color: #1A1E27; color: #E6E8EB;
            border: 1px solid #2A2F3A; border-radius: 4px;
        }
        .stButton > button:hover { border-color: #4FD1C5; color: #4FD1C5; }
        div[data-baseweb="radio"] label { color: #C4C9D1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_block(label, value, accent=None, small=False):
    classes = "stat-value"
    if accent:
        classes += f" {accent}"
    if small:
        classes += " small"
    st.markdown(
        f'<div class="stat-block"><div class="stat-label">{label}</div>'
        f'<div class="{classes}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text, warn=False):
    cls = "callout warn" if warn else "callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


inject_theme()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def get_unified_dataset():
    sorting_df, searching_df = load_cleaned_datasets(DATA_DIR)
    return build_unified_dataset(sorting_df, searching_df)


@st.cache_data
def get_classification_dataset():
    return build_classification_dataset(get_unified_dataset())


@st.cache_data
def get_profile_labels():
    return derive_profile_labels(get_unified_dataset())


@st.cache_data
def get_evaluation_summary():
    with open(os.path.join(MODELS_DIR, "evaluation_summary.json")) as f:
        return json.load(f)


@st.cache_resource
def get_regression_model():
    return joblib.load(os.path.join(MODELS_DIR, "regression_model.pkl"))


@st.cache_resource
def get_classification_model():
    return joblib.load(os.path.join(MODELS_DIR, "classification_model.pkl"))


def lookup_typical_measurements(unified_df, algorithm, domain, input_size, input_condition):
    """
    For a hypothetical (not-yet-run) profile, look up the historical average
    of algorithm-run-time measurements (comparisons/swaps/probes/collisions)
    for the nearest matching configuration. These cannot be supplied by a
    user ahead of time (they are outputs of running the algorithm), so we
    use real historical averages as a stand-in rather than asking the user
    to guess them or fabricating values.
    """
    subset = unified_df[
        (unified_df["algorithm"] == algorithm) &
        (unified_df["domain"] == domain) &
        (unified_df["input_condition"] == input_condition)
    ]
    if subset.empty:
        subset = unified_df[(unified_df["algorithm"] == algorithm) & (unified_df["domain"] == domain)]

    if subset.empty:
        return {"comparisons": 0, "swaps_or_shifts": 0, "probes": 0, "collision_count": 0}

    subset = subset.copy()
    subset["size_diff"] = (subset["input_size"] - input_size).abs()
    nearest_size = subset.loc[subset["size_diff"] <= subset["size_diff"].min() + 1e-9]

    return {
        "comparisons": nearest_size["comparisons"].mean(),
        "swaps_or_shifts": nearest_size["swaps_or_shifts"].mean(),
        "probes": nearest_size["probes"].mean(),
        "collision_count": nearest_size["collision_count"].mean(),
    }


def predict_execution_time(algorithm, domain, input_size, input_condition,
                            presortedness_score, duplicate_ratio, value_range):
    unified_df = get_unified_dataset()
    typical = lookup_typical_measurements(unified_df, algorithm, domain, input_size, input_condition)

    row = pd.DataFrame([{
        "input_size": input_size,
        "comparisons": typical["comparisons"],
        "swaps_or_shifts": typical["swaps_or_shifts"],
        "presortedness_score": presortedness_score,
        "duplicate_ratio": duplicate_ratio,
        "value_range": value_range,
        "probes": typical["probes"],
        "collision_count": typical["collision_count"],
        "domain": domain,
        "algorithm": algorithm,
        "input_condition": input_condition,
    }])[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    model = get_regression_model()
    return model.predict(row)[0]


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Overview",
    "Algorithm Explorer",
    "Performance Comparison",
    "AI Prediction",
    "Algorithm Recommendation",
    "Model Performance",
    "Predictability Gap",
    "About / Methodology",
]

st.sidebar.markdown("# Algorithm Analyser")
st.sidebar.markdown(
    '<p style="color:#8B93A1; font-size:0.85rem; margin-top:-0.5rem;">'
    "Sorting, searching & complexity analysis, backed by ML.</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def render_overview():
    st.title("AI-Based Algorithm Performance Analyser")
    st.markdown(
        "Benchmarking classical sorting and searching algorithms, then using "
        "machine learning to predict execution time and recommend the best "
        "algorithm for a given input profile."
    )

    unified_df = get_unified_dataset()
    evaluation_summary = get_evaluation_summary()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_block("Sorting Algorithms", unified_df[unified_df["domain"] == "sorting"]["algorithm"].nunique())
    with col2:
        stat_block("Searching Algorithms", unified_df[unified_df["domain"] == "searching"]["algorithm"].nunique())
    with col3:
        stat_block("Benchmark Rows", f"{len(unified_df):,}")
    with col4:
        stat_block("ML Tasks", "2")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Status")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        stat_block("Regression R² (execution time)", f"{evaluation_summary['regression']['r2']:.3f}", accent="teal")
    with mcol2:
        stat_block("Classification Macro F1 (best algorithm)", f"{evaluation_summary['classification']['f1_macro']:.3f}", accent="amber")

    callout(
        "Notice the gap between these two scores — this is the project's central "
        "research finding, explored in depth on the <b>Predictability Gap</b> page."
    )


# ---------------------------------------------------------------------------
# Page: Algorithm Explorer
# ---------------------------------------------------------------------------

def render_algorithm_explorer():
    st.title("Algorithm Explorer")
    st.markdown(
        "Select an algorithm and input profile to see its theoretical complexity "
        "alongside actual measured performance from the benchmark dataset."
    )

    unified_df = get_unified_dataset()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("Domain", ["Sorting", "Searching"])
    domain_key = domain.lower()

    if domain_key == "sorting":
        algo_options = SORTING_ALGOS
    else:
        algo_options = SEARCHING_ALGOS

    condition_options = sorted(unified_df[unified_df["domain"] == domain_key]["input_condition"].unique())
    condition_label = "Data Type" if domain_key == "sorting" else "Case Type"

    with col2:
        algorithm = st.selectbox("Algorithm", algo_options, format_func=lambda x: x.replace("_", " ").title())

    col3, col4 = st.columns(2)
    with col3:
        input_size = st.slider("Input Size", min_value=10, max_value=2000, value=1000, step=10)
    with col4:
        condition = st.selectbox(condition_label, condition_options, format_func=lambda x: x.replace("_", " ").title())

    st.markdown("---")
    complexity = get_complexity(algorithm)

    st.subheader("Theoretical Complexity")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    with tcol1:
        stat_block("Best Case", complexity["best"], accent="teal")
    with tcol2:
        stat_block("Average Case", complexity["average"])
    with tcol3:
        stat_block("Worst Case", complexity["worst"], accent="amber")
    with tcol4:
        stat_block("Space", complexity["space"])

    st.markdown(f"<p style='margin-top:1rem; color:#8B93A1;'>{complexity['notes']}</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Measured Performance")

    matching_rows = unified_df[
        (unified_df["algorithm"] == algorithm) &
        (unified_df["input_size"] == input_size) &
        (unified_df["input_condition"] == condition)
    ]

    if matching_rows.empty:
        st.warning("No measured data found for this exact combination (may have been removed as an outlier during cleaning).")
    else:
        mean_row = matching_rows.mean(numeric_only=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            stat_block("Mean Execution Time", f"{mean_row['time_taken']*1000:.4f} ms")
        with mcol2:
            stat_block("Memory Used", f"{int(mean_row['memory_used']):,} bytes")
        with mcol3:
            stat_block("Trials Averaged", int(matching_rows.shape[0]))

        st.markdown("<br>", unsafe_allow_html=True)
        if domain_key == "sorting":
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                stat_block("Comparisons", f"{mean_row['comparisons']:.0f}")
            with scol2:
                stat_block("Swaps / Shifts", f"{mean_row['swaps_or_shifts']:.0f}")
            with scol3:
                stat_block("Presortedness", f"{mean_row['presortedness_score']:.3f}")
            with scol4:
                stat_block("Duplicate Ratio", f"{mean_row['duplicate_ratio']:.3f}")
        else:
            scol1, scol2 = st.columns(2)
            with scol1:
                stat_block("Probes", f"{mean_row['probes']:.0f}")
            with scol2:
                stat_block("Collisions", f"{mean_row['collision_count']:.0f}" if algorithm == "hashing_search" else "N/A")

    if algorithm == "hashing_search":
        callout(
            "Theoretical average-case is O(1), but measured execution time for "
            "<code>hashing_search</code> grows with input size in this dataset — "
            "the hash table is deliberately fixed-size, so collisions increase as "
            "more items are packed in. See <b>Predictability Gap</b> for more on "
            "theory-vs-practice gaps like this."
        )


# ---------------------------------------------------------------------------
# Page: Performance Comparison
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#12151C",
    plot_bgcolor="#12151C",
    font=dict(color="#C4C9D1", family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#2A2F3A"),
    yaxis=dict(gridcolor="#2A2F3A"),
)


def render_performance_comparison():
    st.title("Performance Comparison")
    st.markdown("Interactively compare measured performance across algorithms, data types, and input sizes.")

    unified_df = get_unified_dataset()

    st.markdown("<br>", unsafe_allow_html=True)
    domain = st.radio("Domain", ["Sorting", "Searching"], horizontal=True)
    domain_key = domain.lower()
    domain_df = unified_df[unified_df["domain"] == domain_key]

    algo_options = sorted(domain_df["algorithm"].unique())
    condition_options = sorted(domain_df["input_condition"].unique())
    condition_label = "Data Type" if domain_key == "sorting" else "Case Type"

    col1, col2 = st.columns(2)
    with col1:
        selected_algos = st.multiselect(
            "Algorithms", algo_options, default=algo_options,
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with col2:
        selected_condition = st.selectbox(
            condition_label, condition_options, format_func=lambda x: x.replace("_", " ").title()
        )

    size_range = st.slider("Input Size Range", 10, 2000, (10, 2000), step=10)

    filtered = domain_df[
        (domain_df["algorithm"].isin(selected_algos)) &
        (domain_df["input_condition"] == selected_condition) &
        (domain_df["input_size"].between(size_range[0], size_range[1]))
    ]

    if filtered.empty or not selected_algos:
        st.warning("No data matches the current filters.")
        return

    st.markdown("---")
    st.subheader("Execution Time vs Input Size")
    grouped = filtered.groupby(["algorithm", "input_size"])["time_taken"].mean().reset_index()
    fig1 = px.line(grouped, x="input_size", y="time_taken", color="algorithm",
                    labels={"input_size": "Input Size", "time_taken": "Mean Execution Time (s)", "algorithm": "Algorithm"})
    fig1.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

    metric_col = "comparisons" if domain_key == "sorting" else "probes"
    metric_label = "Comparisons" if domain_key == "sorting" else "Probes"

    st.subheader(f"{metric_label} vs Input Size")
    grouped2 = filtered.groupby(["algorithm", "input_size"])[metric_col].mean().reset_index()
    fig2 = px.line(grouped2, x="input_size", y=metric_col, color="algorithm",
                    labels={"input_size": "Input Size", metric_col: f"Mean {metric_label}", "algorithm": "Algorithm"})
    fig2.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Memory Used vs Input Size")
    grouped3 = filtered.groupby(["algorithm", "input_size"])["memory_used"].mean().reset_index()
    fig3 = px.line(grouped3, x="input_size", y="memory_used", color="algorithm",
                    labels={"input_size": "Input Size", "memory_used": "Mean Memory (bytes)", "algorithm": "Algorithm"})
    fig3.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Algorithm Ranking (mean execution time, current filters)")
    ranking = filtered.groupby("algorithm")["time_taken"].mean().sort_values().reset_index()
    fig4 = px.bar(ranking, x="time_taken", y="algorithm", orientation="h",
                   labels={"time_taken": "Mean Execution Time (s)", "algorithm": "Algorithm"},
                   color_discrete_sequence=[TEAL])
    fig4.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: AI Prediction
# ---------------------------------------------------------------------------

def render_ai_prediction():
    st.title("AI Prediction")
    st.markdown("Enter an input profile to predict its execution time using the trained regression model.")

    unified_df = get_unified_dataset()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("Domain", ["Sorting", "Searching"], key="pred_domain")
    domain_key = domain.lower()
    algo_options = SORTING_ALGOS if domain_key == "sorting" else SEARCHING_ALGOS
    condition_options = sorted(unified_df[unified_df["domain"] == domain_key]["input_condition"].unique())
    condition_label = "Data Type" if domain_key == "sorting" else "Case Type"

    with col2:
        algorithm = st.selectbox("Algorithm", algo_options, format_func=lambda x: x.replace("_", " ").title(), key="pred_algo")

    col3, col4 = st.columns(2)
    with col3:
        input_size = st.number_input("Input Size", min_value=1, max_value=2000, value=1000, step=10, key="pred_size")
        if input_size <= 0:
            st.error("Input size must be greater than 0.")
            return
    with col4:
        condition = st.selectbox(condition_label, condition_options, format_func=lambda x: x.replace("_", " ").title(), key="pred_cond")

    if domain_key == "sorting":
        c5, c6, c7 = st.columns(3)
        with c5:
            presortedness = st.slider("Presortedness", 0.0, 1.0, 0.5, key="pred_presort")
        with c6:
            duplicate_ratio = st.slider("Duplicate Ratio", 0.0, 1.0, 0.0, key="pred_dup")
        with c7:
            value_range = st.number_input("Value Range", min_value=1, value=90000, key="pred_vr")
    else:
        presortedness, duplicate_ratio, value_range = 0.0, 0.0, 0.0
        st.caption("Presortedness / duplicate ratio / value range are sorting-specific and not applicable here.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Execution Time", type="primary"):
        predicted_time = predict_execution_time(
            algorithm, domain_key, input_size, condition,
            presortedness, duplicate_ratio, value_range,
        )
        st.markdown("---")
        stat_block("Predicted Execution Time", f"{predicted_time*1000:.4f} ms", accent="teal")
        callout(
            "This prediction uses the trained Random Forest regression model "
            f"(test R² = {get_evaluation_summary()['regression']['r2']:.3f}). "
            "Algorithm-internal measurements not knowable in advance (comparisons/"
            "swaps/probes) are estimated from historical averages for this "
            "algorithm and input size, not supplied by you directly."
        )


# ---------------------------------------------------------------------------
# Page: Algorithm Recommendation
# ---------------------------------------------------------------------------

def render_algorithm_recommendation():
    st.title("Algorithm Recommendation")
    st.markdown("Enter an input profile to get a model-recommended algorithm, cross-checked against regression-predicted execution times.")

    unified_df = get_unified_dataset()

    st.markdown("<br>", unsafe_allow_html=True)
    domain = st.selectbox("Domain", ["Sorting", "Searching"], key="rec_domain")
    domain_key = domain.lower()
    condition_options = sorted(unified_df[unified_df["domain"] == domain_key]["input_condition"].unique())
    condition_label = "Data Type" if domain_key == "sorting" else "Case Type"
    algo_options = SORTING_ALGOS if domain_key == "sorting" else SEARCHING_ALGOS

    col1, col2 = st.columns(2)
    with col1:
        input_size = st.number_input("Input Size", min_value=1, max_value=2000, value=1000, step=10, key="rec_size")
        if input_size <= 0:
            st.error("Input size must be greater than 0.")
            return
    with col2:
        condition = st.selectbox(condition_label, condition_options, format_func=lambda x: x.replace("_", " ").title(), key="rec_cond")

    if domain_key == "sorting":
        c3, c4, c5 = st.columns(3)
        with c3:
            presortedness = st.slider("Presortedness", 0.0, 1.0, 0.5, key="rec_presort")
        with c4:
            duplicate_ratio = st.slider("Duplicate Ratio", 0.0, 1.0, 0.0, key="rec_dup")
        with c5:
            value_range = st.number_input("Value Range", min_value=1, value=90000, key="rec_vr")
    else:
        presortedness, duplicate_ratio, value_range = 0.0, 0.0, 0.0
        st.caption("Presortedness / duplicate ratio / value range are sorting-specific and not applicable here.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Get Recommendation", type="primary"):
        clf_model = get_classification_model()
        clf_row = pd.DataFrame([{
            "input_size": input_size,
            "presortedness_score": presortedness,
            "duplicate_ratio": duplicate_ratio,
            "value_range": value_range,
            "domain": domain_key,
            "input_condition": condition,
        }])[CLASSIFICATION_NUMERICAL + CLASSIFICATION_CATEGORICAL]

        classifier_pick = clf_model.predict(clf_row)[0]

        proba = clf_model.predict_proba(clf_row)[0]
        classes = clf_model.classes_
        proba_series = pd.Series(proba, index=classes).sort_values(ascending=False)

        reg_predictions = {}
        for algo in algo_options:
            reg_predictions[algo] = predict_execution_time(
                algo, domain_key, input_size, condition,
                presortedness, duplicate_ratio, value_range,
            )
        reg_ranking = sorted(reg_predictions.items(), key=lambda x: x[1])
        regression_pick = reg_ranking[0][0]

        st.markdown("---")
        st.subheader("Classifier Recommendation")
        st.markdown(
            f'<div class="rec-card"><div class="rec-rank">Model recommendation</div>'
            f'<div class="rec-name primary">{classifier_pick.replace("_", " ").title()}</div></div>',
            unsafe_allow_html=True,
        )

        st.subheader("Regression-Based Predicted Ranking")
        for i, (algo, t) in enumerate(reg_ranking[:5], start=1):
            cls = "rec-name primary" if i == 1 else "rec-name"
            st.markdown(
                f'<div class="rec-card"><div class="rec-rank">#{i}</div>'
                f'<div class="{cls}">{algo.replace("_", " ").title()}</div>'
                f'<div class="rec-time">Predicted: {t*1000:.4f} ms</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if classifier_pick == regression_pick:
            callout(
                f"Both the classifier and the regression-based ranking agree: "
                f"<b>{classifier_pick.replace('_', ' ').title()}</b> is recommended."
            )
        else:
            gap = reg_predictions[classifier_pick] - reg_predictions[regression_pick]
            rel_gap = gap / reg_predictions[regression_pick] if reg_predictions[regression_pick] > 0 else 0
            callout(
                f"<b>Model disagreement:</b> the classifier picked "
                f"<b>{classifier_pick.replace('_', ' ').title()}</b>, but the regression "
                f"model predicts <b>{regression_pick.replace('_', ' ').title()}</b> would be "
                f"slightly faster ({rel_gap*100:.1f}% difference in predicted time). "
                "This usually means the top candidates are in close competition for "
                "this profile — see the Predictability Gap page for why that happens.",
                warn=True,
            )

        st.subheader("Classifier Confidence (Top Candidates)")
        top_proba = proba_series.head(5).reset_index()
        top_proba.columns = ["algorithm", "probability"]
        fig = px.bar(top_proba, x="probability", y="algorithm", orientation="h",
                     labels={"probability": "Predicted Probability", "algorithm": "Algorithm"},
                     color_discrete_sequence=[AMBER])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

def render_model_performance():
    st.title("Model Performance")
    st.markdown("Metrics computed from the trained models' held-out test sets. Nothing here is hardcoded.")

    evaluation_summary = get_evaluation_summary()
    reg = evaluation_summary["regression"]
    clf = evaluation_summary["classification"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Regression: Execution Time Prediction")
    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        stat_block("R²", f"{reg['r2']:.4f}", accent="teal")
    with rcol2:
        stat_block("MAE", f"{reg['mae']*1000:.4f} ms")
    with rcol3:
        stat_block("RMSE", f"{reg['rmse']*1000:.4f} ms")

    actual_vs_pred_path = os.path.join(VIZ_DIR, "actual_vs_predicted_log.png")
    if os.path.exists(actual_vs_pred_path):
        st.image(actual_vs_pred_path, caption="Actual vs Predicted Execution Time (log scale)", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Classification: Best Algorithm Recommendation")
    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        stat_block("Accuracy", f"{clf['accuracy']:.4f}")
    with ccol2:
        stat_block("Macro F1", f"{clf['f1_macro']:.4f}", accent="amber")
    with ccol3:
        stat_block("Weighted F1", f"{clf['f1_weighted']:.4f}")

    callout(clf["note"], warn=True)

    confusion_path = os.path.join(VIZ_DIR, "confusion_matrix.png")
    if os.path.exists(confusion_path):
        st.image(confusion_path, caption="Confusion Matrix (Random Forest Classifier)", use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Predictability Gap
# ---------------------------------------------------------------------------

def render_predictability_gap():
    st.title("Predictability Gap")
    st.markdown(
        "Why can execution time be predicted almost perfectly, while recommending "
        "the single best algorithm is much harder? This page walks through the evidence."
    )

    evaluation_summary = get_evaluation_summary()
    profile_labels = get_profile_labels()

    st.markdown("<br>", unsafe_allow_html=True)
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        stat_block("Regression R²", f"{evaluation_summary['regression']['r2']:.3f}", accent="teal")
    with gcol2:
        stat_block("Classification Macro F1", f"{evaluation_summary['classification']['f1_macro']:.3f}", accent="amber")

    gap_path = os.path.join(VIZ_DIR, "predictability_gap.png")
    if os.path.exists(gap_path):
        st.image(gap_path, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Margin Analysis")
    st.markdown(
        "For each of the 2000 unique input profiles, we measure the relative gap "
        "between the best and second-best algorithm's execution time."
    )

    close_call_pct = (profile_labels["is_close_call"]).mean() * 100
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        stat_block("Median Relative Margin", f"{profile_labels['relative_margin'].median()*100:.1f}%")
    with mcol2:
        stat_block("Close Calls (≤10% margin)", f"{close_call_pct:.1f}%", accent="amber")
    with mcol3:
        stat_block("Total Profiles", len(profile_labels))

    margin_path = os.path.join(VIZ_DIR, "margin_distribution.png")
    if os.path.exists(margin_path):
        st.image(margin_path, use_container_width=True)

    accuracy_margin_path = os.path.join(VIZ_DIR, "accuracy_by_margin.png")
    if os.path.exists(accuracy_margin_path):
        st.subheader("Accuracy Drops as Competition Gets Closer")
        st.image(accuracy_margin_path, use_container_width=True)
        callout(
            "Classification accuracy ranges from ~42% when algorithms are in tight "
            "competition (0-5% margin) to ~96% when one algorithm clearly dominates "
            "(50%+ margin) — direct, quantitative evidence that the model's "
            "difficulty tracks exactly where algorithm performance genuinely is close."
        )

    confusion_path = os.path.join(VIZ_DIR, "confusion_matrix.png")
    if os.path.exists(confusion_path):
        st.subheader("Confusion Concentrates Among Close Competitors")
        st.image(confusion_path, use_container_width=True)
        callout(
            "Misclassification is almost entirely confined to binary/exponential/"
            "fibonacci/interpolation search — algorithms with genuinely similar "
            "performance profiles — while bubble_sort, radix_sort, and insertion_sort "
            "(with clearly dominant or weak profiles) are classified with zero errors."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Why This Happens")
    st.markdown(
        """
1. **Execution time is continuous** — regression errors are partially forgiven (predicting 0.052s vs true 0.050s is a small, quantifiable miss).
2. **Best-algorithm is discrete** — there's no partial credit; predicting the wrong algorithm is equally "wrong" whether the true margin was 50% or 0.01%.
3. **Small timing fluctuations flip labels** — 18.1% of profiles have a margin ≤10%, well within normal trial-to-trial noise.
4. **Some algorithms are near-identical in performance** — the minimum observed margin was 0.009%.
5. **The model reflects this** — confusion concentrates exactly where competition is closest (see confusion matrix above).
6. **Practical takeaway** — when the top algorithms are close, trust the regression model's predicted times over a single classifier label; a "close call" should be surfaced, not hidden behind false certainty.
        """
    )


# ---------------------------------------------------------------------------
# Page: About / Methodology
# ---------------------------------------------------------------------------

def render_about():
    st.title("About / Methodology")

    st.markdown(
        """
### Project

**AI-Based Algorithm Performance Analyser** benchmarks 13 classical sorting
and searching algorithms, then uses machine learning to go beyond static
theoretical complexity: predicting real execution time (regression) and
recommending the best algorithm for a given input profile (classification).

### Team & Work Allocation

| Member | Responsibilities |
|---|---|
| Nooryen (Member A) | Algorithm engine (13 algorithms), data generation pipeline, dataset cleaning, theoretical complexity documentation |
| Sudipta (Member B) | Feature engineering, regression + classification models, evaluation & predictability-gap analysis, interactive dashboard, final report |

### Dataset

- **Sorting:** 7 algorithms × 5 data types × 200 input sizes (10-2000, step 10) × 5 trials ≈ 35,000 rows
- **Searching:** 6 algorithms × 5 case types × 200 input sizes × 5 trials ≈ 30,000 rows
- Cleaned via a median-ratio outlier filter (more robust than IQR for small trial groups)

### ML Methodology

- **Feature engineering:** unified schema across domains; structural NaNs (features not applicable to a domain) filled with 0, with `domain` as a categorical signal so the model learns when a 0 is structural vs. real.
- **Regression:** Random Forest Regressor (`n_estimators=100, max_depth=12, min_samples_leaf=5`), grouped train/test split by configuration to prevent trial-level leakage.
- **Classification:** Random Forest Classifier, target derived from measured data (best_algorithm = lowest mean execution time per profile), evaluated with macro F1 due to severe class imbalance.
- **Predictability gap:** the project's central research contribution — see the dedicated page.

### Complexity Reference
        """
    )

    all_complexity = {**SORTING_COMPLEXITY, **SEARCHING_COMPLEXITY}
    rows = []
    for name, data in all_complexity.items():
        rows.append({
            "Algorithm": name.replace("_", " ").title(),
            "Best": data["best"],
            "Average": data["average"],
            "Worst": data["worst"],
            "Space": data["space"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
### Repository

Source code, notebooks, and models: `AI-Algorithm-Performance-Analyser` (GitHub),
branch `feature/sudipta-ml-dashboard`.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTES = {
    "Overview": render_overview,
    "Algorithm Explorer": render_algorithm_explorer,
    "Performance Comparison": render_performance_comparison,
    "AI Prediction": render_ai_prediction,
    "Algorithm Recommendation": render_algorithm_recommendation,
    "Model Performance": render_model_performance,
    "Predictability Gap": render_predictability_gap,
    "About / Methodology": render_about,
}

ROUTES[page]()