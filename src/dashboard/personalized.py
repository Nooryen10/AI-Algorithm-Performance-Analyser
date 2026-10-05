"""
Personalized Prediction -- an ADDITIVE dashboard feature.

The user types their own numbers (and, for searching, a target value). The
system then:
  1. computes the data's features with the project's own feature extractor,
  2. feeds them to the EXISTING trained regression + classification models,
  3. recommends an algorithm,
  4. actually runs the recommended algorithm on the user's data using the
     project's own instrumented implementations, and
  5. validates the prediction by timing every algorithm on that same data.

Design rules:
  * No model is retrained or replaced; the pickled pipelines used by the other
    pages are reused, and the existing `predict_execution_time` is injected via
    `ctx` so the logic is not duplicated.
  * Nothing here writes to disk or touches any other page. All state lives in
    `st.session_state` under the "pz_" prefix.
"""

import importlib
import logging
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# Range the models were trained on (see data/processed/*.csv).
SIZE_MIN, SIZE_MAX = 10, 2000


def _pretty(name):
    return str(name).replace("_", " ").title()


def _confidence_label(p):
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# "Enter My Own Data": user gives real numbers -> features are COMPUTED from
# them -> existing ML models predict/recommend -> the recommended algorithm is
# then actually executed on the user's data using the project's own
# instrumented implementations.
# ---------------------------------------------------------------------------

DATA_MAX_VALUE = 1_000_000
DATA_MAX_COUNT = 5000  # keeps slow O(n^2) sorts responsive in the browser
EXAMPLE_DATA = "64, 25, 12, 22, 11, 90, 5, 77, 31, 48, 19, 3"
# Search algorithms whose returned index refers to the internally SORTED copy.
_INDEX_IN_SORTED = {"binary_search", "interpolation_search",
                    "exponential_search", "fibonacci_search"}


def parse_user_numbers(text):
    """Parse comma/space/newline separated whole numbers. Returns (list, errors)."""
    tokens = [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]
    if not tokens:
        return [], ["Please enter some numbers, separated by commas or spaces."]
    bad = [t for t in tokens if not re.fullmatch(r"-?\d+", t)]
    if bad:
        shown = ", ".join(repr(b) for b in bad[:5]) + (" ..." if len(bad) > 5 else "")
        return [], [f"These entries are not whole numbers: {shown}. Use whole numbers only."]
    return [int(t) for t in tokens], []


def validate_user_data(arr, task, target=None):
    """Return error strings for a parsed array (and target when searching)."""
    errors = []
    if len(arr) < 1 or len(arr) > DATA_MAX_COUNT:
        errors.append(
            f"Please enter between 1 and {DATA_MAX_COUNT:,} numbers (you entered "
            f"{len(arr):,}).")
    if any(x < 0 for x in arr):
        errors.append("Negative numbers are not supported (Radix Sort needs non-negative integers).")
    if any(x > DATA_MAX_VALUE for x in arr):
        errors.append(f"Values must not exceed {DATA_MAX_VALUE:,}.")
    if task == "Search" and (target is None or target < 0 or target > DATA_MAX_VALUE * 2):
        errors.append("Please enter a whole-number target value to search for (0 or more).")
    return errors


def data_range_warnings(n):
    """Non-blocking notes when the list is outside the size range the models saw."""
    if n < SIZE_MIN:
        return [f"Your list has only {n} number(s); the models were trained on {SIZE_MIN}-{SIZE_MAX:,}. "
                "The data is still sorted/searched correctly, but the ML time predictions are "
                "rough at this size."]
    if n > SIZE_MAX:
        return [f"Your list has {n:,} numbers; the models were trained on {SIZE_MIN}-{SIZE_MAX:,}. "
                "The data is still processed correctly, but predictions beyond that range are "
                "extrapolated and less reliable."]
    return []


def derive_sorting_condition(presortedness, dup_ratio):
    """Map computed features to the data_type labels the models were trained on,
    using the ranges observed in the training data."""
    if presortedness >= 0.9999:
        return "sorted"
    if presortedness <= 0.0001:
        return "reverse_sorted"
    if dup_ratio >= 0.20:
        return "duplicates"
    if presortedness >= 0.75:
        return "nearly_sorted"
    return "random"


def derive_search_case(arr, target):
    """Map (array, target) to the case_type labels used by the training data
    (same definitions as src/data_generation/search_case_generator.py)."""
    if target not in set(arr):
        return "not_found"
    srt = sorted(arr)
    if target == srt[len(srt) // 2]:
        return "best_case"
    if target == srt[-1]:
        return "worst_case"
    if arr.count(target) > 1:
        return "duplicates_present"
    return "average_case"


def _load_algorithm(domain, name):
    return getattr(importlib.import_module(f"src.algorithms.{domain}.{name}"), name)


def analyze_own_data(task, arr, target, ctx):
    """Compute features from the user's data, run the existing models, then
    execute the recommended algorithm (and every other one, for validation)."""
    from src.data_generation.feature_extractor import extract_sorting_features

    domain = "sorting" if task == "Sort" else "searching"
    algos = ctx.sorting_algos if domain == "sorting" else ctx.searching_algos
    n = len(arr)

    if domain == "sorting":
        feats = extract_sorting_features(arr)  # the project's own feature extractor
        presort, dup, vr = feats["presortedness_score"], feats["duplicate_ratio"], feats["value_range"]
        condition = derive_sorting_condition(presort, dup)
    else:
        presort = dup = vr = 0.0  # sorting-specific features are not used for searching
        condition = derive_search_case(arr, target)

    # --- existing models --------------------------------------------------
    reg_pred = {a: float(ctx.predict_execution_time(a, domain, n, condition, presort, dup, vr))
                for a in algos}
    reg_ranking = sorted(reg_pred.items(), key=lambda kv: kv[1])

    clf = ctx.get_classification_model()
    row = pd.DataFrame([{
        "input_size": n, "presortedness_score": presort, "duplicate_ratio": dup,
        "value_range": vr, "domain": domain, "input_condition": condition,
    }])[ctx.clf_numerical + ctx.clf_categorical]
    proba = pd.Series(clf.predict_proba(row)[0], index=clf.classes_)
    in_domain = proba[[c for c in proba.index if c in algos]].sort_values(ascending=False)
    clf_pick = in_domain.index[0]  # best class that belongs to this task's domain

    # --- actually run the algorithms on the user's data ---------------------
    runs = {}
    for a in algos:
        fn = _load_algorithm(domain, a)
        out = fn(list(arr)) if domain == "sorting" else fn(list(arr), target)
        runs[a] = out
    chosen = runs[clf_pick]

    if domain == "sorting":
        verified = chosen["result"] == sorted(arr)
    else:
        found = chosen["result"]["found"]
        verified = found == (target in set(arr))

    benchmark = pd.DataFrame([{
        "algorithm": a,
        "predicted_ms": reg_pred[a] * 1000,
        "measured_ms": runs[a]["time_taken"] * 1000,
        "operations": runs[a]["comparisons"] if domain == "sorting" else runs[a]["probes"],
    } for a in algos]).sort_values("predicted_ms").reset_index(drop=True)

    return {
        "task": task, "domain": domain, "n": n, "arr": list(arr), "target": target,
        "presortedness": presort, "duplicate_ratio": dup, "value_range": vr,
        "condition": condition, "reg_ranking": reg_ranking, "reg_pred": reg_pred,
        "clf_pick": clf_pick, "clf_confidence": float(in_domain.iloc[0]),
        "chosen_run": chosen, "verified": bool(verified), "benchmark": benchmark,
        "warnings": data_range_warnings(n),
        "reg_pick": reg_ranking[0][0],
    }


def own_data_recommendation(a):
    """Dynamic sentence built from the real analysis of the user's data."""
    name = _pretty(a["clf_pick"])
    if a["domain"] == "sorting":
        desc = (f"Your {a['n']:,} numbers have presortedness {a['presortedness']:.2f}, "
                f"duplicate ratio {a['duplicate_ratio']:.2f} and value range "
                f"{a['value_range']:,}, so they were classified as "
                f"<b>{_pretty(a['condition']).lower()}</b> data.")
        action = f"The classifier recommends <b>{name}</b> and it was used to sort your data."
    else:
        desc = (f"You are searching {a['n']:,} numbers for <b>{a['target']}</b>, which "
                f"matches the <b>{_pretty(a['condition']).lower()}</b> scenario.")
        action = f"The classifier recommends <b>{name}</b> and it was used to run your search."
    conf = f"Confidence: {_confidence_label(a['clf_confidence']).lower()} " \
           f"({a['clf_confidence']*100:.0f}%)."
    extra = ""
    if a["reg_pick"] != a["clf_pick"]:
        extra = (f" The regression model ranks <b>{_pretty(a['reg_pick'])}</b> fastest, so the "
                 "top candidates are close (see the Predictability Gap page).")
    return f"{desc} {action} {conf}{extra}"


def _own_data_graphs(a, ctx):
    """Predicted (regression) vs actually measured time, per algorithm."""
    b = a["benchmark"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Predicted (ML regression)", x=[_pretty(x) for x in b["algorithm"]],
                         y=b["predicted_ms"], marker_color=ctx.teal))
    fig.add_trace(go.Bar(name="Measured on your data", x=[_pretty(x) for x in b["algorithm"]],
                         y=b["measured_ms"], marker_color=ctx.amber))
    fig.update_layout(**ctx.plotly_layout, barmode="group", yaxis_title="Time (ms)")
    figs = [fig]
    if a["domain"] == "sorting":
        before_after = go.Figure()
        before_after.add_trace(go.Scatter(y=a["arr"], mode="lines+markers", name="Your input order",
                                          line=dict(color="#8B93A1")))
        before_after.add_trace(go.Scatter(y=a["chosen_run"]["result"], mode="lines+markers",
                                          name=f"After {_pretty(a['clf_pick'])}",
                                          line=dict(color=ctx.teal)))
        before_after.update_layout(**ctx.plotly_layout, xaxis_title="Position", yaxis_title="Value")
        figs.append(before_after)
    return figs


def render_own_data_page(ctx):
    """UI for the 'Enter My Own Data' mode."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Your data")
    task = st.radio("What do you want to do?", ["Sort", "Search"], horizontal=True, key="pz_data_task")
    text = st.text_area(
        "Your numbers (whole numbers, separated by commas, spaces or new lines)",
        value=EXAMPLE_DATA, height=110, key="pz_data_text",
        help=f"Any amount from 1 to {DATA_MAX_COUNT:,} non-negative whole numbers, each up to "
             f"{DATA_MAX_VALUE:,}. Predictions are most reliable for {SIZE_MIN}-{SIZE_MAX:,} "
             "numbers, the size range the models were trained on.")
    target = None
    if task == "Search":
        target = st.number_input("Value to search for", value=None, step=1, format="%d",
                                 key="pz_data_target", placeholder="e.g. 48",
                                 help="The number you want to find in your list.")

    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        run = st.button("Run on My Data", type="primary", key="pz_data_run")
    with b2:
        st.button("Reset to Default", on_click=_reset_to_default, key="pz_data_reset")

    if run:
        arr, errors = parse_user_numbers(text)
        if not errors:
            errors = validate_user_data(arr, task, None if target is None else int(target))
        if errors:
            for e in errors:
                st.error(e)
            st.session_state.pop("pz_data_last", None)
        else:
            try:
                st.session_state["pz_data_last"] = analyze_own_data(
                    task, arr, None if target is None else int(target), ctx)
            except Exception:
                logger.exception("Own-data analysis failed")
                st.error("Unable to generate the prediction. Please check your input "
                         "values and try again.")

    a = st.session_state.get("pz_data_last")
    if not a:
        return

    st.markdown("---")
    for w in a.get("warnings", []):
        st.info(w)
    st.subheader("What the system computed from your data")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ctx.stat_block("Elements", f"{a['n']:,}", small=True)
    with c2:
        ctx.stat_block("Detected Scenario", _pretty(a["condition"]), small=True)
    if a["domain"] == "sorting":
        with c3:
            ctx.stat_block("Presortedness", f"{a['presortedness']:.3f}", small=True)
        with c4:
            ctx.stat_block("Duplicate Ratio", f"{a['duplicate_ratio']:.3f}", small=True)
    else:
        with c3:
            ctx.stat_block("Target", f"{a['target']}", small=True)
        with c4:
            ctx.stat_block("Target in list?", "Yes" if a["condition"] != "not_found" else "No", small=True)
    st.caption("These features are calculated from your numbers by the project's own "
               "feature extractor and fed to the trained models.")

    st.subheader("AI Recommendation")
    ctx.callout(own_data_recommendation(a))

    run_ = a["chosen_run"]
    st.subheader("Result of running the recommended algorithm on your data")
    if a["domain"] == "sorting":
        st.text_area("Sorted output", ", ".join(map(str, run_["result"])), height=110,
                     disabled=True, key="pz_data_out")
        ops = f"{run_['comparisons']:,} comparisons, {run_['swaps']:,} swaps/shifts"
    else:
        res = run_["result"]
        if res["found"]:
            positions = [i for i, v in enumerate(a["arr"]) if v == a["target"]]
            where = (f"index {res['index']} in the sorted list" if a["clf_pick"] in _INDEX_IN_SORTED
                     else f"index {res['index']} in your list")
            st.success(f"{a['target']} was found at {where}. "
                       f"It occurs at position(s) {positions[:10]}{' ...' if len(positions) > 10 else ''} "
                       "of your original list (0-based).")
        else:
            st.warning(f"{a['target']} is not in your list.")
        ops = f"{run_['probes']:,} probes"
    r1, r2, r3 = st.columns(3)
    with r1:
        ctx.stat_block("Algorithm Used", _pretty(a["clf_pick"]), accent="teal", small=True)
    with r2:
        ctx.stat_block("Actual Time", f"{run_['time_taken']*1000:.4f} ms", small=True)
    with r3:
        ctx.stat_block("Work Done", ops, small=True)
    st.caption(("Output checked against Python's own sorted(): correct." if a["domain"] == "sorting"
                else "Result checked against a direct membership test: correct.")
               if a["verified"] else "Warning: output did not pass the correctness check.")

    st.subheader("Predicted vs Measured Time (all algorithms on your data)")
    figs = _own_data_graphs(a, ctx)
    st.plotly_chart(figs[0], use_container_width=True)
    st.caption("Predicted = the regression model's estimate; measured = the algorithm actually "
               "run on your numbers. For very small inputs timings are in microseconds, so "
               "measured values are noisy and can differ from the model's averages.")
    if len(figs) > 1:
        st.subheader("Your data before and after sorting")
        st.plotly_chart(figs[1], use_container_width=True)
    show = a["benchmark"].copy()
    show["algorithm"] = show["algorithm"].map(_pretty)
    show = show.rename(columns={"algorithm": "Algorithm", "predicted_ms": "Predicted (ms)",
                                "measured_ms": "Measured (ms)",
                                "operations": "Comparisons" if a["domain"] == "sorting" else "Probes"})
    st.dataframe(show.round(4), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Streamlit state + page
# ---------------------------------------------------------------------------

_STATE_KEYS_PREFIX = "pz_"


def _reset_to_default():
    """Callback: clear the entered data, the result and the graphs."""
    for k in [k for k in st.session_state.keys() if k.startswith(_STATE_KEYS_PREFIX)]:
        del st.session_state[k]


def render_personalized_page(ctx):
    st.title("Personalized Prediction")
    st.markdown(
        "Enter your own numbers. The app computes their features, sends them through the "
        "same trained Random Forest models used elsewhere in this app, recommends an "
        "algorithm, and then actually sorts or searches your data with it."
    )
    render_own_data_page(ctx)