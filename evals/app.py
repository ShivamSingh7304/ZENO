# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Configure Logfire before importing application modules
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from dotenv import load_dotenv

load_dotenv()

import logfire

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="evals"
)


# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st


from evals.pipeline import (
    run_pipeline,
    load_golden_dataset,
)

from evals.guardrail_evals import (
    run_guardrails_eval,
    compute_guardrails_metrics,
)

from evals.metrics import (
    run_all_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ZENO-AI Evaluation Suite",
    page_icon="🧪",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_guardrail_label(intent: str) -> str:

    labels = {

        "safe":
            "✅ Safe",

        "off_topic":
            "🚫 Off Topic",

        "jailbreak":
            "🛡️ Jailbreak",

        "greeting":
            "👋 Greeting",

        "capabilities":
            "🤖 Capabilities",

        "farewell":
            "👋 Farewell",

        "unknown":
            "❓ Unknown",

    }

    return labels.get(
        intent,
        str(intent)
    )


def get_planner_label(intent):

    if intent is None:
        return "—"

    labels = {

        "CRISIS":
            "🚨 CRISIS",

        "CONVERSATIONAL":
            "💬 CONVERSATIONAL",

        "RAG":
            "📚 RAG",

        "unknown":
            "❓ Unknown",

    }

    return labels.get(
        intent,
        str(intent)
    )


def metric_grade(score: float) -> str:

    if score >= 0.90:
        return "Excellent"

    if score >= 0.75:
        return "Good"

    if score >= 0.50:
        return "Needs Improvement"

    return "Poor"


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "golden" not in st.session_state:

    st.session_state.golden = (
        load_golden_dataset()
    )


if "pipeline_done" not in st.session_state:

    st.session_state.pipeline_done = False


if "enriched_dataset" not in st.session_state:

    st.session_state.enriched_dataset = None


if "guardrails_results" not in st.session_state:

    st.session_state.guardrails_results = None


if "metric_results" not in st.session_state:

    st.session_state.metric_results = None


if "pipeline_rows" not in st.session_state:

    st.session_state.pipeline_rows = []


golden = st.session_state.golden


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title(
    "🧪 ZENO-AI Evaluation Suite"
)

st.caption(
    "Step 1: Review golden dataset → "
    "Step 2: Run live pipeline → "
    "Step 3: Evaluate Guardrails and Planner"
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(

    [

        "📋 Step 1 — Ground Truth",

        "🚀 Step 2 — Live Pipeline",

        "📊 Step 3 — Evaluation Metrics",

    ]

)


# =============================================================================
# TAB 1 — GROUND TRUTH
# =============================================================================

with tab1:

    st.subheader(
        "ZENO-AI Golden Evaluation Dataset"
    )

    dataset_info = golden.get(
        "dataset_info",
        {}
    )


    if dataset_info:

        st.info(

            dataset_info.get(
                "description",
                ""
            )

        )


    examples = golden.get(
        "examples",
        []
    )


    # ─────────────────────────────────────────────────────────────────────────
    # Dataset summary
    # ─────────────────────────────────────────────────────────────────────────

    total_examples = len(
        examples
    )

    guardrail_examples = [

        example

        for example in examples

        if example.get(
            "expected_guardrail_intent"
        ) != "safe"

    ]


    safe_examples = [

        example

        for example in examples

        if example.get(
            "expected_guardrail_intent"
        ) == "safe"

    ]


    crisis_examples = [

        example

        for example in examples

        if example.get(
            "expected_planner_intent"
        ) == "CRISIS"

    ]


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Examples",
        total_examples
    )

    col2.metric(
        "Guardrail Tests",
        len(guardrail_examples)
    )

    col3.metric(
        "Safe → Planner Tests",
        len(safe_examples)
    )

    col4.metric(
        "Crisis Tests",
        len(crisis_examples)
    )


    st.divider()


    # ─────────────────────────────────────────────────────────────────────────
    # Full dataset
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader(
        "Ground Truth Examples"
    )


    rows = []


    for example in examples:

        rows.append({

            "ID":
                example.get("id"),

            "Category":
                example.get("category"),

            "Input":
                example.get("input"),

            "Expected Guardrail":
                get_guardrail_label(

                    example.get(
                        "expected_guardrail_intent"
                    )

                ),

            "Expected Planner":
                get_planner_label(

                    example.get(
                        "expected_planner_intent"
                    )

                ),

            "Priority":
                example.get(
                    "priority",
                    ""
                ),

        })


    st.dataframe(

        pd.DataFrame(rows),

        use_container_width=True,

        hide_index=True,

    )


    st.caption(

        f"Loaded {total_examples} golden examples."

    )


    with st.expander(

        "View raw golden_dataset.json"

    ):

        st.json(
            golden
        )


# =============================================================================
# TAB 2 — LIVE PIPELINE
# =============================================================================

with tab2:

    st.subheader(
        "Live Pipeline Evaluation"
    )

    st.markdown(

        """
Sends every example to the running FastAPI endpoint:

`POST http://localhost:8000/query`

The API captures:

- `actual_response`
- `actual_guardrail_intent`
- `actual_planner_intent`

The evaluation tests two gates:

1. **Guardrails**
2. **Planner Routing**
        """

    )


    st.info(

        "Make sure FastAPI is running on "
        "`http://localhost:8000` before starting."

    )


    col1, col2, col3 = st.columns(

        [1, 1, 2]

    )


    run_pipeline_btn = col1.button(

        "▶️ Run Live Pipeline",

        type="primary",

        width="stretch",

        disabled=(
            st.session_state.pipeline_done
        )

    )


    reset_btn = col2.button(

        "🔄 Reset & Re-run",

        width="stretch",

        disabled=(
            not st.session_state.pipeline_done
        )

    )


    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    if reset_btn:

        st.session_state.pipeline_done = False

        st.session_state.enriched_dataset = None

        st.session_state.guardrails_results = None

        st.session_state.metric_results = None

        st.session_state.pipeline_rows = []

        st.rerun()


    # ─────────────────────────────────────────────────────────────────────────
    # Run pipeline
    # ─────────────────────────────────────────────────────────────────────────

    if run_pipeline_btn:

        st.session_state.pipeline_rows = []


        progress_bar = st.progress(

            0,

            text="Starting live pipeline..."

        )


        live_table_slot = st.empty()

        status_slot = st.empty()


        # ─────────────────────────────────────────────────────────────────────
        # Progress callback
        # ─────────────────────────────────────────────────────────────────────

        def pipeline_cb(

            i,

            total,

            question,

            stage,

            response=""

        ):

            if stage == "calling":

                percentage = int(

                    (i / total) * 100

                )

                progress_bar.progress(

                    percentage,

                    text=(
                        f"[{i + 1}/{total}] "
                        f"Calling API..."
                    )

                )


            elif stage == "done":

                short_question = (

                    question[:55] + "..."

                    if len(question) > 55

                    else question

                )


                short_response = (

                    response[:80] + "..."

                    if len(response) > 80

                    else response

                )


                st.session_state.pipeline_rows.append({

                    "#":
                        i + 1,

                    "Question":
                        short_question,

                    "Response":
                        (
                            short_response
                            if short_response
                            else "⚠️ No response"
                        ),

                    "Status":
                        "✅ Done",

                })


                live_table_slot.dataframe(

                    pd.DataFrame(

                        st.session_state.pipeline_rows

                    ),

                    use_container_width=True,

                    hide_index=True,

                )


                progress_bar.progress(

                    int(

                        ((i + 1) / total) * 100

                    ),

                    text=(
                        f"[{i + 1}/{total}] "
                        f"Completed"
                    )

                )


        # ─────────────────────────────────────────────────────────────────────
        # Run live API
        # ─────────────────────────────────────────────────────────────────────

        with logfire.span(

            "Streamlit — Run Live Pipeline"

        ):

            enriched = run_pipeline(

                golden,

                progress_callback=pipeline_cb

            )


            st.session_state.enriched_dataset = (

                enriched

            )


        progress_bar.progress(

            100,

            text="All pipeline responses collected."

        )


        status_slot.success(

            f"{len(enriched.get('examples', []))} "
            f"examples processed successfully."

        )


        # ─────────────────────────────────────────────────────────────────────
        # Show live results
        # ─────────────────────────────────────────────────────────────────────

        st.divider()

        st.subheader(
            "Live Pipeline Results"
        )


        live_rows = []


        for example in enriched.get(
            "examples",
            []
        ):

            live_rows.append({

                "ID":
                    example.get("id"),

                "Input":
                    example.get(
                        "input",
                        ""
                    )[:70],

                "Expected Guardrail":
                    get_guardrail_label(

                        example.get(
                            "expected_guardrail_intent"
                        )

                    ),

                "Actual Guardrail":
                    get_guardrail_label(

                        example.get(
                            "actual_guardrail_intent"
                        )

                    ),

                "Expected Planner":
                    get_planner_label(

                        example.get(
                            "expected_planner_intent"
                        )

                    ),

                "Actual Planner":
                    get_planner_label(

                        example.get(
                            "actual_planner_intent"
                        )

                    ),

                "Response":
                    example.get(
                        "actual_response",
                        ""
                    )[:100],

            })


        st.dataframe(

            pd.DataFrame(
                live_rows
            ),

            use_container_width=True,

            hide_index=True,

        )


        # ─────────────────────────────────────────────────────────────────────
        # Guardrail binary evaluation
        # ─────────────────────────────────────────────────────────────────────

        st.divider()

        st.subheader(
            "Guardrails Binary Evaluation"
        )


        guardrail_samples = [

            {

                **example,

                "expected_blocked": (

                    example.get(
                        "expected_guardrail_intent"
                    ) != "safe"

                )

            }

            for example in examples

        ]


        g_progress = st.progress(

            0,

            text="Running guardrail evaluation..."

        )


        def g_cb(

            i,

            total,

            input_text,

            *args

        ):

            g_progress.progress(

                int(

                    (i / total) * 100

                ),

                text=(

                    f"[{i + 1}/{total}] "

                    f"Testing guardrail..."

                )

            )


        with logfire.span(

            "Streamlit — Guardrails Evaluation"

        ):

            g_results = run_guardrails_eval(

                guardrail_samples,

                progress_callback=g_cb

            )


            g_metrics = (

                compute_guardrails_metrics(

                    g_results

                )

            )


            st.session_state.guardrails_results = (

                g_results

            )


            st.session_state.pipeline_done = True


        g_progress.progress(

            100,

            text="Guardrail evaluation complete."

        )


        # ─────────────────────────────────────────────────────────────────────
        # Display guardrail results
        # ─────────────────────────────────────────────────────────────────────

        guardrail_rows = []


        for result in g_results:

            result_label = {

                "TP":
                    "🟢 TP",

                "TN":
                    "🟢 TN",

                "FP":
                    "🔴 FP",

                "FN":
                    "🔴 FN",

            }.get(

                result.get("result"),

                result.get("result")

            )


            guardrail_rows.append({

                "ID":
                    result.get("id"),

                "Input":
                    result.get(
                        "input",
                        ""
                    )[:70],

                "Expected Intent":
                    get_guardrail_label(

                        result.get(
                            "expected_guardrail_intent"
                        )

                    ),

                "Actual Intent":
                    get_guardrail_label(

                        result.get(
                            "actual_guardrail_intent"
                        )

                    ),

                "Expected":
                    (
                        "Blocked"

                        if result.get(
                            "expected_blocked"
                        )

                        else "Safe"

                    ),

                "Actual":
                    (
                        "Blocked"

                        if result.get(
                            "actual_blocked"
                        )

                        else "Safe"

                    ),

                "Result":
                    result_label,

            })


        st.dataframe(

            pd.DataFrame(
                guardrail_rows
            ),

            use_container_width=True,

            hide_index=True,

        )


        # ─────────────────────────────────────────────────────────────────────
        # Guardrail metrics
        # ─────────────────────────────────────────────────────────────────────

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)


        mc1.metric(

            "Correct",

            f"{g_metrics['correct']}/"
            f"{g_metrics['total']}"

        )


        mc2.metric(

            "Precision",

            f"{g_metrics['precision']:.3f}"

        )


        mc3.metric(

            "Recall",

            f"{g_metrics['recall']:.3f}"

        )


        mc4.metric(

            "F1 Score",

            f"{g_metrics['f1_score']:.3f}"

        )


        mc5.metric(

            "Accuracy",

            f"{g_metrics['accuracy']:.3f}"

        )


    # ─────────────────────────────────────────────────────────────────────────
    # Show previous results
    # ─────────────────────────────────────────────────────────────────────────

    elif st.session_state.pipeline_done:

        st.success(
            "Pipeline already completed."
        )


        enriched = (

            st.session_state.enriched_dataset

        )


        rows = []


        for example in enriched.get(
            "examples",
            []
        ):

            rows.append({

                "ID":
                    example.get("id"),

                "Input":
                    example.get(
                        "input",
                        ""
                    )[:70],

                "Expected Guardrail":
                    get_guardrail_label(

                        example.get(
                            "expected_guardrail_intent"
                        )

                    ),

                "Actual Guardrail":
                    get_guardrail_label(

                        example.get(
                            "actual_guardrail_intent"
                        )

                    ),

                "Expected Planner":
                    get_planner_label(

                        example.get(
                            "expected_planner_intent"
                        )

                    ),

                "Actual Planner":
                    get_planner_label(

                        example.get(
                            "actual_planner_intent"
                        )

                    ),

            })


        st.dataframe(

            pd.DataFrame(rows),

            use_container_width=True,

            hide_index=True,

        )


        if st.session_state.guardrails_results:

            st.divider()

            st.subheader(
                "Guardrails Results"
            )


            guardrail_rows = []


            for result in (

                st.session_state.guardrails_results

            ):

                guardrail_rows.append({

                    "ID":
                        result.get("id"),

                    "Input":
                        result.get(
                            "input",
                            ""
                        )[:70],

                    "Expected Intent":
                        get_guardrail_label(

                            result.get(
                                "expected_guardrail_intent"
                            )

                        ),

                    "Actual Intent":
                        get_guardrail_label(

                            result.get(
                                "actual_guardrail_intent"
                            )

                        ),

                    "Result":
                        result.get(
                            "result"
                        ),

                })


            st.dataframe(

                pd.DataFrame(
                    guardrail_rows
                ),

                use_container_width=True,

                hide_index=True,

            )


# =============================================================================
# TAB 3 — EVALUATION METRICS
# =============================================================================

with tab3:

    st.subheader(
        "Pipeline Evaluation Metrics"
    )


    if not st.session_state.pipeline_done:

        st.warning(

            "Complete Step 2 first."

        )


    else:

        st.markdown(

            """
Evaluates the complete ZENO-AI pipeline:

1. Guardrail Intent Accuracy
2. Planner Routing Accuracy
3. Crisis Recall
4. Crisis False Negatives
5. End-to-End Pipeline Accuracy
6. Category-Level Accuracy
            """

        )


        run_metrics_btn = st.button(

            "▶️ Run Evaluation Metrics",

            type="primary"

        )


        if run_metrics_btn:

            with st.spinner(

                "Calculating evaluation metrics..."

            ):

                with logfire.span(

                    "Streamlit — Evaluation Metrics"

                ):

                    metric_results = (

                        run_all_metrics(

                            st.session_state
                            .enriched_dataset

                        )

                    )


                    st.session_state.metric_results = (

                        metric_results

                    )


            st.success(
                "Evaluation complete."
            )


        if st.session_state.metric_results:

            results = (

                st.session_state.metric_results

            )


            # ─────────────────────────────────────────────────────────────────
            # Summary metrics
            # ─────────────────────────────────────────────────────────────────

            st.divider()

            st.subheader(
                "Evaluation Summary"
            )


            c1, c2, c3, c4 = st.columns(4)


            c1.metric(

                "Guardrail Accuracy",

                f"{results.get('guardrail_accuracy', 0):.3f}"

            )


            c2.metric(

                "Planner Accuracy",

                f"{results.get('planner_accuracy', 0):.3f}"

            )


            c3.metric(

                "Crisis Recall",

                f"{results.get('crisis_recall', 0):.3f}"

            )


            c4.metric(

                "Pipeline Accuracy",

                f"{results.get('pipeline_accuracy', 0):.3f}"

            )


            # ─────────────────────────────────────────────────────────────────
            # Crisis safety metrics
            # ─────────────────────────────────────────────────────────────────

            st.divider()

            st.subheader(
                "🚨 Crisis Safety Evaluation"
            )


            crisis_fn = results.get(

                "crisis_false_negatives",

                0

            )


            crisis_total = (

                len(

                    results.get(
                        "crisis_results",
                        pd.DataFrame()
                    )

                )

            )


            cc1, cc2 = st.columns(2)


            cc1.metric(

                "Crisis False Negatives",

                crisis_fn

            )


            cc2.metric(

                "Total Crisis Examples",

                crisis_total

            )


            if crisis_fn > 0:

                st.error(

                    f"⚠️ {crisis_fn} crisis example(s) "
                    f"were not routed to CRISIS."

                )


            else:

                st.success(

                    "No crisis false negatives detected."

                )


            false_negative_df = results.get(

                "crisis_false_negative_examples"

            )


            if (

                false_negative_df is not None

                and not false_negative_df.empty

            ):

                st.dataframe(

                    false_negative_df,

                    use_container_width=True,

                    hide_index=True,

                )


            # ─────────────────────────────────────────────────────────────────
            # Guardrail results
            # ─────────────────────────────────────────────────────────────────

            st.divider()

            st.subheader(
                "Experiment 1 — Guardrail Intent Results"
            )


            guardrail_df = results.get(

                "guardrail_results"

            )


            if (

                guardrail_df is not None

                and not guardrail_df.empty

            ):

                st.dataframe(

                    guardrail_df,

                    use_container_width=True,

                    hide_index=True,

                )


            # ─────────────────────────────────────────────────────────────────
            # Planner results
            # ─────────────────────────────────────────────────────────────────

            st.divider()

            st.subheader(
                "Experiment 2 — Planner Routing Results"
            )


            planner_df = results.get(

                "planner_results"

            )


            if (

                planner_df is not None

                and not planner_df.empty

            ):

                st.dataframe(

                    planner_df,

                    use_container_width=True,

                    hide_index=True,

                )


            # ─────────────────────────────────────────────────────────────────
            # End-to-end results
            # ─────────────────────────────────────────────────────────────────

            st.divider()

            st.subheader(
                "Experiment 5 — End-to-End Pipeline"
            )


            pipeline_df = results.get(

                "pipeline_results"

            )


            if (

                pipeline_df is not None

                and not pipeline_df.empty

            ):

                st.dataframe(

                    pipeline_df,

                    use_container_width=True,

                    hide_index=True,

                )


            # ─────────────────────────────────────────────────────────────────
            # Category results
            # ─────────────────────────────────────────────────────────────────

            st.divider()

            st.subheader(
                "Experiment 6 — Accuracy by Category"
            )


            category_df = results.get(

                "category_results"

            )


            if (

                category_df is not None

                and not category_df.empty

            ):

                st.dataframe(

                    category_df,

                    use_container_width=True,

                    hide_index=True,

                )


            # ─────────────────────────────────────────────────────────────────
            # Crisis false positives
            # ─────────────────────────────────────────────────────────────────

            crisis_fp_df = results.get(

                "crisis_false_positives"

            )


            if (

                crisis_fp_df is not None

                and not crisis_fp_df.empty

            ):

                st.divider()

                st.subheader(
                    "Crisis False Positives"
                )

                st.warning(

                    "These non-crisis examples were "
                    "incorrectly routed to CRISIS."

                )

                st.dataframe(

                    crisis_fp_df,

                    use_container_width=True,

                    hide_index=True,

                )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "ZENO-AI Evaluation Suite | "
    "Guardrails → Planner → Retrieval → Response"
)