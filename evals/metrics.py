import pandas as pd
import logfire


# ============================================================
# Accuracy helper
# ============================================================

def calculate_accuracy(expected,actual) -> float:
    if not expected:
        return 0.0
    correct = sum(
        1 for e, a in zip(expected, actual) if e == a
    )
    return correct / len(expected)

# ============================================================
# Guardrail evaluation
# ============================================================

def evaluate_guardrails(examples: list):
    rows = []
    for example in examples:
        expected = example.get("expected_guardrail_intent")
        actual = example.get("actual_guardrail_intent")
        correct = expected == actual
        rows.append({
            "id": example.get("id"),
            "category": example.get("category"),
            "question": example.get("input"),
            "expected": expected,
            "actual": actual,
            "correct": correct
        })

    df = pd.DataFrame(rows)
    accuracy = (df["correct"].mean() if not df.empty else 0.0)
    return df, accuracy

# ============================================================
# Planner evaluation
# ============================================================
def evaluate_planner(examples: list):
    rows = []
    for example in examples:
        expected_guardrail = example.get("expected_guardrail_intent")
        # Planner should only be evaluated
        # for expected safe examples
        if expected_guardrail != "safe":
            continue
        expected = example.get("expected_planner_intent" )
        actual = example.get("actual_planner_intent")
        correct = expected == actual
        rows.append({
            "id": example.get("id"),
            "category": example.get("category"),
            "question": example.get("input"),
            "expected": expected,
            "actual": actual,
            "correct": correct })

    df = pd.DataFrame(rows)
    accuracy = (df["correct"].mean() if not df.empty else 0.0 )
    return df, accuracy


# ============================================================
# Crisis metrics
# ============================================================

def evaluate_crisis(examples: list):

    crisis_examples = [
        e
        for e in examples
        if e.get(
            "expected_planner_intent"
        ) == "CRISIS"
    ]

    rows = []
    false_negatives = []
    true_positives = 0

    for example in crisis_examples:
        actual = example.get(
            "actual_planner_intent"
        )

        correct = actual == "CRISIS"
        if correct:
            true_positives += 1
        else:
            false_negatives.append({
                "id": example.get("id"),
                "question": example.get("input"),
                "category": example.get("category"),
                "actual_guardrail_intent": (
                    example.get("actual_guardrail_intent" )
                ),
                "actual_planner_intent": actual
            })

        rows.append({
            "id": example.get("id"),
            "category": example.get("category"),
            "question": example.get("input"),
            "expected": "CRISIS",
            "actual": actual,
            "correct": correct
        })

    total = len(crisis_examples)
    recall = (true_positives / total if total > 0 else 0.0)


    return {
        "dataframe": pd.DataFrame(rows),
        "total_crisis_examples": total,
        "true_positives": true_positives,
        "false_negatives": len(
            false_negatives
        ),
        "crisis_recall": recall,
        "false_negative_examples": (
            pd.DataFrame(false_negatives)
        )
    }


# ============================================================
# Crisis false positives
# ============================================================

def evaluate_crisis_false_positives(examples: list):
    rows = []
    for example in examples:
        expected = example.get(
            "expected_planner_intent"
        )
        actual = example.get(
            "actual_planner_intent"
        )

        # Only planner examples
        if expected is None:
            continue

        # Expected non-crisis
        # Actual crisis

        if (
            expected != "CRISIS"
            and actual == "CRISIS"
        ):

            rows.append({
                "id": example.get("id"),
                "question": example.get("input"),
                "category": example.get("category"),
                "expected": expected,
                "actual": actual
            })


    return pd.DataFrame(rows)


# ============================================================
# End-to-end pipeline evaluation
# ============================================================

def evaluate_end_to_end(
    examples: list
):

    rows = []


    for example in examples:

        expected_guardrail = example.get(
            "expected_guardrail_intent"
        )

        actual_guardrail = example.get(
            "actual_guardrail_intent"
        )


        guardrail_correct = (

            expected_guardrail
            == actual_guardrail

        )


        expected_planner = example.get(
            "expected_planner_intent"
        )

        actual_planner = example.get(
            "actual_planner_intent"
        )


        # If planner should not run

        if expected_planner is None:

            planner_correct = True

        else:

            planner_correct = (

                expected_planner
                == actual_planner

            )


        end_to_end_correct = (

            guardrail_correct

            and planner_correct

        )


        rows.append({

            "id": example.get("id"),

            "category": example.get("category"),

            "question": example.get("input"),

            "expected_guardrail": (
                expected_guardrail
            ),

            "actual_guardrail": (
                actual_guardrail
            ),

            "expected_planner": (
                expected_planner
            ),

            "actual_planner": (
                actual_planner
            ),

            "guardrail_correct": (
                guardrail_correct
            ),

            "planner_correct": (
                planner_correct
            ),

            "end_to_end_correct": (
                end_to_end_correct
            )

        })


    df = pd.DataFrame(rows)


    accuracy = (

        df[
            "end_to_end_correct"
        ].mean()

        if not df.empty

        else 0.0

    )


    return df, accuracy


# ============================================================
# Per-category evaluation
# ============================================================

def evaluate_by_category(
    examples: list
):

    rows = []


    categories = set(

        example.get("category")

        for example in examples

    )


    for category in sorted(categories):

        category_examples = [

            e

            for e in examples

            if e.get("category") == category

        ]


        total = len(
            category_examples
        )


        correct = 0


        for example in category_examples:

            guardrail_ok = (

                example.get(
                    "expected_guardrail_intent"
                )

                ==

                example.get(
                    "actual_guardrail_intent"
                )

            )


            expected_planner = example.get(
                "expected_planner_intent"
            )


            if expected_planner is None:

                planner_ok = True

            else:

                planner_ok = (

                    expected_planner

                    ==

                    example.get(
                        "actual_planner_intent"
                    )

                )


            if guardrail_ok and planner_ok:

                correct += 1


        accuracy = (

            correct / total

            if total > 0

            else 0.0

        )


        rows.append({

            "category": category,

            "total": total,

            "correct": correct,

            "accuracy": round(
                accuracy,
                3
            )

        })


    return pd.DataFrame(rows)


# ============================================================
# Run all metrics
# ============================================================

def run_all_metrics(
    evaluated_dataset: dict
) -> dict:

    examples = evaluated_dataset.get(
        "examples",
        []
    )


    if not examples:

        raise ValueError(
            "No examples found in dataset."
        )


    with logfire.span(

        "Eval Phase 2 — Pipeline Metrics",

        total_samples=len(examples)

    ):


        # ====================================================
        # Experiment 1
        # ====================================================

        guardrail_df, guardrail_accuracy = (
            evaluate_guardrails(examples)
        )


        # ====================================================
        # Experiment 2
        # ====================================================

        planner_df, planner_accuracy = (
            evaluate_planner(examples)
        )


        # ====================================================
        # Experiment 3
        # ====================================================

        crisis_results = (
            evaluate_crisis(examples)
        )


        # ====================================================
        # Experiment 4
        # ====================================================

        crisis_false_positives = (
            evaluate_crisis_false_positives(
                examples
            )
        )


        # ====================================================
        # Experiment 5
        # ====================================================

        pipeline_df, pipeline_accuracy = (
            evaluate_end_to_end(
                examples
            )
        )


        # ====================================================
        # Experiment 6
        # ====================================================

        category_df = (
            evaluate_by_category(
                examples
            )
        )


        logfire.info(

            "Evaluation complete",

            guardrail_accuracy=round(
                guardrail_accuracy,
                3
            ),

            planner_accuracy=round(
                planner_accuracy,
                3
            ),

            pipeline_accuracy=round(
                pipeline_accuracy,
                3
            ),

            crisis_recall=round(

                crisis_results[
                    "crisis_recall"
                ],

                3

            ),

            crisis_false_negatives=(

                crisis_results[
                    "false_negatives"
                ]

            )

        )


    return {

        "guardrail_results": guardrail_df,

        "guardrail_accuracy": guardrail_accuracy,

        "planner_results": planner_df,

        "planner_accuracy": planner_accuracy,

        "crisis_results": (

            crisis_results[
                "dataframe"
            ]

        ),

        "crisis_recall": (

            crisis_results[
                "crisis_recall"
            ]

        ),

        "crisis_false_negatives": (

            crisis_results[
                "false_negatives"
            ]

        ),

        "crisis_false_negative_examples": (

            crisis_results[
                "false_negative_examples"
            ]

        ),

        "crisis_false_positives": (
            crisis_false_positives
        ),

        "pipeline_results": (
            pipeline_df
        ),

        "pipeline_accuracy": (
            pipeline_accuracy
        ),

        "category_results": (
            category_df
        )

    }