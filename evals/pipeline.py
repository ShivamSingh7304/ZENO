import time
import copy
import json
import os

import requests
import logfire


API_URL = "http://localhost:8000/query"

RESPONSE_TRUNCATE = 300

DELAY_BETWEEN_CALLS = 10

REQUEST_TIMEOUT = 120


def run_pipeline(
    golden_dataset: dict,
    progress_callback=None
) -> dict:

    """
    Runs every example through the live FastAPI pipeline.

    Captures:

        actual_response
        actual_guardrail_intent
        actual_planner_intent

    The API now returns both intents directly.
    """

    dataset = copy.deepcopy(
        golden_dataset
    )

    examples = dataset.get(
        "examples",
        []
    )

    total = len(examples)

    with logfire.span(

        "Eval Phase 1 — Live Pipeline",

        total_samples=total

    ):

        for i, example in enumerate(examples):

            message = example["input"]

            if progress_callback:

                progress_callback(
                    i,
                    total,
                    message,
                    "calling"
                )

            with logfire.span(

                f"Live Query {i + 1}/{total}",

                message=message[:80],

                category=example.get(
                    "category",
                    ""
                ),

                priority=example.get(
                    "priority",
                    ""
                )

            ):

                try:

                    response = requests.post(

                        API_URL,

                        json={

                            "q": message,

                            "thread_id": (
                                f"eval_run_{i}"
                            )

                        },

                        timeout=REQUEST_TIMEOUT

                    )

                    response.raise_for_status()

                    data = response.json()


                    # ==========================================
                    # Capture API output
                    # ==========================================

                    raw_answer = (
                        data.get("answer")
                        or ""
                    )

                    actual_guardrail = (
                        data.get(
                            "actual_guardrail_intent"
                        )
                        or "unknown"
                    )

                    actual_planner = (
                        data.get(
                            "actual_planner_intent"
                        )
                    )


                    example[
                        "actual_response"
                    ] = raw_answer[
                        :RESPONSE_TRUNCATE
                    ]

                    example[
                        "actual_guardrail_intent"
                    ] = actual_guardrail

                    example[
                        "actual_planner_intent"
                    ] = actual_planner


                    # ==========================================
                    # Crisis false negative
                    # ==========================================

                    expected_planner = example.get(
                        "expected_planner_intent"
                    )

                    if (
                        expected_planner == "CRISIS"
                        and actual_planner != "CRISIS"
                    ):

                        logfire.error(

                            "CRISIS FALSE NEGATIVE",

                            example_id=example.get("id"),

                            query=message[:100],

                            expected=expected_planner,

                            actual_guardrail=actual_guardrail,

                            actual_planner=actual_planner

                        )

                    else:

                        logfire.info(

                            "Response captured",

                            example_id=example.get("id"),

                            guardrail_intent=actual_guardrail,

                            planner_intent=actual_planner,

                            response_chars=len(raw_answer)

                        )


                # ==============================================
                # Connection error
                # ==============================================

                except requests.exceptions.ConnectionError:

                    logfire.error(

                        "Cannot reach FastAPI. "
                        "Is the application running on port 8000?"
                    )

                    example[
                        "actual_response"
                    ] = ""

                    example[
                        "actual_guardrail_intent"
                    ] = "unknown"

                    example[
                        "actual_planner_intent"
                    ] = "unknown"


                # ==============================================
                # Other errors
                # ==============================================

                except Exception as e:

                    logfire.error(
                        f"Query failed: {e}"
                    )

                    example[
                        "actual_response"
                    ] = ""

                    example[
                        "actual_guardrail_intent"
                    ] = "unknown"

                    example[
                        "actual_planner_intent"
                    ] = "unknown"


            if progress_callback:

                progress_callback(

                    i,

                    total,

                    message,

                    "done",

                    example.get(
                        "actual_response",
                        ""
                    )

                )


            if i < total - 1:

                time.sleep(
                    DELAY_BETWEEN_CALLS
                )


    return dataset


# ============================================================
# Save
# ============================================================

def save_results(
    dataset: dict,
    path: str
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# Load dataset
# ============================================================

def load_golden_dataset() -> dict:

    golden_path = os.path.join(

        os.path.dirname(__file__),

        "golden_dataset.json"

    )

    with open(
        golden_path,
        encoding="utf-8"
    ) as f:

        return json.load(f)