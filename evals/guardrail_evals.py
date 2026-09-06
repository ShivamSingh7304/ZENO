"""
Guardrails evaluation.

Sends each test input to the live /query API.

The API returns the exact guardrail intent:

    off_topic
    jailbreak
    greeting
    capabilities
    farewell
    safe
    unknown

A request is considered blocked when the returned
actual_guardrail_intent is NOT "safe".

Each result is classified as:

    TP - Expected blocked, actually blocked
    TN - Expected safe, actually safe
    FP - Expected safe, but blocked
    FN - Expected blocked, but allowed

Also captures the actual guardrail intent for detailed analysis.
"""


import time
import copy
import requests
import logfire

API_URL = "http://localhost:8000/query"
REQUEST_TIMEOUT = 120
DELAY_BETWEEN_CALLS = 10

# ============================================================
# Guardrail blocking detection
# ============================================================
def _get_guardrail_intent(response_json: dict) -> str:
    """
    Gets the exact guardrail intent directly from the API response.
    Expected values:
        off_topic
        jailbreak
        greeting
        capabilities
        farewell
        safe
        unknown
    """
    intent = response_json.get(
        "actual_guardrail_intent",
        "unknown"
    )
    if intent is None:
        return "unknown"
    return str(intent).lower().strip()

def _is_blocked(response_json: dict) -> bool:
    """
    Determines whether the guardrail blocked the request.
    Any known non-safe guardrail intent means the request was blocked.
    """
    intent = _get_guardrail_intent(response_json)
    blocked_intents = {
        "off_topic",
        "jailbreak",
        "greeting",
        "capabilities",
        "farewell",
    }
    return intent in blocked_intents


# ============================================================
# Run evaluation
# ============================================================

def run_guardrails_eval(
    guardrails_samples: list,
    progress_callback=None
) -> list:
    """
    Runs each guardrail test case against the live FastAPI API.
    Adds:
        actual_blocked
        actual_guardrail_intent
        result
    Result is one of:
        TP
        TN
        FP
        FN
    Returns the enriched list.
    """
    samples = copy.deepcopy(guardrails_samples)
    total = len(samples)
    with logfire.span(
        "Eval — Guardrails Tests",
        total=total
    ):
        for i, sample in enumerate(samples):
            message = sample["input"]
            if progress_callback:
                progress_callback(
                    i,
                    total,
                    message
                )

            with logfire.span(
                f"Guardrail Test {sample['id']}",
                input_text=message[:80],
                expected_blocked=sample.get(
                    "expected_blocked"
                )
            ):
                try:
                    response = requests.post(
                        API_URL,
                        json={
                            "q": message,
                            "thread_id": (
                                f"guardrail_eval_{i}"
                            )
                        },
                        timeout=REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    data = response.json()
                    # ==========================================
                    # Get exact intent directly from API
                    # ==========================================
                    actual_intent = (
                        _get_guardrail_intent(data)
                    )
                    blocked = (
                        _is_blocked(data)
                    )

                # ==============================================
                # Connection error
                # ==============================================

                except requests.exceptions.ConnectionError:
                    logfire.error(
                        "Cannot reach FastAPI — "
                        "is the app running on port 8000?"
                    )
                    actual_intent = "unknown"
                    blocked = False

                # ==============================================
                # Other API errors
                # ==============================================

                except Exception as e:
                    logfire.error(
                        f"Guardrails test error: {e}"
                    )
                    actual_intent = "unknown"
                    blocked = False

                # ==============================================
                # Expected result
                # ==============================================

                expected_blocked = sample.get(
                    "expected_blocked",
                    False
                )

                # Store actual values
                sample[
                    "actual_guardrail_intent"
                ] = actual_intent
                sample[
                    "actual_blocked"
                ] = blocked


                # ==============================================
                # TP / TN / FP / FN
                # ==============================================

                if expected_blocked and blocked:
                    result = "TP"
                elif expected_blocked and not blocked:
                    result = "FN"
                elif not expected_blocked and not blocked:
                    result = "TN"
                else:
                    result = "FP"

                sample["result"] = result

                # ==============================================
                # Log result
                # ==============================================

                logfire.info(
                    f"Guardrail Result: {result}",
                    expected_blocked=expected_blocked,
                    actual_blocked=blocked,
                    actual_guardrail_intent=actual_intent,
                    input_preview=message[:60]
                )


                # ==============================================
                # Highlight false negatives
                # ==============================================

                if result == "FN":
                    logfire.warning(
                        "GUARDRAIL FALSE NEGATIVE",
                        sample_id=sample.get("id"),
                        expected_blocked=expected_blocked,
                        actual_guardrail_intent=actual_intent,
                        input_preview=message[:100]
                    )


            # ====================================================
            # Progress callback after completion
            # ====================================================

            if progress_callback:
                progress_callback(
                    i,
                    total,
                    message,
                    "done"
                )

            # ====================================================
            # Rate limiting delay
            # ====================================================

            if i < total - 1:
                time.sleep(
                    DELAY_BETWEEN_CALLS
                )

    return samples


# ============================================================
# Compute binary guardrail metrics
# ============================================================

def compute_guardrails_metrics(
    results: list
) -> dict:
    """
    Computes:
        TP
        TN
        FP
        FN
        Precision
        Recall
        Accuracy
        F1 Score
    """

    tp = sum(
        1
        for result in results
        if result["result"] == "TP"
    )
    tn = sum(
        1
        for result in results
        if result["result"] == "TN"
    )
    fp = sum(
        1
        for result in results
        if result["result"] == "FP"
    )

    fn = sum(
        1
        for result in results
        if result["result"] == "FN"
    )


    # ========================================================
    # Precision
    # ========================================================

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    # ========================================================
    # Recall
    # ========================================================

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    # ========================================================
    # Accuracy
    # ========================================================

    accuracy = (
        (tp + tn) / len(results)
        if results
        else 0.0
    )


    # ========================================================
    # F1 Score
    # ========================================================

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(
            precision,
            3
        ),

        "recall": round(
            recall,
            3
        ),

        "f1_score": round(
            f1,
            3
        ),

        "accuracy": round(
            accuracy,
            3
        ),

        "total": len(results),

        "correct": tp + tn,

    }