import json
import logfire

from App.guardrails.guard_rules import (
    GuardIntent,
    GUARD_RESPONSES,
    CLASSIFIER_SYSTEM_PROMPT,
    keyword_jailbreak_check,
)

from App.gateways.client import get_langchain_llm


_classifier_llm = None


def initialize_rails() -> None:
    """
    Initialize the guardrails classifier.
    """

    global _classifier_llm

    try:
        _classifier_llm = get_langchain_llm(feature="guardrails")

        logfire.info(
            "Guardrails classifier initialized successfully"
        )

    except Exception as e:
        _classifier_llm = None

        logfire.error(
            f"Guardrails classifier initialization failed: {e}"
        )

        raise


async def guard(message: str) -> tuple[bool, str | None, str]:
    """
    Run message through ZENO-AI guardrails.

    Returns:
        rail_fired
        response
        intent

    Example:

        True, "response...", "jailbreak"

        False, None, "safe"

        False, None, "unknown"
    """

    # ============================================================
    # Layer 1: Deterministic jailbreak keyword check
    # ============================================================

    if keyword_jailbreak_check(message):

        intent = GuardIntent.JAILBREAK

        logfire.warning(
            f"GUARDRAIL FIRED (keyword pre-check) | "
            f"intent={intent.value} | "
            f"query='{message[:100]}'"
        )

        return (
            True,
            GUARD_RESPONSES[intent],
            intent.value,
        )

    # ============================================================
    # Layer 2: LLM classification
    # ============================================================

    if _classifier_llm is None:

        logfire.error(
            "Guardrails classifier not initialized — "
            "failing open."
        )

        return False, None, "unknown"

    with logfire.span("Guardrails Check"):

        try:

            result = await _classifier_llm.ainvoke(
                [
                    {
                        "role": "system",
                        "content": CLASSIFIER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ]
            )

            raw_content = (
                result.content
                if hasattr(result, "content")
                else str(result)
            )

        except Exception as e:

            logfire.error(
                f"Guardrails classification failed — "
                f"failing open: {e}"
            )

            return False, None, "unknown"

        logfire.info(
            f"Classifier raw response: {raw_content!r}"
        )

        # ========================================================
        # Parse JSON
        # ========================================================

        try:

            cleaned = raw_content.strip()

            if cleaned.startswith("```"):

                cleaned = cleaned.strip("`")

                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            parsed = json.loads(cleaned)

            intent_str = (
                parsed
                .get("intent", "safe")
                .lower()
                .strip()
            )

            intent = GuardIntent(intent_str)

        except (json.JSONDecodeError, ValueError) as e:

            logfire.error(
                f"Guardrails classifier returned invalid output: {e}"
            )

            return False, None, "unknown"

        # ========================================================
        # SAFE
        # ========================================================

        if intent == GuardIntent.SAFE:

            logfire.info(
                f"Guardrails passed | "
                f"intent=safe | "
                f"query='{message[:100]}'"
            )

            return False, None, "safe"

        # ========================================================
        # Guardrail fired
        # ========================================================

        response = GUARD_RESPONSES.get(intent)

        if response is None:

            logfire.error(
                f"No response configured for "
                f"intent={intent.value}"
            )

            return False, None, "unknown"

        logfire.warning(
            f"GUARDRAIL FIRED | "
            f"intent={intent.value} | "
            f"query='{message[:100]}'"
        )

        return (
            True,
            response,
            intent.value,
        )