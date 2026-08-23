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
    Initialize the guardrails classifier LLM.

    Kept the name `initialize_rails` (and `guard()`'s signature below)
    unchanged from the previous NeMo-based implementation so App/main.py
    and anything else importing from App.guardrails needs no changes —
    this is a drop-in internal replacement, not a public interface change.
    """

    global _classifier_llm

    try:
        _classifier_llm = get_langchain_llm(feature="guardrails")

        logfire.info(
            "Guardrails classifier initialized (direct Groq/Portkey, "
            "no NeMo)."
        )

    except Exception as e:
        _classifier_llm = None

        logfire.error(
            f"Guardrails classifier initialization failed: {e}"
        )

        raise


async def guard(message: str) -> tuple[bool, str | None]:
    """
    Run the message through ZENO's guardrails.

    Two layers:
      1. Deterministic keyword pre-check for jailbreak phrasing (fast,
         no model call, catches obvious formulaic attempts).
      2. LLM-based structured classification for everything else
         (off_topic / jailbreak / greeting / capabilities / farewell / safe).

    Returns:
        True, response
            A guardrail fired; return this response immediately,
            skip the RAG pipeline entirely.

        False, None
            Message is safe, OR the gate could not run; proceed to
            LangGraph either way (fail-open — see rationale below).

    Fail-open rationale: if the classifier is unavailable or errors out,
    we let the message through rather than blocking it — a Groq hiccup
    should never stop someone in distress from reaching the crisis-routing
    planner downstream. Fail-open paths log at `error` level so this is
    visible in production rather than blending into normal info-level noise.
    """

    # ----------------------------------------------------------------
    # Layer 1: deterministic jailbreak keyword pre-check
    # ----------------------------------------------------------------
    if keyword_jailbreak_check(message):
        logfire.warning(
            f"GUARDRAIL FIRED (keyword pre-check) | "
            f"intent=jailbreak | query='{message[:100]}'"
        )
        return True, GUARD_RESPONSES[GuardIntent.JAILBREAK]

    # ----------------------------------------------------------------
    # Layer 2: LLM structured classification
    # ----------------------------------------------------------------
    if _classifier_llm is None:
        logfire.error(
            "Guardrails classifier not initialized — failing open, "
            "skipping gate."
        )
        return False, None

    with logfire.span("Guardrails Check"):

        try:
            result = await _classifier_llm.ainvoke(
                [
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ]
            )

            raw_content = (
                result.content if hasattr(result, "content") else str(result)
            )

        except Exception as e:
            logfire.error(
                f"Guardrails classification call failed — failing open: {e}"
            )
            return False, None

        logfire.info(f"Classifier raw response: {raw_content!r}")

        # --------------------------------------------------
        # Parse the structured JSON response
        # --------------------------------------------------
        try:
            # Defensive: strip markdown code fences if the model added
            # them despite instructions not to.
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            parsed = json.loads(cleaned)
            intent_str = parsed.get("intent", "safe").lower().strip()
            intent = GuardIntent(intent_str)

        except (json.JSONDecodeError, ValueError) as e:
            logfire.error(
                f"Guardrails classifier returned unparseable output "
                f"({e}) — failing open. raw='{raw_content[:200]}'"
            )
            return False, None

        # --------------------------------------------------
        # Route based on classified intent
        # --------------------------------------------------
        if intent == GuardIntent.SAFE:
            logfire.info(f"Guardrails passed | query='{message[:100]}'")
            return False, None

        response = GUARD_RESPONSES.get(intent)

        if response is None:
            # Shouldn't happen given the enum, but fail open defensively
            # rather than returning a blank response to the user.
            logfire.error(
                f"Guardrails classified intent='{intent}' but no response "
                f"is defined for it — failing open."
            )
            return False, None

        logfire.warning(
            f"GUARDRAIL FIRED | intent={intent.value} | "
            f"query='{message[:100]}'"
        )

        return True, response