import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from App.config import settings
from App.guardrails.guard_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-20b for fast intent classification at the gate —
    the heavier openai/gpt-oss-20b is reserved for the RAG pipeline.
    """
    global _rails
    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0
    )
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )
    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(" NeMo Guardrails initialised (openai/gpt-oss-20b).")


async def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                  skip the RAG pipeline entirely.
        (False, None)          — message is clean, OR the gate could not run;
                                  proceed to LangGraph either way (fail-open).

    Fail-open rationale: if the gate is unavailable or errors out, we let the
    message through rather than blocking it — a Groq hiccup should never stop
    someone in distress from reaching the crisis-routing planner downstream.
    The tradeoff is that jailbreak/off-topic protection goes dark during that
    window, so both fail-open paths log at `error` level to make that visible
    in production rather than blending into normal info-level noise.
    """
    if _rails is None:
        logfire.error(" Guardrails not initialised — failing open, skipping gate.")
        return False, None

    with logfire.span(" Guardrails Check"):
        try:
            result = await _rails.generate_async(
                messages=[{"role": "user", "content": message}]
            )
        except Exception as e:
            logfire.error(f" Guardrails call failed — failing open: {e}")
            return False, None

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f" Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info(" Guardrails passed.")
        return False, None