import logfire

from App.agents.states import AgentState


def crisis_node(state: AgentState):
    """
    Handles messages classified by the planner as CRISIS.

    This bypasses:
    - Qdrant retrieval
    - Jina reranking
    - Normal LLM generation

    A fixed response is used so a crisis request does not depend on
    external retrieval or an LLM call succeeding.
    """

    logfire.warning(
        "CRISIS ROUTE ACTIVATED",
        query=state["messages"][-1]["content"][:200]
        if state.get("messages")
        else ""
    )

    crisis_response = """
I'm really sorry you're going through this. You don't have to handle this alone.

If you feel that you might hurt yourself or someone else, or if you're in immediate danger, please contact your local emergency services or go to the nearest emergency department now.

If possible, please also reach out to someone you trust and let them know that you need support. You could simply say: "I'm not feeling safe being alone right now. Can you stay with me or talk to me?"

If you can, move away from anything you could use to hurt yourself and stay somewhere with other people.

I'm here with you. If you're able to, tell me: are you in immediate danger right now?
""".strip()

    return {
        "final_answer": crisis_response,
        "status": "Crisis support response provided.",
        "plan": state["plan"] + [
            "Crisis route activated",
            "Retrieval: Skipped",
            "LLM Generation: Skipped",
            "Response: Crisis safety template",
        ],
        "messages": [
            {
                "role": "assistant",
                "content": crisis_response,
            }
        ],
    }