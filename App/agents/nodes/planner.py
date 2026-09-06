import logfire

from App.agents.states import AgentState
from App.gateways.client import get_langchain_llm


llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    """
    Determines whether the user needs:

    1. CRISIS routing
    2. CONVERSATIONAL emotional support
    3. RETRIEVAL for explicit information/resources
    """

    messages = state.get("messages", [])

    history = ""

    # Build conversation history
    for msg in messages[:-1]:

        if msg.type == "human":
            role = "User"
        elif msg.type == "ai":
            role = "ZENO"
        else:
            role = "System"

        history += f"{role}: {msg.content}\n"

    # Latest user message
    user_message = (
        messages[-1].content
        if messages
        else ""
    )

    prompt = f"""
You are the routing planner for ZENO, a conversational mental wellbeing companion.

Your PRIMARY goal is to preserve a natural human-like conversation.

Analyze the latest user message in the context of the conversation.

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE:
"{user_message}"

You must choose ONE of these outputs:

CRISIS

CONVERSATIONAL

or a concise SEARCH QUERY.

IMPORTANT ROUTING RULES:

1. CRISIS

Output CRISIS if the user expresses:

- wanting to die
- suicidal intent
- self-harm intent
- a plan to harm themselves
- intent to harm someone else
- immediate danger

If unsure about immediate safety risk, choose CRISIS.


2. CONVERSATIONAL

This should be your DEFAULT.

Output CONVERSATIONAL when the user is:

- sharing feelings
- feeling lonely
- feeling distressed
- feeling sad
- feeling anxious
- venting
- talking about their life
- answering a previous question
- continuing an emotional conversation
- asking you to talk with them
- reflecting on something
- describing work stress
- describing project stress
- saying they feel like a failure
- saying they feel stuck
- saying they feel like they are going nowhere
- saying they don't have anyone

Short replies in an existing emotional conversation should also be:

CONVERSATIONAL

Examples:

"I feel lonely"
→ CONVERSATIONAL

"work stress"
→ CONVERSATIONAL

"projects"
→ CONVERSATIONAL

"I feel like I'm failing"
→ CONVERSATIONAL

"I don't have anyone"
→ CONVERSATIONAL

"Can you talk to me?"
→ CONVERSATIONAL

"What should I do?"
→ CONVERSATIONAL


3. SEARCH QUERY

ONLY use a search query when the user explicitly asks for factual
information, educational information, or external resources.

Examples:

"What are symptoms of anxiety?"
→ symptoms of anxiety

"What is CBT?"
→ cognitive behavioral therapy

"Explain depression"
→ depression information

"Find a therapist"
→ mental health therapist

"Find a support group"
→ mental health support groups


DEFAULT RULE:

When uncertain between CONVERSATIONAL and SEARCH QUERY,
ALWAYS choose CONVERSATIONAL.

The user is talking to a companion, not a search engine.

Output ONLY one of:

CRISIS

CONVERSATIONAL

or the concise search query.

No explanation.
No punctuation.
No quotes.
"""

    with logfire.span("Planner Decision"):

        decision = llm.invoke(prompt).content.strip()

        logfire.info(
            f"Planner decision: {decision}"
        )

    decision_upper = decision.upper()


    # -----------------------------
    # CRISIS
    # -----------------------------

    if decision_upper == "CRISIS":

        return {
            "current_query": "CRISIS",
            "status": "Crisis detected — routing to safety support.",
            "plan": [
                "Intent: Crisis",
                "Retrieval: Skipped",
                "Response: Safety support"
            ]
        }


    # -----------------------------
    # CONVERSATIONAL
    # -----------------------------

    if decision_upper == "CONVERSATIONAL":

        return {
            "current_query": "CONVERSATIONAL",
            "status": "Continuing supportive conversation.",
            "plan": [
                "Intent: Conversational",
                "Retrieval: Skipped"
            ]
        }


    # -----------------------------
    # RETRIEVAL
    # -----------------------------

    return {
        "current_query": decision,
        "status": "Looking for relevant information.",
        "plan": [
            "Intent: Information/Resource Request",
            f"Search Term: {decision}"
        ]
    }