from App.agents.states import AgentState
from langchain_groq import ChatGroq
from App.config import settings
import logfire 

llm = ChatGroq(api_key=settings.GROQ_API_KEY , model=settings.LLM_MODEL)

def planner_node(state: AgentState):
    '''
    The Planner determines if a search is needed based on the ENTIRE conversation.
    '''

    history= ""

    for msg in state['messages'][:-1]:
        role = "User" if msg["role"] == "user " else "Assistant"
        history += f"{role}: {msg["content"]}\n"

        user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner for a mental wellbeing companion app.
    Analyze the conversation history and the latest user message.
    CONVERSATION HISTORY:
    {history}
    LATEST MESSAGE:
    "{user_message}"
    Task (check in this order):
    1. CRISIS CHECK (highest priority — overrides everything below):
    If the latest message contains ANY sign of crisis — suicidal ideation, 
    self-harm intent or plans, hopelessness combined with wanting to die or 
    disappear, or intent to harm others — output 'CRISIS'.
    This applies even if such signs appear alongside unrelated or casual content,
    are stated indirectly, or are mixed with a request for something else.
    If you are unsure whether something qualifies, treat it as CRISIS.
    2. CONVERSATIONAL:
    If the message is a greeting, small talk, emotional check-in, venting, or a
    question answerable using ONLY the conversation history above (e.g., "what
    did I say I was struggling with"), output 'CONVERSATIONAL'.
    3. SEARCH QUERY:
    If it asks for fresh, external, or local information (e.g., "find a
    therapist near me", "grounding exercise for panic attacks", "support groups
    in my city", "crisis hotline number"), output a refined, concise search query.
    Output ONLY one of: 'CRISIS', 'CONVERSATIONAL', or the search query.
    No punctuation, quotes, explanation, or extra text.
    """

    with logfire.span(" Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CRISIS":
        logfire.warning("Crisis signal detected in user message")
        return {
            "current_query": "CRISIS",
            "status": "Crisis detected — routing to safety resources...",
            "plan": ["Intent: Crisis", "Retrieval: Skipped", "Response: Fixed crisis-resource template"]
        }

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Reflecting with you (using our conversation so far)...",
            "plan": ["Intent: Conversational/Reflective", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Looking that up for you: {decision}",
        "plan": ["Intent: Resource/Search", f"Search Term: {decision}"]
}