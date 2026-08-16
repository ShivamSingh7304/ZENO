from App.agents.states import AgentState
from langchain_groq import ChatGroq
from App.config import settings
import logfire 

llm = ChatGroq(api_key=settings.GROQ_API_KEY , model=settings.LLM_MODEL)

def generate_node(state: AgentState):
    '''
    Synthesizes a response using both Documentation Context AND Conversation History.
    '''

    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""


    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational (companion) response using memory.")
        prompt = f"""
        You are a warm, supportive companion for a mental wellbeing app.

        Guidelines (in priority order):
        1. Safety first: if anything in the LATEST MESSAGE suggests distress escalating
        toward crisis, do not attempt to handle it yourself — this should already be
        caught upstream, but if you sense risk, respond with care, avoid problem-solving
        or cheerful redirection, and gently encourage the person to reach out to a
        trusted person or professional.
        2. Validate feelings without judgment. Reflect back what the person is
        expressing before offering anything else.
        3. Keep language calm, plain, and conversational — no clinical jargon, no
        diagnostic language, no labels the person hasn't used themselves.
        4. Do not offer medical, therapeutic, or diagnostic advice. You may offer
        1-2 gentle, general coping ideas if it fits naturally — never prescriptive.
        5. When unsure what the person needs, ask rather than assume.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating resource-grounded (companion) response.")
        max_context_chars = 25000
        full_context = ""
        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a compassionate mental wellness resource assistant.

        Guidelines (in priority order):
        1. Ground every claim in the SUPPORT CONTEXT below. Do not invent clinical
        facts, statistics, hotline numbers, or provider details not present in it.
        2. If the SUPPORT CONTEXT doesn't fully answer the question, say so plainly
        instead of guessing or filling gaps with assumptions.
        3. Present information gently and clearly — avoid alarming, clinical, or
        overly technical language.
        4. Never give specific medical directives (dosages, diagnoses, treatment
        decisions). Encourage professional consultation for anything beyond
        general information.
        5. If the SUPPORT CONTEXT includes crisis resources (hotlines, emergency
        contacts) and the question relates to safety, prioritize surfacing those
        clearly and accurately over other information.

        SUPPORT CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("LLM Synthesised"):
        try:
            content = llm.invoke(prompt).content
            logfire.info("Response synthesised via LLM")

            return {
                "final_answer": content,
                "status":"Response generated",
                "plan":state["plan"],
                "messages":[{"role":"Assistant" , "content":content}]
            }
        except Exception as e:
            logfire.error(f"LLM resonse generation failed : {e}")