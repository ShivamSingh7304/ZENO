import logfire

from langchain_core.messages import AIMessage

from App.agents.states import AgentState
from App.gateways.client import (
    portkey_client,
    extract_cache_status
)


def generate_node(state: AgentState):

    query = state["current_query"]

    # ============================================================
    # BUILD CONVERSATION HISTORY
    # ============================================================

    graph_messages = state.get("messages", [])

    history_str = ""

    for msg in graph_messages[:-1]:

        if msg.type == "human":
            role = "User"

        elif msg.type == "ai":
            role = "ZENO"

        else:
            role = "System"

        history_str += f"{role}: {msg.content}\n"


    # ============================================================
    # GET LATEST USER MESSAGE
    # ============================================================

    user_msg = (
        graph_messages[-1].content
        if graph_messages
        else ""
    )


    # ============================================================
    # CONVERSATIONAL MODE
    # ============================================================

    if query == "CONVERSATIONAL":

        logfire.info(
            "Generating conversational companion response using memory."
        )

        prompt = f"""
You are ZENO, a warm and emotionally intelligent mental wellbeing companion.

Your role is to have a natural, human-like supportive conversation.

You are NOT:

- a questionnaire
- a customer support bot
- a search engine
- a therapist conducting an assessment
- a chatbot that asks a question after every message

CONVERSATION RULES:

1. RESPOND TO THE EMOTION FIRST.

Before giving advice or solutions, acknowledge what the person seems
to be feeling.

2. DO NOT ASK A QUESTION EVERY TURN.

Questions are optional.

Sometimes simply reflect the user's experience.

Do not automatically end every response with a question.

3. USE THE CONVERSATION HISTORY.

Treat the conversation as connected.

Do not treat every new message as an isolated request.

If the user previously mentioned loneliness, work stress, or projects,
connect those ideas naturally when relevant.

4. DO NOT RUSH TO SOLUTIONS.

Do not immediately give lists of coping strategies.

Sometimes the best response is simply to stay with what the user is
feeling.

5. SPEAK LIKE A COMPANION.

Use warm, natural language.

Avoid overly formal phrases like:

"Here are some strategies you may consider."

"It is important to recognize."

6. DO NOT ASSUME A DIAGNOSIS.

Never assume depression, anxiety, or another condition unless the user
explicitly mentions it.

7. RESPONSE LENGTH.

Usually respond in 2 to 5 sentences.

Keep the conversation natural.

CONVERSATION HISTORY:

{history_str}

LATEST USER MESSAGE:

"{user_msg}"

Respond naturally as ZENO.
"""

        temperature = 0.5


    # ============================================================
    # RAG / RESOURCE MODE
    # ============================================================

    else:

        logfire.info(
            "Generating resource-grounded companion response."
        )

        max_context_chars = 25000

        full_context = ""

        for doc in state.get("documents", []):

            if len(full_context) + len(doc) < max_context_chars:

                full_context += doc + "\n\n"

            else:

                logfire.warning(
                    "Context truncated to fit model limits."
                )

                break


        prompt = f"""
You are ZENO, a compassionate mental wellness resource assistant.

Use the SUPPORT CONTEXT to answer the user's question.

Rules:

1. Ground factual claims in the support context.

2. Do not invent medical facts, statistics, hotline numbers,
or professional details.

3. If the context does not answer the question, say so clearly.

4. Keep the tone warm and conversational.

5. Do not provide diagnoses or specific medical treatment instructions.

SUPPORT CONTEXT:

{full_context}

CONVERSATION HISTORY:

{history_str}

USER QUESTION:

"{user_msg}"

Respond naturally and clearly.
"""

        temperature = 0.1


    # ============================================================
    # PORTKEY MESSAGE PAYLOAD
    # ============================================================

    llm_messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]


    # ============================================================
    # LLM CALL
    # ============================================================

    with logfire.span("LLM Synthesised"):

        try:

            response = portkey_client.chat.completions.create(

                messages=llm_messages,

                temperature=temperature

            )

            content = response.choices[0].message.content

            cache_status = extract_cache_status(response)

            is_cache_hit = cache_status == "HIT"


            if is_cache_hit:

                logfire.info("Gateway Cache Hit")

                plan_update = state["plan"] + [
                    "Cache: Hit"
                ]

                status = "Cache hit — instant response."


            else:

                logfire.info(
                    "Response synthesised via LLM."
                )

                plan_update = state["plan"]

                status = "Response generated."


            # IMPORTANT:
            # Return LangChain AIMessage to LangGraph

            return {

                "final_answer": content,

                "status": status,

                "plan": plan_update,

                "messages": [
                    AIMessage(content=content)
                ]

            }


        except Exception as e:

            logfire.error(
                f"LLM Generation failed: {e}"
            )

            raise