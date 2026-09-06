from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from App.config import settings
from App.agents.states import AgentState

from App.agents.nodes.planner import planner_node
from App.agents.nodes.retriever import retrieve_node
from App.agents.nodes.responder import generate_node
from App.agents.nodes.crisis import crisis_node


# ============================================================
# Neon PostgreSQL persistent checkpointer
# ============================================================

checkpointer_cm = PostgresSaver.from_conn_string(
    settings.DATABASE_URL
)

checkpointer = checkpointer_cm.__enter__()

# Creates LangGraph checkpoint tables if they don't exist
checkpointer.setup()


# ============================================================
# Build workflow
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("crisis", crisis_node)


# ============================================================
# Routing
# ============================================================

def route_planner(state: AgentState):
    """
    Route based on the planner decision.
    """

    decision = state["current_query"]

    if decision == "CRISIS":
        return "crisis"

    if decision == "CONVERSATIONAL":
        return "responder"

    return "retriever"


# ============================================================
# Graph connections
# ============================================================

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "crisis": "crisis",
        "responder": "responder",
        "retriever": "retriever",
    },
)

workflow.add_edge("retriever", "responder")

workflow.add_edge("responder", END)
workflow.add_edge("crisis", END)


# ============================================================
# Compile with Neon persistence
# ============================================================

rag_agent = workflow.compile(
    checkpointer=checkpointer
)