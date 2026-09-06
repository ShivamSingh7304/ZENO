from typing import TypedDict, List, Optional, Annotated

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_query: str
    documents: list
    plan: List[str]
    status: str
    planner_intent: Optional[str]
    final_answer: Optional[str]