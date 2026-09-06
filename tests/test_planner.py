from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

import App.agents.nodes.planner as planner_module


def create_state(message: str):

    return {
        "messages": [
            HumanMessage(content=message)
        ],
        "current_query": "",
        "documents": [],
        "plan": [],
        "status": "",
        "planner_intent": None,
        "final_answer": None
    }


# TEST CONVERSATIONAL ROUTING

def test_conversational_routing(monkeypatch):
    mock_response = MagicMock()
    mock_response.content = "CONVERSATIONAL"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    monkeypatch.setattr(
        planner_module,
        "llm",
        mock_llm
    )
    state = create_state(
        "I feel stressed about my projects"
    )
    result = planner_module.planner_node(state)
    assert result["current_query"] == "CONVERSATIONAL"
    assert result["status"] == (
        "Continuing supportive conversation."
    )
    assert "Intent: Conversational" in result["plan"]


# TEST CRISIS ROUTING

def test_crisis_routing(monkeypatch):
    mock_response = MagicMock()
    mock_response.content = "CRISIS"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    monkeypatch.setattr(
        planner_module,
        "llm",
        mock_llm
    )
    state = create_state(
        "I want to harm myself"
    )
    result = planner_module.planner_node(state)
    assert result["current_query"] == "CRISIS"
    assert result["status"] == (
        "Crisis detected — routing to safety support."
    )
    assert "Intent: Crisis" in result["plan"]

# TEST RETRIEVAL ROUTING

def test_retrieval_routing(monkeypatch):
    mock_response = MagicMock()
    mock_response.content = "symptoms of anxiety"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    monkeypatch.setattr(
        planner_module,
        "llm",
        mock_llm
    )
    state = create_state(
        "What are the symptoms of anxiety?"
    )
    result = planner_module.planner_node(state)
    assert result["current_query"] == (
        "symptoms of anxiety"
    )
    assert result["status"] == (
        "Looking for relevant information."
    )
    assert (
        "Intent: Information/Resource Request"
        in result["plan"]
    )

# TEST WHITESPACE HANDLING

def test_decision_whitespace(monkeypatch):
    mock_response = MagicMock()
    mock_response.content = "  CONVERSATIONAL\n\n"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    monkeypatch.setattr(
        planner_module,
        "llm",
        mock_llm
    )
    state = create_state(
        "I feel lonely"
    )
    result = planner_module.planner_node(state)
    assert result["current_query"] == "CONVERSATIONAL"