from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage

import App.agents.nodes.responder as generate_module

# TEST 1
# CONVERSATION HISTORY IS USED

def test_conversation_uses_history(monkeypatch):
    """
    Test that previous conversation messages are included
    when generating a conversational response.
    """
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="It sounds like your projects have been weighing on you."
            )
        )
    ]

    mock_create = MagicMock(
        return_value=mock_response
    )

    monkeypatch.setattr(
        generate_module.portkey_client.chat.completions,
        "create",
        mock_create
    )

    monkeypatch.setattr(
        generate_module,
        "extract_cache_status",
        lambda response: "MISS"
    )

    state = {
        "messages": [
            HumanMessage(
                content="I feel stressed about my projects"
            ),
            AIMessage(
                content="That sounds difficult. What about the projects feels stressful?"
            ),
            HumanMessage(
                content="I feel like I'm failing"
            ),
        ],

        "current_query": "CONVERSATIONAL",
        "documents": [],
        "plan": [],
        "status": "",
        "planner_intent": None,
        "final_answer": None,
    }

    result = generate_module.generate_node(state)
    assert result["final_answer"] is not None
    assert "messages" in result
    assert mock_create.called

# TEST 2
# HISTORY IS SENT TO THE LLM PROMPT

def test_conversation_sends_history_to_llm(monkeypatch):
    """
    Verify that conversation history appears in the prompt
    sent to the LLM.
    """

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="I'm here with you."
            )
        )
    ]
    mock_create = MagicMock(
        return_value=mock_response
    )
    monkeypatch.setattr(
        generate_module.portkey_client.chat.completions,
        "create",
        mock_create
    )
    monkeypatch.setattr(
        generate_module,
        "extract_cache_status",
        lambda response: "MISS"
    )
    state = {
        "messages": [
            HumanMessage(
                content="I feel lonely"
            ),
            AIMessage(
                content="That sounds really hard."
            ),
            HumanMessage(
                content="Work stress is making it worse"
            ),
        ],

        "current_query": "CONVERSATIONAL",
        "documents": [],
        "plan": [],
        "status": "",
        "planner_intent": None,
        "final_answer": None,
    }

    generate_module.generate_node(state)

    # Get arguments passed to the LLM
    call_args = mock_create.call_args
    messages_sent_to_llm = call_args.kwargs["messages"]
    prompt = messages_sent_to_llm[0]["content"]
    # Verify previous conversation exists in prompt
    assert "I feel lonely" in prompt
    assert "That sounds really hard." in prompt
    # Latest message should also exist
    assert "Work stress is making it worse" in prompt


# TEST 3
# AI RESPONSE IS ADDED AS AN AIMessage

def test_conversation_response_is_added_to_messages(monkeypatch):
    """
    Test that the generated AI response is returned
    as a LangChain AIMessage.
    """
    mock_response = MagicMock()
    expected_response = (
        "That sounds like a lot to carry right now."
    )
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=expected_response
            )
        )
    ]
    mock_create = MagicMock(
        return_value=mock_response
    )
    monkeypatch.setattr(
        generate_module.portkey_client.chat.completions,
        "create",
        mock_create
    )
    monkeypatch.setattr(
        generate_module,
        "extract_cache_status",
        lambda response: "MISS"
    )
    state = {
        "messages": [
            HumanMessage(
                content="I feel overwhelmed"
            ),
        ],
        "current_query": "CONVERSATIONAL",
        "documents": [],
        "plan": [],
        "status": "",
        "planner_intent": None,
        "final_answer": None,
    }

    result = generate_module.generate_node(state)

    # VERIFY MESSAGE RETURNED

    assert "messages" in result
    assert len(result["messages"]) == 1
    assistant_message = result["messages"][0]
    # The responder now returns a LangChain AIMessage
    assert isinstance(
        assistant_message,
        AIMessage
    )
    assert assistant_message.type == "ai"
    assert (
        assistant_message.content
        == expected_response
    )

# TEST 4
# CONVERSATIONAL MODE DOES NOT USE RETRIEVAL DOCUMENTS

def test_conversation_does_not_use_retrieval_documents(monkeypatch):
    """
    Test that conversational responses do not depend on
    retrieval documents.
    """
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="I'm here with you. That sounds difficult."
            )
        )
    ]
    mock_create = MagicMock(
        return_value=mock_response
    )
    monkeypatch.setattr(
        generate_module.portkey_client.chat.completions,
        "create",
        mock_create
    )
    monkeypatch.setattr(
        generate_module,
        "extract_cache_status",
        lambda response: "MISS"
    )
    state = {
        "messages": [
            HumanMessage(
                content="I feel stressed"
            ),
        ],

        "current_query": "CONVERSATIONAL",
        # Deliberately include retrieval documents
        "documents": [
            "Some retrieved medical document",
            "Another knowledge base document",
        ],
        "plan": [],

        "status": "",
        "planner_intent": None,
        "final_answer": None,
    }

    generate_module.generate_node(state)
    call_args = mock_create.call_args
    messages_sent_to_llm = call_args.kwargs["messages"]
    prompt = messages_sent_to_llm[0]["content"]
    # Conversational prompt should not use retrieval documents
    assert "Some retrieved medical document" not in prompt
    assert "Another knowledge base document" not in prompt

# OPTIONAL TEST 5
# CACHE STATUS

def test_conversation_cache_hit(monkeypatch):
    """
    Test that a cache hit updates the plan correctly.
    """
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="I'm here with you."
            )
        )
    ]
    monkeypatch.setattr(
        generate_module.portkey_client.chat.completions,
        "create",
        MagicMock(return_value=mock_response)
    )
    monkeypatch.setattr(
        generate_module,
        "extract_cache_status",
        lambda response: "HIT"
    )
    state = {
        "messages": [
            HumanMessage(
                content="I feel stressed"
            ),
        ],
        "current_query": "CONVERSATIONAL",
        "documents": [],
        "plan": [
            "Intent: Conversational"
        ],
        "status": "",
        "planner_intent": None,
        "final_answer": None,
    }

    result = generate_module.generate_node(state)
    assert result["status"] == "Cache hit — instant response."
    assert "Cache: Hit" in result["plan"]