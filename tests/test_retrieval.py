import json
from unittest.mock import MagicMock

from App.agents.nodes.retriever import (
    retrieve_node,
    _create_cache_key,
)


# TEST 1: CACHE KEY SHOULD BE CONSISTENT

def test_cache_key_same_for_same_query():
    key1 = _create_cache_key(
        "What is anxiety?"
    )
    key2 = _create_cache_key(
        "what is anxiety?"
    )
    assert key1 == key2

# TEST 2: CACHE KEY SHOULD IGNORE EXTRA SPACES

def test_cache_key_ignores_spaces():
    key1 = _create_cache_key(
        "What is anxiety?"
    )
    key2 = _create_cache_key(
        "   What is anxiety?   "
    )
    assert key1 == key2


# TEST 3: REDIS CACHE HIT

def test_retrieve_cache_hit(monkeypatch):
    # Fake cached documents
    cached_documents = [
        "CONTENT: Anxiety is a feeling of worry.",
        "CONTENT: Stress can affect mental wellbeing."
    ]
    # Mock Redis GET
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps(
        cached_documents
    )
    # Replace real Redis client
    monkeypatch.setattr(
        "App.agents.nodes.retriever.redis_client",
        mock_redis
    )
    state = {
        "current_query": "What is anxiety?",
        "documents": [],
        "plan": [
            "Intent: Information/Resource Request"
        ]
    }
    result = retrieve_node(state)
    # Check Redis was called
    mock_redis.get.assert_called_once()
    # Check cached documents returned
    assert result["documents"] == cached_documents
    # Check cache status
    assert (
        "Retrieval Cache: Hit ⚡"
        in result["plan"]
    )

# TEST 4: REDIS CACHE MISS
def test_retrieve_cache_miss(monkeypatch):
    # Mock Redis
    mock_redis = MagicMock()
    # Redis returns nothing
    mock_redis.get.return_value = None
    monkeypatch.setattr(
        "App.agents.nodes.retriever.redis_client",
        mock_redis
    )
    # Mock Qdrant search
    fake_qdrant_results = [
        {
            "content": "Anxiety can involve feelings of worry."
        },
        {
            "content": "Stress may affect emotional wellbeing."
        }
    ]
    monkeypatch.setattr(
        "App.agents.nodes.retriever.search_enterprise_knowledge",
        lambda query, limit: fake_qdrant_results
    )
    # Mock reranking
    fake_reranked_documents = [
        "Anxiety can involve feelings of worry."
    ]
    monkeypatch.setattr(
        "App.agents.nodes.retriever.rerank_documents",
        lambda query, documents, top_n:
        fake_reranked_documents
    )
    state = {
        "current_query": "What is anxiety?",
        "documents": [],
        "plan": [
            "Intent: Information/Resource Request"
        ]
    }
    result = retrieve_node(state)
    # Redis should be checked
    mock_redis.get.assert_called_once()
    # Redis should store result after cache miss
    mock_redis.set.assert_called_once()
    # Check documents
    assert result["documents"] == [
        "CONTENT: Anxiety can involve feelings of worry."
    ]
    # Check plan
    assert "Context Retrieved" in result["plan"]
    assert "Retrieval Cache: Stored" in result["plan"]

# TEST 5: REDIS FAILURE SHOULD NOT BREAK RETRIEVAL

def test_redis_failure_does_not_break_retrieval(monkeypatch):
    mock_redis = MagicMock()
    # Cache miss
    mock_redis.get.return_value = None
    # Redis SET fails
    mock_redis.set.side_effect = Exception(
        "Redis connection failed"
    )
    monkeypatch.setattr(
        "App.agents.nodes.retriever.redis_client",
        mock_redis
    )
    # Mock Qdrant
    fake_qdrant_results = [
        {
            "content": "Stress can affect wellbeing."
        }
    ]
    monkeypatch.setattr(
        "App.agents.nodes.retriever.search_enterprise_knowledge",
        lambda query, limit: fake_qdrant_results
    )
    # Mock reranker
    monkeypatch.setattr(
        "App.agents.nodes.retriever.rerank_documents",
        lambda query, documents, top_n: documents
    )
    state = {
        "current_query": "stress",
        "documents": [],
        "plan": [
            "Intent: Information/Resource Request"
        ]
    }
    # This should NOT crash
    result = retrieve_node(state)
    # Retrieval should still succeed
    assert result["documents"] == [
        "CONTENT: Stress can affect wellbeing."
    ]
    assert "Context Retrieved" in result["plan"]