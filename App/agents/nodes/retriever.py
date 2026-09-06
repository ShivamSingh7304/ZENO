import hashlib
import json
import logfire

from App.agents.states import AgentState
from App.services.retrieval.qdrant_service import search_enterprise_knowledge
from App.services.retrieval.ranking_service import rerank_documents
from App.services.cache.redis_service import redis_client

def _create_cache_key(query: str) -> str:
    """
    Creates a consistent Redis key for a search query.
    """
    normalized_query = query.strip().lower()
    query_hash = hashlib.sha256(
        normalized_query.encode("utf-8")
    ).hexdigest()
    return f"zeno:retrieval:{query_hash}"

def retrieve_node(state: AgentState):
    """
    Performs:
    1. Redis cache lookup
    2. Qdrant vector search if cache misses
    3. Semantic reranking
    4. Stores results in Redis for future reuse
    """

    query = state["current_query"]
    cache_key = _create_cache_key(query)

   
    # STEP 1: CHECK REDIS CACHE
   
    with logfire.span("Retrieval Cache Check"):

        cached_result = redis_client.get(cache_key)
        if cached_result:
            logfire.info(
                f"Redis Cache HIT for retrieval query: {query}"
            )
            documents = json.loads(cached_result)
            return {
                "documents": documents,
                "status": "Retrieved context from Redis cache.",
                "plan": state["plan"] + [
                    "Retrieval Cache: Hit ⚡"
                ]
            }
        logfire.info(
            f"Redis Cache MISS for retrieval query: {query}"
        )

    
    # STEP 2: QDRANT VECTOR SEARCH

    with logfire.span("Knowledge Retrieval"):

        logfire.info(
            f"Searching Qdrant for: {query}"
        )

        raw_results = search_enterprise_knowledge(
            query,
            limit=15
        )

        logfire.info(
            f"Retrieved {len(raw_results)} candidates from Vector DB"
        )

        # Extract document text
        doc_contents = [
            doc["content"]
            for doc in raw_results
        ]

        # STEP 3: SEMANTIC RERANKING

        with logfire.span("Semantic Reranking"):

            reranked_contents = rerank_documents(
                query,
                doc_contents,
                top_n=5
            )

            logfire.info(
                "Reranking complete. Kept top 5 relevant chunks."
            )

        # Format documents for LLM
        formatted_docs = [
            f"CONTENT: {doc}"
            for doc in reranked_contents
        ]

    
    # STEP 4: SAVE RESULTS TO REDIS

    with logfire.span("Saving Retrieval Cache"):

        try:

            redis_client.set(
                cache_key,
                json.dumps(formatted_docs),
                ex=3600
            )

            logfire.info(
                "Stored retrieval results in Redis cache. TTL: 1 hour."
            )

        except Exception as e:

            # Redis failure should never break retrieval
            logfire.warning(
                f"Failed to store retrieval cache: {e}"
            )

    # STEP 5: RETURN RESULTS

    return {
        "documents": formatted_docs,
        "status": "Found relevant context.",
        "plan": state["plan"] + [
            "Context Retrieved",
            "Retrieval Cache: Stored"
        ]
    }