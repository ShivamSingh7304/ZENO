import time
import requests
import logfire

from App.config import settings


# =============================================================================
# JINA RERANKER CONFIGURATION
# =============================================================================

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

# Good multilingual choice for a mental-health knowledge base.
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

REQUEST_TIMEOUT = 60


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[str]:
    """
    Reranks retrieved documents using the Jina Reranker API.

    Flow:
        Qdrant retrieves candidate documents
                ↓
        Jina reranks them against the user query
                ↓
        Returns the top_n most relevant documents

    If reranking fails, falls back to the original Qdrant ranking
    so the RAG pipeline can still continue.
    """

    if not documents:
        return []

    start_time = time.time()

    # Ensure top_n never exceeds the number of documents.
    top_n = min(top_n, len(documents))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.JINA_API_KEY}",
    }

    payload = {
        "model": JINA_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": True,
    }

    try:
        with logfire.span(
            "Jina Semantic Reranking",
            model=JINA_RERANK_MODEL,
            candidates=len(documents),
            top_n=top_n,
        ):

            logfire.info(
                f"Sending {len(documents)} documents to Jina Reranker"
            )

            response = requests.post(
                JINA_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            results = data.get("results", [])

            reranked_docs = []

            for result in results:
                # Jina can return the document as an object or directly
                # depending on API response formatting.
                document = result.get("document", "")

                if isinstance(document, dict):
                    text = document.get("text", "")
                else:
                    text = document

                if text:
                    reranked_docs.append(text)

            duration = time.time() - start_time

            top_score = (
                results[0].get("relevance_score")
                if results
                else "N/A"
            )

            logfire.info(
                f"Jina reranking complete in {duration:.2f}s. "
                f"Top relevance score: {top_score}"
            )

            return reranked_docs[:top_n]

    except requests.exceptions.RequestException as e:

        duration = time.time() - start_time

        logfire.error(
            f"Jina reranking failed after {duration:.2f}s: {e}"
        )

        # Graceful fallback:
        # Qdrant already returns results ordered by similarity score.
        return documents[:top_n]

    except Exception as e:

        logfire.error(
            f"Unexpected reranking error: {e}"
        )

        return documents[:top_n]