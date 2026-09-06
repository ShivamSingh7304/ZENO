import os
import logfire
import requests

from App.config import settings


# =============================================================================
# JINA EMBEDDINGS CONFIGURATION
# =============================================================================

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

JINA_MODEL = "jina-embeddings-v3"

# Jina embeddings v3 default dimension
EMBEDDING_DIM = 1024

BATCH_SIZE = 32

_request_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {settings.JINA_API_KEY}",
}

# =============================================================================
# GET EMBEDDING DIMENSION
# =============================================================================

def get_embedding_dim() -> int:
    """
    Returns the embedding dimension used by the application.
    """
    return EMBEDDING_DIM

# =============================================================================
# INTERNAL JINA API CALL
# =============================================================================

def _get_embeddings(
    texts: list[str],
    task: str
) -> list[list[float]]:
    """
    Sends texts to Jina Embeddings API and returns embedding vectors.

    task:
        retrieval.query
        retrieval.passage
    """

    if not texts:
        return []

    payload = {
        "model": JINA_MODEL,
        "input": texts,
        "task": task,
    }

    try:

        with logfire.span(
            "Jina Embeddings API",
            model=JINA_MODEL,
            task=task,
            batch_size=len(texts),
        ):

            response = requests.post(
                JINA_API_URL,
                headers=_request_headers,
                json=payload,
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            embeddings = [
                item["embedding"]
                for item in data["data"]
            ]

            logfire.info(
                "Jina embeddings generated",
                model=JINA_MODEL,
                count=len(embeddings),
            )

            return embeddings

    except requests.exceptions.RequestException as e:

        logfire.error(
            f"Jina embedding request failed: {e}"
        )

        raise


# =============================================================================
# EMBED QUERY
# =============================================================================

def embed_query(query: str) -> list[float]:
    """
    Generates an embedding optimized for a retrieval query.
    """

    embeddings = _get_embeddings(
        texts=[query],
        task="retrieval.query",
    )

    return embeddings[0]


# =============================================================================
# EMBED DOCUMENTS
# =============================================================================

def embed_texts(
    texts: list[str]
) -> list[list[float]]:
    """
    Generates embeddings optimized for document retrieval.

    Processes texts in batches.
    """

    all_embeddings = []

    for i in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            i:i + BATCH_SIZE
        ]

        with logfire.span(
            "Embed Document Batch",
            model=JINA_MODEL,
            start=i,
            size=len(batch),
        ):

            embeddings = _get_embeddings(
                texts=batch,
                task="retrieval.passage",
            )

            all_embeddings.extend(
                embeddings
            )

    return all_embeddings