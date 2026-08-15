import logfire
from sentence_transformers import SentenceTransformer


# Configuration


BATCH_SIZE = 32

PRIMARY_MODEL = "BAAI/bge-base-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-mpnet-base-v2"

EMBEDDING_DIM = 768


# Active model

_active_model = None
_model_type: str | None = None


# Load primary model

def _load_primary():
    """
    Load BGE embedding model.
    """
    try:
        logfire.info(f"Loading primary embedding model: {PRIMARY_MODEL}")
        model = SentenceTransformer(PRIMARY_MODEL)
        # Test the model
        model.encode(["loading"])
        logfire.info(f"Primary embedding model ready: {PRIMARY_MODEL}")
        return model
    except Exception as e:
        logfire.warning(f"Primary embedding model failed: {e}")
        return None

    
# Load fallback model

def _load_fallback():
    """
    Load Sentence Transformer fallback model.
    """
    logfire.info(f"Loading fallback embedding model: {FALLBACK_MODEL}")
    model = SentenceTransformer(FALLBACK_MODEL)
    logfire.info(f"Fallback embedding model ready: {FALLBACK_MODEL}")
    return model


# Initialize embedding model
def _init():
    global _active_model, _model_type
    # Already initialized
    if _active_model is not None:
        return
    # Try primary model
    primary = _load_primary()
    if primary is not None:
        _active_model = primary
        _model_type = "bge"

    else:
        # Use fallback
        _active_model = _load_fallback()
        _model_type = "mpnet"


# Get embedding dimension

def get_embedding_dim() -> int:
    _init()
    return EMBEDDING_DIM


# Embed one query
def embed_query(query: str) -> list[float]:
    _init()
    with logfire.span("Embed Query",model=_model_type):
        embedding = _active_model.encode(query,normalize_embeddings=True)
        return embedding.tolist()



# Embed multiple texts
def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("Embed Batch",model=_model_type,start=i,size=len(batch)):
            embeddings = _active_model.encode(batch,normalize_embeddings=True,show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
    return all_embeddings