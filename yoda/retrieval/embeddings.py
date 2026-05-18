"""Converts text into embedding vectors using OpenAI text-embedding-3-small.

embed_texts(texts) returns one 1536-dim vector per input string via the
OpenAI API. Cost is ~$0.004 per 10-Q filing.
"""

from openai import OpenAI

from yoda import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenAI embedding model and output dimension.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Maximum number of texts to send in a single OpenAI API call.
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused for every call)
# ---------------------------------------------------------------------------

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a 1536-dim embedding vector for each string in *texts*.

    The output list has the same length and order as the input.
    Empty list returns an empty list.
    """
    if not texts:
        return []
    # Batched calls keep individual responses small and progress observable.
    results: list[list[float]] = []
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            results.append(item.embedding)
    return results
