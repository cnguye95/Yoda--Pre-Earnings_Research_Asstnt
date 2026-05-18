"""ChromaDB wrapper for storing and querying SEC filing chunks by embedding.

ChromaStore holds a persistent Chroma collection ("filings_openai") where
every chunk from every processed filing is stored as a 1536-dim OpenAI vector.

Documents are keyed by "{accession_number}_{chunk_index}" so upserting the
same filing twice is idempotent — the second call overwrites the first.

Critical correctness invariant: every query filters by accession_number via
Chroma's `where=` clause. Without this filter, a query for one company's
filing could return chunks from a completely different company.

The data/chroma/ directory is created by Chroma on first use and is already
gitignored (added in Phase 0).
"""

import chromadb

from yoda.ingest.chunker import Chunk
from yoda.retrieval.embeddings import embed_texts


# ---------------------------------------------------------------------------
# ChromaStore class
# ---------------------------------------------------------------------------

class ChromaStore:
    """Persistent Chroma vector store for SEC filing chunks.

    A class is used here because it holds the Chroma client and collection
    objects across multiple upsert/query calls within a single process. Using
    a class avoids reopening the on-disk database on every call.
    """

    def __init__(self) -> None:
        # Open (or create) the persistent Chroma database at data/chroma/.
        # Chroma creates the directory if it doesn't exist.
        self._client = chromadb.PersistentClient(path="data/chroma")

        # Get or create the collection. Cosine distance is the standard metric
        # for unit-vector embeddings from text-embedding-3-small.
        self._collection = self._client.get_or_create_collection(
            name="filings_openai",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        accession_number: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their embeddings in the collection.

        If a chunk with the same ID already exists (same accession + index),
        its document, embedding, and metadata are overwritten. This makes
        repeated calls for the same filing safe and idempotent.

        Parameters
        ----------
        accession_number : str
            The SEC accession number for the filing (e.g. "0000320193-25-000123").
            Used as part of every chunk's ID and stored in metadata for filtering.
        chunks : list[Chunk]
            The Chunk objects produced by chunk_filing().
        embeddings : list[list[float]]
            Parallel list of vectors, one per chunk. Must be the same length as *chunks*.
        """
        if not chunks:
            return

        # Build parallel lists that Chroma's upsert expects.
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Unique ID per chunk within this filing.
            chunk_id = f"{accession_number}_{chunk.chunk_index}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "accession":   accession_number,
                "section":     chunk.section,
                "chunk_index": chunk.chunk_index,
                "char_start":  chunk.char_start,
                "char_end":    chunk.char_end,
            })

        # Upsert all chunks at once. Chroma accepts parallel lists.
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        accession_number: str,
        query_text: str,
        k: int = 5,
    ) -> list[Chunk]:
        """Return the *k* chunks most semantically similar to *query_text*.

        Results are filtered to *accession_number* only — chunks from other
        filings are never returned, even if they score higher. This is the
        critical correctness invariant that prevents cross-filing pollution.

        Parameters
        ----------
        accession_number : str
            Only chunks from this filing are considered.
        query_text : str
            The search query. It is embedded with the same model used at
            upsert time so the vector spaces match.
        k : int
            Number of results to return (default 5).
        """
        # Embed the query text so it lives in the same vector space as the stored chunks.
        query_vector = embed_texts([query_text])[0]

        # Query Chroma with the where= filter locked to this accession.
        # Without the filter, results from other filings could appear.
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where={"accession": accession_number},
        )

        # Reconstruct Chunk objects from the raw Chroma response.
        # results["documents"][0] and results["metadatas"][0] are parallel
        # lists of the k nearest neighbours in order of similarity.
        chunks: list[Chunk] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for doc, meta in zip(documents, metadatas):
            chunks.append(
                Chunk(
                    text=doc,
                    section=meta["section"],
                    chunk_index=meta["chunk_index"],
                    char_start=meta["char_start"],
                    char_end=meta["char_end"],
                )
            )

        return chunks
