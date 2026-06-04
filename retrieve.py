"""
retrieve.py — Semantic retrieval from ChromaDB

Given a user query string, embeds it and returns the top-k most
relevant chunks from the vector store, along with their source files.

Usage:
    from retrieve import retrieve
    results = retrieve("Is the pizza at Roadrunner good?")
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "utsa_dining"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# Module-level singletons so model/client load once per process
_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed the query and return the top_k most similar chunks.

    Returns a list of dicts, each with:
      - text: chunk content
      - source: source document filename
      - distance: cosine distance (lower = more similar; 0 = identical)
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "distance": round(dist, 4),
        })

    return chunks


if __name__ == "__main__":
    # Quick retrieval test — run with: python retrieve.py
    test_queries = [
        "What's the best time to eat lunch to avoid crowds?",
        "Do meal swipes work at Chick-fil-A?",
        "What food is available after 9pm on campus?",
    ]

    for query in test_queries:
        print(f"\nQUERY: {query}")
        print("=" * 60)
        results = retrieve(query)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] SOURCE: {r['source']} | distance: {r['distance']}")
            print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
        print()
