"""
embed.py — Embedding and ChromaDB vector store setup

Embeds all document chunks using all-MiniLM-L6-v2 (sentence-transformers)
and stores them in a local ChromaDB collection with source metadata.

Run this once to build the vector store before querying:
    python embed.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
from ingest import build_chunks

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "utsa_dining"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_or_create_collection(chroma_path: str = CHROMA_PATH) -> chromadb.Collection:
    """Initialize ChromaDB client and return the dining collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for sentence embeddings
    )
    return collection


def build_vector_store(chroma_path: str = CHROMA_PATH) -> chromadb.Collection:
    """
    Full embedding pipeline:
    1. Load and chunk all documents via ingest.py
    2. Embed each chunk with all-MiniLM-L6-v2
    3. Store in ChromaDB with source metadata

    Idempotent: if the collection already has documents, this function
    deletes and rebuilds it so re-running always produces a clean store.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building chunks from documents...")
    chunks = build_chunks()

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Set up ChromaDB — delete existing collection to ensure clean rebuild
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection if it exists (clean rebuild)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' for clean rebuild.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB expects lists
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )

    print(f"\nVector store built: {collection.count()} chunks in ChromaDB at '{chroma_path}'")
    return collection


if __name__ == "__main__":
    collection = build_vector_store()
    print("\nDone. Run app.py or retrieve.py to query the system.")
