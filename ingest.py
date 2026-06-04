"""
ingest.py — Document loading and chunking pipeline

Loads all .txt files from the documents/ directory, cleans them,
and splits them into overlapping chunks ready for embedding.

Chunk size: 600 characters
Overlap: 100 characters
"""

import os
import re
from pathlib import Path


DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_raw_documents(docs_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """Load all .txt files from the documents directory."""
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "raw_text": text})
    print(f"Loaded {len(docs)} documents from {docs_dir}")
    return docs


def clean_text(text: str) -> str:
    """
    Remove document headers (SOURCE/DATE lines) and other boilerplate.
    Keeps review content, opinions, facts.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Skip header lines like "SOURCE: ..." and "DATE: ..."
        if re.match(r"^(SOURCE|DATE):", line.strip()):
            continue
        # Skip separator lines (only dashes)
        if re.match(r"^-{3,}$", line.strip()):
            continue
        cleaned_lines.append(line)

    # Collapse multiple blank lines into one
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-level chunks.

    Strategy: slide a window of `chunk_size` chars over the text,
    stepping by (chunk_size - overlap) each time. This ensures
    no content is stranded at a boundary without context.

    Paragraph boundaries are preferred over mid-word splits —
    the splitter walks back from the boundary to the nearest
    newline when one exists within the last 20% of the chunk.
    """
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size

        # Prefer to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for a newline within the last 20% of the chunk
            search_start = start + int(chunk_size * 0.8)
            newline_pos = text.rfind("\n", search_start, end)
            if newline_pos != -1:
                end = newline_pos

        chunk = text[start:end].strip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)

        start += step

    return chunks


def build_chunks(docs_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """
    Full pipeline: load → clean → chunk → return list of chunk dicts.

    Each chunk dict contains:
      - text: the chunk content
      - source: filename of the source document
      - chunk_id: unique string ID (source + index)
    """
    raw_docs = load_raw_documents(docs_dir)
    all_chunks = []

    for doc in raw_docs:
        cleaned = clean_text(doc["raw_text"])
        chunks = chunk_text(cleaned)

        for i, chunk_text_content in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text_content,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}__chunk{i:03d}",
            })

    print(f"Total chunks produced: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print("\n--- 5 sample chunks ---\n")
    import random
    samples = random.sample(chunks, min(5, len(chunks)))
    for c in samples:
        print(f"SOURCE: {c['source']} | ID: {c['chunk_id']}")
        print(c["text"])
        print("-" * 60)
