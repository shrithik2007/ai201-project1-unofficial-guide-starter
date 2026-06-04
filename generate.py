"""
generate.py — Grounded response generation via Groq

Retrieves relevant chunks for a query, then calls Groq's
llama-3.3-70b-versatile to generate an answer grounded ONLY
in the retrieved documents.

Every response includes source attribution.

Usage:
    from generate import answer
    result = answer("Is the pizza at Roadrunner Café good?")
    print(result["answer"])
    print(result["sources"])
"""

import os
from groq import Groq
from dotenv import load_dotenv
from retrieve import retrieve

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Unofficial Guide assistant for UTSA campus dining.

Your job is to answer student questions about campus dining using ONLY the document excerpts provided below. Do not use any outside knowledge or make up information. If the provided excerpts do not contain enough information to answer the question, say exactly: "I don't have enough information about that in my documents."

Rules:
- Answer only from the provided context.
- Be specific and practical — students want real advice, not vague generalities.
- If multiple excerpts give conflicting information, acknowledge the disagreement.
- Keep your answer concise (2–5 sentences) unless a longer answer is clearly needed.
- Do NOT add disclaimers like "as an AI" or "I recommend verifying this." Just answer.
"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Excerpt {i} — from {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def answer(query: str, top_k: int = 5) -> dict:
    """
    End-to-end RAG: retrieve relevant chunks → generate grounded answer.

    Returns:
        {
            "answer": str,         # LLM response grounded in retrieved context
            "sources": list[str],  # deduplicated list of source filenames
            "chunks": list[dict],  # raw retrieved chunks for inspection
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Build context
    context = format_context(chunks)

    user_message = f"""Here are excerpts from student-generated documents about UTSA campus dining:

{context}

---

Student question: {query}

Answer using only the excerpts above."""

    # Step 3: Generate
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,   # low temperature = more factual, less creative
        max_tokens=512,
    )

    response_text = completion.choices[0].message.content.strip()

    # Deduplicated source list (preserves order of first appearance)
    seen = set()
    sources = []
    for chunk in chunks:
        s = chunk["source"]
        if s not in seen:
            seen.add(s)
            sources.append(s)

    return {
        "answer": response_text,
        "sources": sources,
        "chunks": chunks,
    }


if __name__ == "__main__":
    # Quick end-to-end test
    test_query = "What's the best time to eat lunch at Roadrunner Café to avoid the crowds?"
    print(f"QUERY: {test_query}\n")

    result = answer(test_query)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  • {s}")
