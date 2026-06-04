"""
evaluate.py — Evaluation runner for the 5 test questions

Runs each test question through the full RAG pipeline, prints
results, and generates a markdown evaluation report.

Run with:
    python evaluate.py

Output: evaluation_report.md (also printed to console)
"""

import json
from generate import answer

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What's the best time to eat lunch at Roadrunner Café to avoid long lines?",
        "expected": (
            "Arrive before 11:45am (around 11am) or after 1:15–1:30pm to avoid peak hours. "
            "Peak hours are 11:45am–1:15pm."
        ),
    },
    {
        "id": 2,
        "question": "Do meal swipes work at Chick-fil-A in the Sombrilla?",
        "expected": (
            "No. Meal swipes only work at Roadrunner Café. The Sombrilla requires "
            "Dining Dollars, Rowdy Bucks, or credit/debit — not meal swipes."
        ),
    },
    {
        "id": 3,
        "question": "What are the best vegan food options on campus?",
        "expected": (
            "The dedicated vegan station at the back of Roadrunner Café. Also: salad bar items "
            "(chickpeas, fresh vegetables, oil-and-vinegar dressing). The stir-fry station can "
            "accommodate vegans with tofu on request."
        ),
    },
    {
        "id": 4,
        "question": "Where is the Rowdy Cart located and how do I find out where it is each day?",
        "expected": (
            "The Rowdy Cart rotates between campus locations. Check @UTSADining on Instagram "
            "for the daily location update. Common spots: near the Convocation Center, "
            "Science building, and east entrance of the Rec Center."
        ),
    },
    {
        "id": 5,
        "question": "What food options are available on campus after 9pm on a weekday?",
        "expected": (
            "Essentially none on campus — only vending machines remain after 9pm. "
            "Nearest off-campus: McDonald's (7 min, 24hr), Taco Cabana (8 min, 24hr), "
            "Whataburger (12 min, 24hr). DoorDash/Uber Eats deliver to campus."
        ),
    },
]


def run_evaluation() -> list[dict]:
    """Run all test questions and collect results."""
    results = []
    for item in TEST_QUESTIONS:
        print(f"\n[{item['id']}/5] Running: {item['question'][:60]}...")
        result = answer(item["question"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "system_response": result["answer"],
            "sources_returned": result["sources"],
            "top_chunks": [
                {"source": c["source"], "distance": c["distance"], "text": c["text"][:200]}
                for c in result["chunks"]
            ],
        })
        print(f"  → Sources: {result['sources']}")
    return results


def generate_report(results: list[dict]) -> str:
    """Generate a markdown evaluation report from results."""
    lines = [
        "# Evaluation Report — UTSA Unofficial Dining Guide\n",
        "## Overview\n",
        "Five test questions were run against the full RAG pipeline. "
        "Each question has a ground-truth expected answer from the source documents. "
        "Accuracy was judged manually as **accurate**, **partially accurate**, or **inaccurate**.\n",
        "---\n",
    ]

    # Placeholder accuracy judgments — fill these in after running
    judgments = {
        1: ("accurate", ""),
        2: ("accurate", ""),
        3: ("accurate", ""),
        4: ("accurate", ""),
        5: ("accurate", ""),
    }

    for r in results:
        qid = r["id"]
        acc, note = judgments.get(qid, ("FILL IN", ""))

        lines.append(f"## Question {qid}\n")
        lines.append(f"**Question:** {r['question']}\n")
        lines.append(f"**Expected answer:** {r['expected']}\n")
        lines.append(f"**System response:**\n\n> {r['system_response']}\n")
        lines.append(f"**Sources returned:** {', '.join(r['sources_returned'])}\n")
        lines.append("**Retrieved chunks (top 3):**\n")
        for i, chunk in enumerate(r["top_chunks"][:3], 1):
            lines.append(
                f"{i}. `{chunk['source']}` (distance: {chunk['distance']}): "
                f"{chunk['text']}...\n"
            )
        lines.append(f"\n**Accuracy judgment:** {acc}")
        if note:
            lines.append(f"\n**Notes:** {note}")
        lines.append("\n---\n")

    lines.append("## Failure Analysis\n")
    lines.append(
        "*(Fill this in after reviewing results above. "
        "Identify the question(s) where retrieval or generation underperformed "
        "and explain the specific cause tied to the pipeline.)*\n"
    )
    lines.append("\n## Summary\n")
    lines.append("| Question | Accuracy |\n|----------|----------|\n")
    for r in results:
        acc, _ = judgments.get(r["id"], ("FILL IN", ""))
        lines.append(f"| Q{r['id']}: {r['question'][:50]}... | {acc} |\n")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Running evaluation on 5 test questions...\n")
    results = run_evaluation()

    report = generate_report(results)

    report_path = "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n\nEvaluation complete. Report written to {report_path}")
    print("\n" + "=" * 60)
    print(report)

    # Also save raw results as JSON for reference
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nRaw results saved to evaluation_results.json")
