"""
app.py — Gradio web interface for the UTSA Unofficial Dining Guide

Run with:
    python app.py

Then open http://localhost:7860 in your browser.

Prerequisites: embed.py must have been run first to build the vector store.
"""

import gradio as gr
from generate import answer


def handle_query(question: str):
    """
    Called when the user submits a question.
    Returns (answer_text, sources_text, debug_text).
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    result = answer(question.strip())

    answer_text = result["answer"]

    sources_text = "\n".join(f"• {s}" for s in result["sources"])

    # Build debug view of retrieved chunks for transparency
    debug_parts = []
    for i, chunk in enumerate(result["chunks"], 1):
        debug_parts.append(
            f"[Chunk {i}] Source: {chunk['source']} | Distance: {chunk['distance']}\n"
            f"{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}"
        )
    debug_text = "\n\n---\n\n".join(debug_parts)

    return answer_text, sources_text, debug_text


# Example questions shown in the UI
examples = [
    ["What's the best time to eat lunch at Roadrunner Café to avoid long lines?"],
    ["Do meal swipes work at Chick-fil-A in the Sombrilla?"],
    ["What are the best vegan options on campus?"],
    ["Where is the Rowdy Cart and how do I find it?"],
    ["What can I eat on campus after 9pm?"],
    ["Is the pizza at Roadrunner Café good?"],
    ["Which meal plan should I get as a freshman living on campus?"],
    ["Where's the best coffee on campus?"],
]

with gr.Blocks(title="UTSA Unofficial Dining Guide", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🍽️ UTSA Unofficial Dining Guide
        ### The real student knowledge — searchable

        Ask any question about UTSA campus dining: which food is good, when to go,
        how meal plans work, where to find the best deals, dietary options, and more.
        Answers are drawn from actual student reviews and guides — not the official dining website.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your question",
                placeholder='e.g. "What time should I eat lunch to avoid the rush?"',
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary", size="lg")

        with gr.Column(scale=3):
            answer_output = gr.Textbox(
                label="Answer",
                lines=6,
                interactive=False,
            )
            sources_output = gr.Textbox(
                label="Retrieved from",
                lines=3,
                interactive=False,
            )

    with gr.Accordion("🔍 See retrieved chunks (for transparency)", open=False):
        debug_output = gr.Textbox(
            label="Raw retrieved context",
            lines=15,
            interactive=False,
        )

    gr.Examples(
        examples=examples,
        inputs=question_input,
        label="Example questions",
    )

    # Wire up events
    submit_btn.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, debug_output],
    )
    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, debug_output],
    )

    gr.Markdown(
        """
        ---
        *Answers are generated from student-written documents and may reflect opinions,
        not official UTSA policy. Always verify hours and policies directly with UTSA Dining.*
        """
    )

if __name__ == "__main__":
    print("Starting UTSA Unofficial Dining Guide...")
    print("Make sure you have run embed.py first to build the vector store.")
    demo.launch(server_name="0.0.0.0", server_port=7860)
