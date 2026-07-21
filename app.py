"""
app.py
------
Gradio front end for the HR support assistant.

Run locally:
    python app.py
Then open the local URL it prints (default http://127.0.0.1:7860).

On Hugging Face Spaces (SDK: Gradio), this file is the entry point. Add your
GOOGLE_API_KEY under Settings > Secrets - do NOT commit the .env file.
"""

import gradio as gr

from services.graph import answer_query

EXAMPLES = [
    "What is the policy on international travel?",
    "What are the standard working hours?",
    "How many days of medical leave do I get?",
    "What is the policy on business leave for renewing a driving permit?",
    "How much does the group insurance cover for dental?",
    "When is salary paid each month?",
    "How do I apply for ordination leave (ลาบวช)?",
]


def respond(message, history):
    """Gradio ChatInterface callback -> run the LangGraph workflow."""
    return answer_query(message)


demo = gr.ChatInterface(
    fn=respond,
    title="🏢 HR Support Assistant (RAG + LangGraph)",
    description=(
        "Ask about HR policies: working hours, leave, overtime, insurance, "
        "payroll, and more. Answers are grounded in the company knowledge base."
    ),
    examples=EXAMPLES,
)


if __name__ == "__main__":
    # theme belongs on launch() (ChatInterface.__init__ doesn't accept it).
    # ssr_mode=False: the SSR Node proxy is flaky on Hugging Face Spaces and can
    # tear the whole app down right after startup ("Stopping Node.js server...").
    demo.launch(theme="soft", ssr_mode=False)
