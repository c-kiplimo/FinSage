#!/usr/bin/env python
"""
FinSage — Agentic RAG Financial Advisor
Entry point: launches the Gradio chat interface.
"""

from typing import List, Tuple
import gradio as gr

from agentic_rag_financial_advisor.retriever import Retriever
from agentic_rag_financial_advisor.planner import Planner
from agentic_rag_financial_advisor.rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Knowledge base
# Replace or extend this list with your own documents, PDFs, or DB records.
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    "Diversification spreads risk across asset classes, lowering portfolio volatility.",
    "Index funds historically outperform most actively managed funds over a 10-year horizon.",
    "The expense ratio of an ETF directly reduces investor returns and should be kept low.",
    "Time in the market beats timing the market — staying invested is key to compounding.",
    "A common rule of thumb allocates 60% to equities and 40% to bonds for balanced growth.",
    "Dollar-cost averaging reduces the impact of volatility by investing fixed amounts regularly.",
    "A Roth IRA grows tax-free; contributions are made with after-tax dollars.",
    "High-yield savings accounts and money market funds are safer alternatives to stocks for short-term goals.",
    "Rebalancing your portfolio once or twice a year keeps your target allocation on track.",
    "An emergency fund of 3–6 months of expenses should be in place before investing.",
]

retriever = Retriever(SAMPLE_DOCS)
planner = Planner()
rag = RAGPipeline(retriever)


def chat_fn(user_input: str, history: List[Tuple[str, str]]):
    """
    Called on every user message by Gradio.

    history is Gradio's built-in list of (user, bot) tuples — we use it
    directly instead of global variables, making the app stateless and
    safe for concurrent users.
    """
    # Convert Gradio history to flat chat_history strings for the pipeline
    chat_history: List[str] = []
    for user_msg, bot_msg in history:
        chat_history.append(f"User: {user_msg}")
        if bot_msg:
            chat_history.append(f"Bot: {bot_msg}")

    mode = planner.decide(user_input, chat_history)
    use_rag = (mode == "rag")
    response = rag.answer(user_input, chat_history, use_rag=use_rag)

    history.append((user_input, response))
    return history


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat_fn,
        title="FinSage — Agentic RAG Financial Advisor",
        description=(
            "Ask any finance question. FinSage decides whether to retrieve "
            "knowledge from its document base or answer from conversation context."
        ),
    ).launch()
