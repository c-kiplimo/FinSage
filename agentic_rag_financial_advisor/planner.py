
"""
Planner — decides whether to use RAG (knowledge retrieval) or rely on
chat history alone. Uses an expanded finance keyword list and detects
follow-up questions by checking recent conversation context.
"""
from typing import List

FINANCE_TERMS = {
    "stock", "invest", "etf", "portfolio", "bond", "market", "equity",
    "fund", "dividend", "return", "risk", "asset", "wealth", "saving",
    "retirement", "401k", "ira", "crypto", "inflation", "interest rate",
    "hedge", "mutual fund", "index fund", "expense ratio", "rebalance",
    "allocation", "diversif", "compound", "bull", "bear", "trade",
    "financial", "finance", "money", "income", "budget", "debt", "loan",
    "mortgage", "roth", "brokerage", "capital gain", "tax", "yield",
    "commodity", "derivative", "option", "future", "liquidity", "p/e",
    "earnings", "revenue", "profit", "loss", "balance sheet", "cash flow",
}


class Planner:
    """Routes each query to the appropriate pipeline strategy."""

    def decide(self, query: str, chat_history: List[str]) -> str:
        """
        Returns 'rag' if the query is finance-related or follows a finance
        conversation, otherwise returns 'chat_history' for a lightweight
        conversational response.
        """
        lowered = query.lower()

        # Direct match on the current query
        if any(term in lowered for term in FINANCE_TERMS):
            return "rag"

        # Follow-up detection: if recent context was financial, treat this
        # query as part of the same financial conversation
        recent = chat_history[-4:] if len(chat_history) >= 4 else chat_history
        if recent and any(
            any(term in msg.lower() for term in FINANCE_TERMS)
            for msg in recent
        ):
            return "rag"

        return "chat_history"
