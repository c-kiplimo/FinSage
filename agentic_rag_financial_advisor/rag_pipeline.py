
from typing import List
from agentic_rag_financial_advisor.retriever import Retriever
from agentic_rag_financial_advisor.llm import call_llm

SYS_PROMPT = (
    "You are FinSage, a knowledgeable and concise financial advisor. "
    "When CONTEXT is provided, use it to ground your answer in facts. "
    "If the context does not cover the question, say so and offer general guidance. "
    "Never fabricate specific figures, rates, or regulations."
)


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def answer(self, query: str, chat_history: List[str], use_rag: bool = True) -> str:
        """
        Build a prompt and call the LLM.

        Parameters
        ----------
        use_rag : bool
            True  → retrieve relevant documents and include them as CONTEXT.
            False → skip retrieval; answer purely from conversation history.
        """
        history_text = "\n".join(chat_history) if chat_history else "None"

        if use_rag:
            docs = self.retriever.retrieve(query, k=5)
            context = "\n".join(docs)
            prompt = (
                f"{SYS_PROMPT}\n\n"
                f"### CONTEXT\n{context}\n\n"
                f"### CHAT HISTORY\n{history_text}\n\n"
                f"### USER QUESTION\n{query}\n\n"
                f"### RESPONSE"
            )
        else:
            prompt = (
                f"{SYS_PROMPT}\n\n"
                f"### CHAT HISTORY\n{history_text}\n\n"
                f"### USER QUESTION\n{query}\n\n"
                f"### RESPONSE"
            )

        return call_llm([{"role": "user", "content": prompt}])
