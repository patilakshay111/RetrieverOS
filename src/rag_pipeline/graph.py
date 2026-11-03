# graph.py: defines the simple Pydantic-state LangGraph retrieval and citation workflow

from typing import Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from rag_pipeline.config import Settings
from rag_pipeline.generation import insufficient_context_answer
from rag_pipeline.models import (
    AccessContext,
    AnswerState,
    CitationCheck,
    RetrievedChunk,
)


class Retriever(Protocol):
    def search(
        self, query: str, mode: str, access_context: AccessContext | None = None
    ) -> tuple[list[RetrievedChunk], float]: ...


class Generator(Protocol):
    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[str] | None = None,
    ) -> str: ...

    def verify_citations(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> list[CitationCheck]: ...

    def citation_coverage(self, checks: list[CitationCheck]) -> float: ...

    def assess_completeness(self, question: str, answer: str) -> tuple[float, str]: ...


# Compile the workflow with an injected saver or the local in-memory default.
def build_graph(
    retriever: Retriever,
    generator: Generator,
    settings: Settings,
    use_memory_saver: bool = True,
    checkpointer=None,
):
    # Retrieve only the chunks relevant to the current question.
    def retrieve(state: AnswerState, config: RunnableConfig) -> dict:
        retrieval_query = "\n".join([*state.history[-6:], f"Current question: {state.question}"])
        configurable = config.get("configurable") or {}
        configured_scope = configurable.get("access_context")
        access_context = (
            state.access_context
            if configured_scope is None
            else AccessContext.model_validate(configured_scope)
        )
        chunks, confidence = retriever.search(
            retrieval_query, state.retrieval_mode, access_context
        )
        return {
            "retrieved_chunks": chunks,
            "retrieval_confidence": confidence,
            "insufficient_context": confidence < settings.min_retrieval_confidence,
        }

    # Generate a grounded response or an explicit no-answer message.
    def generate(state: AnswerState) -> dict:
        if state.insufficient_context:
            answer, documents = insufficient_context_answer(
                state.question, state.retrieved_chunks
            )
            return {"answer": answer, "documents_to_check": documents}
        return {
            "answer": generator.generate(
                state.question, state.retrieved_chunks, state.history[-6:]
            )
        }

    # Verify every cited claim against the numbered evidence returned by retrieval.
    def verify(state: AnswerState) -> dict:
        if state.insufficient_context:
            return {"citation_checks": [], "citation_coverage": 0.0}
        checks = generator.verify_citations(state.answer, state.retrieved_chunks)
        return {
            "citation_checks": checks,
            "citation_coverage": generator.citation_coverage(checks),
        }

    # Combine retrieval, citation, and completeness signals into one transparent score.
    def score(state: AnswerState) -> dict:
        if state.insufficient_context:
            return {
                "answer_completeness": 0.0,
                "confidence": round(state.retrieval_confidence * 0.4, 4),
                "history": [f"User: {state.question}", f"Assistant: {state.answer}"],
            }
        completeness, _ = generator.assess_completeness(state.question, state.answer)
        confidence = (
            0.4 * state.retrieval_confidence
            + 0.35 * state.citation_coverage
            + 0.25 * completeness
        )
        return {
            "answer_completeness": completeness,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "history": [f"User: {state.question}", f"Assistant: {state.answer}"],
        }

    graph = StateGraph(AnswerState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("score", score)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "score")
    graph.add_edge("score", END)
    graph.set_entry_point("retrieve")
    saver = checkpointer if checkpointer is not None else (
        MemorySaver() if use_memory_saver else None
    )
    return graph.compile(checkpointer=saver)
