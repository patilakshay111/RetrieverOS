# service.py: wires embeddings, persistent storage, retrieval, generation, and LangGraph

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from rag_pipeline.generation import AnswerGenerator
from rag_pipeline.graph import build_graph
from rag_pipeline.models import AnswerResponse, AnswerState
from rag_pipeline.retrieval import CrossEncoderReranker, HybridRetriever
from rag_pipeline.storage import KnowledgeBase


# Assemble the reusable components shared by API and evaluation requests.
class RAGService:
    # Initialize the persistent corpus and the grounded answer graph.
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider | None = None,
        generator: AnswerGenerator | None = None,
        use_memory_saver: bool = True,
    ):
        self.settings = settings
        self.embeddings = embeddings or OpenAIEmbeddingProvider(settings)
        self.knowledge_base = KnowledgeBase(settings, self.embeddings)
        self.generator = generator or AnswerGenerator(settings)
        reranker = (
            CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
        )
        self.retriever = HybridRetriever(
            self.knowledge_base, self.embeddings, settings, reranker
        )
        self.graph = build_graph(
            self.retriever,
            self.generator,
            settings,
            use_memory_saver=use_memory_saver,
        )

    # Run the graph with a stable checkpoint thread and return a typed response.
    def ask(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        thread_id: str = "chat_session_1",
    ) -> AnswerResponse:
        initial = AnswerState(
            question=question,
            thread_id=thread_id,
            retrieval_mode=retrieval_mode,
        )
        result = self.graph.invoke(
            initial,
            config={"configurable": {"thread_id": thread_id}},
        )
        state = AnswerState.model_validate(result)
        return AnswerResponse(
            question=state.question,
            answer=state.answer,
            retrieval_mode=state.retrieval_mode,
            citations=state.citation_checks,
            retrieved_chunks=state.retrieved_chunks,
            confidence=state.confidence,
            confidence_breakdown={
                "retrieval_confidence": state.retrieval_confidence,
                "citation_coverage": state.citation_coverage,
                "answer_completeness": state.answer_completeness,
            },
            insufficient_context=state.insufficient_context,
            documents_to_check=state.documents_to_check,
            thread_id=state.thread_id,
        )
