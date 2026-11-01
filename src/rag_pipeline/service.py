# service.py: wires embeddings, persistent storage, retrieval, generation, and LangGraph

import hashlib
import threading
from collections import OrderedDict

from langgraph.checkpoint.memory import MemorySaver

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from rag_pipeline.generation import AnswerGenerator
from rag_pipeline.graph import build_graph
from rag_pipeline.models import AccessContext, AnswerResponse, AnswerState
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
        self.embeddings = embeddings or (
            OpenAIEmbeddingProvider(settings) if settings.openai_api_key else None
        )
        self.knowledge_base = KnowledgeBase(settings, self.embeddings)
        self.generator = generator or (
            AnswerGenerator(settings) if settings.openai_api_key else None
        )
        reranker = (
            CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
        )
        self._reranker = reranker
        self._checkpointer = MemorySaver() if use_memory_saver else None
        self._graph_cache: OrderedDict[str, object] = OrderedDict()
        self._graph_lock = threading.Lock()
        self.retriever = (
            HybridRetriever(self.knowledge_base, self.embeddings, settings, reranker)
            if self.embeddings is not None
            else None
        )
        self.graph = (
            build_graph(
                self.retriever,
                self.generator,
                settings,
                use_memory_saver=False,
                checkpointer=self._checkpointer,
            )
            if self.retriever is not None and self.generator is not None
            else None
        )

    # Reuse model clients and a graph per credential without retaining plaintext keys.
    def _graph_for_key(self, api_key: str | None):
        if api_key is None:
            if self.graph is None:
                raise RuntimeError("An OpenAI API key must be configured for this account.")
            return self.graph
        if api_key == self.settings.openai_api_key and self.graph is not None:
            return self.graph
        cache_key = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        with self._graph_lock:
            graph = self._graph_cache.get(cache_key)
            if graph is not None:
                self._graph_cache.move_to_end(cache_key)
                return graph
            embeddings = OpenAIEmbeddingProvider(self.settings, api_key)
            generator = AnswerGenerator(self.settings, api_key)
            retriever = HybridRetriever(
                self.knowledge_base, embeddings, self.settings, self._reranker
            )
            graph = build_graph(
                retriever,
                generator,
                self.settings,
                use_memory_saver=False,
                checkpointer=self._checkpointer,
            )
            self._graph_cache[cache_key] = graph
            if len(self._graph_cache) > 64:
                self._graph_cache.popitem(last=False)
            return graph

    # Run the graph with a stable checkpoint thread and return a typed response.
    def ask(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        thread_id: str = "chat_session_1",
        access_context: AccessContext | None = None,
        api_key: str | None = None,
    ) -> AnswerResponse:
        checkpoint_id = thread_id
        if access_context is not None:
            checkpoint_id = hashlib.sha256(
                f"{access_context.workspace_id}:{access_context.user_id}:{thread_id}".encode()
            ).hexdigest()
        initial = AnswerState(
            question=question,
            thread_id=thread_id,
            retrieval_mode=retrieval_mode,
            access_context=access_context,
        )
        graph = self._graph_for_key(api_key)
        result = graph.invoke(
            initial,
            config={"configurable": {"thread_id": checkpoint_id}},
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
