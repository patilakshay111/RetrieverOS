# service.py: wires embeddings, persistent storage, retrieval, generation, and LangGraph

import hashlib
import threading
from collections import OrderedDict
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.engine import make_url

from rag_pipeline.config import Settings
from rag_pipeline.conversations import conversation_thread_id
from rag_pipeline.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from rag_pipeline.generation import AnswerGenerator
from rag_pipeline.graph import build_graph
from rag_pipeline.lexical_store import LexicalStore
from rag_pipeline.models import AccessContext, AnswerResponse, AnswerState
from rag_pipeline.retrieval import CrossEncoderReranker, HybridRetriever
from rag_pipeline.storage import KnowledgeBase
from rag_pipeline.vector_store import VectorStore


# Assemble the reusable components shared by API and evaluation requests.
class RAGService:
    # Initialize the persistent corpus and the grounded answer graph.
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider | None = None,
        generator: AnswerGenerator | None = None,
        use_memory_saver: bool | None = None,
        vector_store: VectorStore | None = None,
        lexical_store: LexicalStore | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.settings = settings
        self.embeddings = embeddings or (
            OpenAIEmbeddingProvider(settings) if settings.openai_api_key else None
        )
        shared_retrieval_configured = bool(settings.qdrant_url and settings.database_url)
        if lexical_store is not None and not shared_retrieval_configured:
            raise ValueError(
                "Qdrant and PostgreSQL must both be configured for shared retrieval."
            )
        self._owns_vector_store = shared_retrieval_configured and vector_store is None
        self._owns_lexical_store = shared_retrieval_configured and lexical_store is None
        self.vector_store = (
            vector_store
            if vector_store is not None
            else (
                VectorStore(
                    settings.qdrant_url,
                    vector_size=settings.qdrant_vector_size,
                )
                if shared_retrieval_configured
                else None
            )
        )
        self.lexical_store = (
            lexical_store
            if lexical_store is not None
            else (
                LexicalStore(settings.database_url)
                if shared_retrieval_configured
                else None
            )
        )
        durable_vector_store = (
            self.vector_store if shared_retrieval_configured else None
        )
        durable_lexical_store = (
            self.lexical_store if shared_retrieval_configured else None
        )
        self.knowledge_base = KnowledgeBase(
            settings,
            self.embeddings,
            vector_store=durable_vector_store,
            lexical_store=durable_lexical_store,
        )
        self.generator = generator or (
            AnswerGenerator(settings) if settings.openai_api_key else None
        )
        reranker = (
            CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
        )
        self._reranker = reranker
        if checkpointer is not None:
            self._checkpointer = checkpointer
        elif use_memory_saver is not None:
            self._checkpointer = MemorySaver() if use_memory_saver else None
        else:
            postgres_configured = bool(
                settings.database_url
                and make_url(settings.database_url).drivername.startswith(
                    "postgresql"
                )
            )
            self._checkpointer = None if postgres_configured else MemorySaver()
        self._graph_cache: OrderedDict[str, object] = OrderedDict()
        self._graph_lock = threading.Lock()
        self.retriever = (
            HybridRetriever(
                self.knowledge_base,
                self.embeddings,
                settings,
                reranker,
                vector_store=durable_vector_store,
                lexical_store=durable_lexical_store,
            )
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
                self.knowledge_base,
                embeddings,
                self.settings,
                self._reranker,
                vector_store=(
                    self.vector_store
                    if self.settings.qdrant_url and self.settings.database_url
                    else None
                ),
                lexical_store=(
                    self.lexical_store
                    if self.settings.qdrant_url and self.settings.database_url
                    else None
                ),
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

    def close(self) -> None:
        if self._owns_lexical_store and self.lexical_store is not None:
            self.lexical_store.close()
        if self._owns_vector_store and self.vector_store is not None:
            self.vector_store.close()

    @property
    def answer_cache_enabled(self) -> bool:
        return self._checkpointer is None

    @property
    def answer_cache_index_version(self) -> str:
        return self.knowledge_base.snapshot().generation

    def delete_conversation_checkpoint(
        self, access_context: AccessContext, conversation_id: UUID
    ) -> None:
        if self._checkpointer is not None:
            for variant in (None, "hybrid-compare", "dense-compare"):
                self._checkpointer.delete_thread(
                    conversation_thread_id(
                        access_context, conversation_id, variant
                    )
                )

    # Run the graph with a stable checkpoint thread and return a typed response.
    def ask(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        thread_id: str = "chat_session_1",
        access_context: AccessContext | None = None,
        api_key: str | None = None,
        conversation_id: UUID | None = None,
        checkpoint_variant: str | None = None,
    ) -> AnswerResponse:
        if conversation_id is not None and access_context is not None:
            checkpoint_id = conversation_thread_id(
                access_context, conversation_id, checkpoint_variant
            )
            thread_id = checkpoint_id
        else:
            checkpoint_id = thread_id
        if access_context is not None and conversation_id is None:
            checkpoint_id = hashlib.sha256(
                f"{access_context.workspace_id}:{access_context.user_id}:{thread_id}".encode()
            ).hexdigest()
        initial = AnswerState(
            question=question,
            thread_id=thread_id,
            retrieval_mode=retrieval_mode,
        )
        graph = self._graph_for_key(api_key)
        configurable: dict[str, object] = {"thread_id": checkpoint_id}
        if access_context is not None:
            configurable["access_context"] = access_context.model_dump(mode="json")
        result = graph.invoke(
            initial,
            config={"configurable": configurable},
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
            conversation_id=conversation_id,
        )
