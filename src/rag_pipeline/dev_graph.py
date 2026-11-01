"""LangGraph CLI entrypoint for local development and Studio testing."""

from rag_pipeline.config import Settings
from rag_pipeline.generation import AnswerGenerator
from rag_pipeline.graph import build_graph
from rag_pipeline.models import AccessContext, CitationCheck, RetrievedChunk
from rag_pipeline.service import RAGService


class _LazyRuntime:
    """Create the credential-dependent RAG service only when the graph is invoked."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.service: RAGService | None = None

    def get_service(self) -> RAGService:
        if self.service is None:
            self.service = RAGService(self.settings, use_memory_saver=False)
        return self.service


class _LazyRetriever:
    """Defer embeddings and local-index initialization until retrieval runs."""

    def __init__(self, runtime: _LazyRuntime):
        self.runtime = runtime

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        access_context: AccessContext | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        return self.runtime.get_service().retriever.search(query, mode, access_context)


class _LazyGenerator:
    """Defer model client construction until an answer needs model calls."""

    def __init__(self, runtime: _LazyRuntime):
        self.runtime = runtime

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[str] | None = None,
    ) -> str:
        return self.runtime.get_service().generator.generate(question, chunks, history)

    def verify_citations(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> list[CitationCheck]:
        return self.runtime.get_service().generator.verify_citations(answer, chunks)

    @staticmethod
    def citation_coverage(checks: list[CitationCheck]) -> float:
        return AnswerGenerator.citation_coverage(checks)

    def assess_completeness(self, question: str, answer: str) -> tuple[float, str]:
        return self.runtime.get_service().generator.assess_completeness(question, answer)


settings = Settings.from_env()
runtime = _LazyRuntime(settings)
graph = build_graph(
    _LazyRetriever(runtime),
    _LazyGenerator(runtime),
    settings,
    use_memory_saver=False,
)
