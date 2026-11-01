# embeddings.py: provides OpenAI embeddings and deterministic normalization helpers

from collections.abc import Sequence
import numpy as np
from langchain_openai import ChatOpenAI

from rag_pipeline.config import Settings


# Define the small interface used by semantic chunking, indexing, and query retrieval.
class EmbeddingProvider:
    # Embed a batch of text strings as a two-dimensional float32 matrix.
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    # Embed one query as a one-dimensional float32 vector.
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


# Create the interview-rule one-line OpenAI chat helper for shared application use.
def llm() -> ChatOpenAI: return ChatOpenAI(model=Settings.from_env().chat_model, temperature=0)


# Wrap OpenAI's text-embedding-3-small-compatible endpoint behind the local interface.
class OpenAIEmbeddingProvider(EmbeddingProvider):
    # Initialize the OpenAI embeddings client without embedding or downloading anything.
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding and generation.")
        from langchain_openai import OpenAIEmbeddings

        self.client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    # Embed documents in bounded batches to avoid oversized API requests.
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 128):
            vectors.extend(self.client.embed_documents(list(texts[offset : offset + 128])))
        return np.asarray(vectors, dtype=np.float32)

    # Embed one question and return the normalized numeric representation.
    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.client.embed_query(text), dtype=np.float32)
