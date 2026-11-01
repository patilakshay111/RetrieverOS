# chunking.py: implements fixed, heading-aware recursive, and embedding-based semantic chunking

from collections.abc import Callable
import re
from typing import Literal

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.documents import DocumentSection


ChunkStrategy = Literal["fixed", "recursive", "semantic"]


# Store intermediate chunk text with its originating heading and PDF page.
class ChunkDraft:
    # Create one unpersisted chunk draft.
    def __init__(self, text: str, heading: str | None, page_number: int | None):
        self.text = text.strip()
        self.heading = heading
        self.page_number = page_number


# Split sections using LangChain's recursive splitter while preserving metadata.
def _split_sections(
    sections: list[DocumentSection],
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True,
    )
    chunks: list[ChunkDraft] = []
    for section in sections:
        for document in splitter.create_documents([section.text]):
            chunks.append(ChunkDraft(document.page_content, section.heading, section.page_number))
    return chunks


# Split semantic units when adjacent sentence embeddings indicate a topic change.
def _semantic_split(
    sections: list[DocumentSection],
    chunk_size: int,
    chunk_overlap: int,
    threshold: float,
    embed_documents: Callable[[list[str]], np.ndarray],
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for section in sections:
        sentences = [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+|\n+", section.text)
            if value.strip()
        ]
        if not sentences:
            continue
        vectors = embed_documents(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for index, sentence in enumerate(sentences[1:], start=1):
            previous = vectors[index - 1]
            current = vectors[index]
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
            similarity = float(np.dot(previous, current) / denominator) if denominator else 0.0
            proposed = " ".join(groups[-1] + [sentence])
            if similarity < threshold or len(proposed) > chunk_size:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        for group_index, group in enumerate(groups):
            text = " ".join(group)
            if len(text) > chunk_size:
                pieces = _split_sections(
                    [DocumentSection(section.heading, text, section.page_number)],
                    chunk_size,
                    chunk_overlap,
                    ["\n\n", "\n", ". ", " ", ""],
                )
                chunks.extend(pieces)
            else:
                chunks.append(ChunkDraft(text, section.heading, section.page_number))
            if group_index and chunk_overlap:
                previous_tail = " ".join(groups[group_index - 1])
                overlap = previous_tail[-chunk_overlap:].strip()
                if overlap and chunks:
                    chunks[-1].text = f"{overlap} {chunks[-1].text}".strip()
    return chunks


# Dispatch to one of the three named, configurable chunking strategies.
def chunk_sections(
    sections: list[DocumentSection],
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    semantic_threshold: float,
    embed_documents: Callable[[list[str]], np.ndarray],
) -> list[ChunkDraft]:
    if strategy == "fixed":
        return _split_sections(
            sections, chunk_size, chunk_overlap, ["\n\n", "\n", " ", ""]
        )
    if strategy == "recursive":
        return _split_sections(
            sections, chunk_size, chunk_overlap, ["\n\n", "\n", ". ", " ", ""]
        )
    if strategy == "semantic":
        return _semantic_split(
            sections, chunk_size, chunk_overlap, semantic_threshold, embed_documents
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")
