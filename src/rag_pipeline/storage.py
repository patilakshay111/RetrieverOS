# storage.py: persists source files and atomically versioned FAISS-backed document indexes

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import uuid

import faiss
import numpy as np

from rag_pipeline.chunking import ChunkStrategy, chunk_sections
from rag_pipeline.config import Settings
from rag_pipeline.documents import load_document
from rag_pipeline.embeddings import EmbeddingProvider
from rag_pipeline.models import Chunk, DocumentInfo, IngestionResult


# Keep each active index generation internally consistent across its chunk and vector files.
@dataclass(frozen=True)
class IndexSnapshot:
    generation: str
    chunks: list[Chunk]
    vectors: np.ndarray
    dense_index: faiss.Index | None
    documents: dict[str, DocumentInfo]


# Manage the raw source archive and immutable, atomically published index generations.
class KnowledgeBase:
    # Load the current snapshot or initialize an empty, local FAISS corpus.
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider):
        self.settings = settings
        self.embeddings = embeddings
        self.root = settings.data_dir
        self.raw_dir = self.root / "raw"
        self.snapshots_dir = self.root / "index" / "snapshots"
        self.current_file = self.root / "index" / "CURRENT"
        self._write_lock = threading.RLock()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot = self._load_current()

    # Read a canonical JSON file while rejecting malformed persisted state.
    @staticmethod
    def _read_json(path: Path):
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    # Load the exact committed generation named by the atomic CURRENT pointer.
    def _load_current(self) -> IndexSnapshot:
        if not self.current_file.exists():
            return IndexSnapshot("empty", [], np.empty((0, 0), dtype=np.float32), None, {})
        generation = self.current_file.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[a-f0-9]{32}", generation):
            raise RuntimeError("The index CURRENT pointer is invalid; restore a valid snapshot.")
        directory = self.snapshots_dir / generation
        try:
            chunks = [Chunk.model_validate(item) for item in self._read_json(directory / "chunks.json")]
            documents = {
                item["source"]: DocumentInfo.model_validate(item)
                for item in self._read_json(directory / "documents.json")
            }
            vectors = np.load(directory / "vectors.npy", allow_pickle=False).astype(np.float32)
            dense_index = faiss.read_index(str(directory / "dense.faiss")) if chunks else None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unable to load committed index generation {generation}.") from error
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RuntimeError("Persisted chunks and embeddings are inconsistent.")
        if chunks and (dense_index is None or dense_index.ntotal != len(chunks)):
            raise RuntimeError("Persisted chunks and FAISS index are inconsistent.")
        return IndexSnapshot(generation, chunks, vectors, dense_index, documents)

    # Return the current immutable generation for lock-free concurrent reads.
    def snapshot(self) -> IndexSnapshot:
        return self._snapshot

    # Save the uploaded bytes once under a content-addressed, path-safe name.
    def _save_raw(self, filename: str, digest: str, content: bytes) -> None:
        path = self.raw_path(filename, digest)
        if path.exists():
            return
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    # Resolve a raw source archive path from its content digest and safe basename.
    def raw_path(self, filename: str, digest: str) -> Path:
        safe_name = Path(filename.replace("\\", "/")).name
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name)[:120] or "document"
        return self.raw_dir / f"{digest}_{safe_name}"

    # Publish a complete new generation before switching the atomic CURRENT pointer.
    def _commit(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        documents: dict[str, DocumentInfo],
    ) -> IndexSnapshot:
        generation = uuid.uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.snapshots_dir))
        target = self.snapshots_dir / generation
        index: faiss.Index | None = None
        if len(chunks):
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                raise ValueError("Each chunk must have exactly one embedding vector.")
            index = faiss.IndexFlatIP(int(vectors.shape[1]))
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        try:
            (staging / "chunks.json").write_text(
                json.dumps([item.model_dump(mode="json") for item in chunks], ensure_ascii=False),
                encoding="utf-8",
            )
            (staging / "documents.json").write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in documents.values()],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (staging / "vectors.npy").open("wb") as file:
                np.save(file, vectors, allow_pickle=False)
            if index is not None:
                faiss.write_index(index, str(staging / "dense.faiss"))
            os.replace(staging, target)
            pointer = self.current_file.with_suffix(f".{generation}.tmp")
            pointer.write_text(generation, encoding="utf-8")
            os.replace(pointer, self.current_file)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            committed = (
                self.current_file.exists()
                and self.current_file.read_text(encoding="utf-8").strip() == generation
            )
            if target.exists() and not committed:
                shutil.rmtree(target, ignore_errors=True)
            raise
        return IndexSnapshot(generation, chunks, vectors, index, documents)

    # Replace one named document, chunk it, skip near-duplicates, and commit both indexes together.
    def ingest(self, filename: str, content: bytes, strategy: ChunkStrategy) -> IngestionResult:
        if not content:
            raise ValueError("The uploaded document is empty.")
        digest = hashlib.sha256(content).hexdigest()
        self._save_raw(filename, digest, content)
        sections = load_document(filename, content)
        drafts = chunk_sections(
            sections,
            strategy,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.semantic_similarity_threshold,
            self.embeddings.embed_documents,
        )
        drafts = [draft for draft in drafts if draft.text.strip()]
        if not drafts:
            raise ValueError("The document contains no extractable text.")
        with self._write_lock:
            previous = self._snapshot
            retained = [chunk for chunk in previous.chunks if chunk.source != filename]
            retained_indices = [
                index for index, chunk in enumerate(previous.chunks) if chunk.source != filename
            ]
            retained_vectors = previous.vectors[retained_indices] if retained_indices else np.empty(
                (0, previous.vectors.shape[1] if previous.vectors.ndim == 2 else 0),
                dtype=np.float32,
            )
            new_vectors = self.embeddings.embed_documents([draft.text for draft in drafts])
            if new_vectors.ndim != 2 or new_vectors.shape[0] != len(drafts):
                raise ValueError("Embedding provider returned an invalid document-vector matrix.")
            if len(retained_vectors) and retained_vectors.shape[1] != new_vectors.shape[1]:
                raise ValueError("Embedding dimensions changed; re-index the complete corpus.")
            normalized_new = np.ascontiguousarray(new_vectors, dtype=np.float32)
            faiss.normalize_L2(normalized_new)
            existing_index = None
            if len(retained_vectors):
                normalized_retained = np.ascontiguousarray(
                    retained_vectors, dtype=np.float32
                ).copy()
                faiss.normalize_L2(normalized_retained)
                existing_index = faiss.IndexFlatIP(int(normalized_retained.shape[1]))
                existing_index.add(normalized_retained)
            accepted_chunks: list[Chunk] = []
            accepted_vectors: list[np.ndarray] = []
            accepted_index = None
            skipped = 0
            for index, draft in enumerate(drafts):
                vector = normalized_new[index : index + 1]
                if existing_index is not None:
                    score, _ = existing_index.search(vector, 1)
                    if float(score[0][0]) > self.settings.deduplication_threshold:
                        skipped += 1
                        continue
                if accepted_index is not None:
                    score, _ = accepted_index.search(vector, 1)
                    if float(score[0][0]) > self.settings.deduplication_threshold:
                        skipped += 1
                        continue
                chunk_index = len(accepted_chunks)
                chunk_id = hashlib.sha256(
                    f"{digest}:{strategy}:{chunk_index}:{draft.text}".encode("utf-8")
                ).hexdigest()
                accepted_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=draft.text,
                        source=filename,
                        source_sha256=digest,
                        chunk_index=chunk_index,
                        section_heading=draft.heading,
                        page_number=draft.page_number,
                        chunking_strategy=strategy,
                        character_count=len(draft.text),
                    )
                )
                accepted_vectors.append(vector[0].copy())
                if accepted_index is None:
                    accepted_index = faiss.IndexFlatIP(int(vector.shape[1]))
                accepted_index.add(vector)
            all_chunks = retained + accepted_chunks
            if accepted_vectors:
                inserted_vectors = np.asarray(accepted_vectors, dtype=np.float32)
                all_vectors = (
                    np.concatenate((retained_vectors, inserted_vectors), axis=0)
                    if len(retained_vectors)
                    else inserted_vectors
                )
            else:
                all_vectors = retained_vectors.astype(np.float32, copy=False)
            documents = {
                source: info
                for source, info in previous.documents.items()
                if source != filename
            }
            document = DocumentInfo(
                source=filename,
                source_sha256=digest,
                bytes=len(content),
                chunk_count=len(accepted_chunks),
                chunking_strategy=strategy,
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
            documents[filename] = document
            snapshot = self._commit(all_chunks, all_vectors, documents)
            self._snapshot = snapshot
            return IngestionResult(
                document=document,
                inserted_chunks=len(accepted_chunks),
                duplicate_chunks_skipped=skipped,
            )

    # Return the document inventory in stable source-name order.
    def list_documents(self) -> list[DocumentInfo]:
        return sorted(self._snapshot.documents.values(), key=lambda item: item.source.casefold())
