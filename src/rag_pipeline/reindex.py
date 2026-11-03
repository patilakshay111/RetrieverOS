# reindex.py: rebuilds indexed chunks and embeddings from the archived original source files

import argparse
import hashlib
import json
import re

from rag_pipeline.config import Settings
from rag_pipeline.models import AccessContext, DocumentInfo
from rag_pipeline.service import RAGService
from rag_pipeline.storage import KnowledgeBase


def verify_archived_sources(
    knowledge_base: KnowledgeBase,
) -> list[tuple[DocumentInfo, int]]:
    documents = knowledge_base.list_documents()
    if not documents:
        raise RuntimeError("No indexed document manifest is available to re-index.")

    sources: list[tuple[DocumentInfo, int]] = []
    for document in documents:
        if not re.fullmatch(r"[a-f0-9]{64}", document.source_sha256):
            raise RuntimeError(f"Raw source digest mismatch for {document.source}.")
        source_path = knowledge_base.raw_path(
            document.source, document.source_sha256
        )
        source_digest = hashlib.sha256()
        byte_count = 0
        with source_path.open("rb") as source_file:
            while block := source_file.read(1024 * 1024):
                source_digest.update(block)
                byte_count += len(block)
        if source_digest.hexdigest() != document.source_sha256:
            raise RuntimeError(f"Raw source digest mismatch for {document.source}.")
        sources.append((document, byte_count))
    return sources


# Rebuild the active index from each current document's verified raw source bytes.
def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index archived source documents.")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify archived source digests and report planned documents without changing the index.",
    )
    arguments = parser.parse_args()
    service = RAGService(Settings.from_env())
    try:
        verified_sources = verify_archived_sources(service.knowledge_base)
        if arguments.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "documents": [
                            {
                                "source": document.source,
                                "source_sha256": document.source_sha256,
                                "bytes": byte_count,
                                "workspace_id": document.workspace_id,
                                "owner_id": document.owner_id,
                                "required_roles": document.required_roles,
                                "chunking_strategy": (
                                    arguments.strategy
                                    or document.chunking_strategy
                                ),
                            }
                            for document, byte_count in verified_sources
                        ],
                    },
                    indent=2,
                )
            )
            return
        results = []
        for document, _byte_count in verified_sources:
            content = service.knowledge_base.raw_path(
                document.source, document.source_sha256
            ).read_bytes()
            if hashlib.sha256(content).hexdigest() != document.source_sha256:
                raise RuntimeError(
                    f"Raw source digest mismatch for {document.source}."
                )
            results.append(
                service.knowledge_base.ingest(
                    document.source,
                    content,
                    arguments.strategy or document.chunking_strategy,
                    access_context=AccessContext(
                        user_id=document.owner_id,
                        workspace_id=document.workspace_id,
                        roles=frozenset(document.required_roles),
                        platform_admin=True,
                    ),
                    required_roles=document.required_roles,
                )
            )
        print(json.dumps([item.model_dump(mode="json") for item in results], indent=2))
    finally:
        service.close()


# Run the re-index command when this script is executed directly.
if __name__ == "__main__":
    main()
