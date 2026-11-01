# reindex.py: rebuilds indexed chunks and embeddings from the archived original source files

import argparse
import hashlib
import json

from rag_pipeline.config import Settings
from rag_pipeline.models import AccessContext
from rag_pipeline.service import RAGService


# Rebuild the active index from each current document's verified raw source bytes.
def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index archived source documents.")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"])
    arguments = parser.parse_args()
    service = RAGService(Settings.from_env())
    documents = service.knowledge_base.list_documents()
    if not documents:
        raise RuntimeError("No indexed document manifest is available to re-index.")
    results = []
    for document in documents:
        source_path = service.knowledge_base.raw_path(
            document.source, document.source_sha256
        )
        content = source_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != document.source_sha256:
            raise RuntimeError(f"Raw source digest mismatch for {document.source}.")
        strategy = arguments.strategy or document.chunking_strategy
        results.append(
            service.knowledge_base.ingest(
                document.source,
                content,
                strategy,
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


# Run the re-index command when this script is executed directly.
if __name__ == "__main__":
    main()
