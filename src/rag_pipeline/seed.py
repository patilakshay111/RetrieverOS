# seed.py: indexes the bundled sample documentation into the configured local FAISS store

import argparse
import json
from pathlib import Path

from rag_pipeline.config import Settings
from rag_pipeline.service import RAGService


# Seed or refresh the sample corpus with the selected chunking strategy.
def main() -> None:
    parser = argparse.ArgumentParser(description="Index the sample internal documentation.")
    parser.add_argument(
        "--strategy", choices=["fixed", "recursive", "semantic"], default="recursive"
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    service = RAGService(Settings.from_env())
    results = []
    for path in sorted((project_root / "data" / "sample_docs").glob("*")):
        if path.is_file():
            results.append(
                service.knowledge_base.ingest(path.name, path.read_bytes(), arguments.strategy)
            )
    print(json.dumps([item.model_dump(mode="json") for item in results], indent=2))


# Run the seed command when this script is executed directly.
if __name__ == "__main__":
    main()
