# Recreate the Internal Docs Hybrid RAG Project

This guide rebuilds the project from an empty Python folder. Use Python 3.11 or 3.12 for the documented support matrix.

## 1. Create the project and environment

Create a working directory, initialize a virtual environment, and install the project dependencies:

```powershell
mkdir internal-docs-hybrid-rag
cd internal-docs-hybrid-rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Add a `pyproject.toml` with FastAPI/Uvicorn, Streamlit, LangGraph, LangChain OpenAI, LangChain text splitters, FAISS CPU, `rank-bm25`, `sentence-transformers`, Pydantic, `python-dotenv`, BeautifulSoup, and `pypdf`. Add `pytest` and `httpx` as development dependencies, then run:

```powershell
python -m pip install -e ".[dev]"
```

Use a `src/rag_pipeline/` package layout, `tests/` for tests, `docs/` for public documentation, and `data/` for the synthetic corpus and evaluation set. Configure setuptools package discovery to search under `src`; put command-line entrypoints in the package and expose them through `[project.scripts]`.

FAISS and PyTorch platform wheels vary by operating system. If Windows cannot resolve a compatible FAISS wheel, use the supplied Docker image or a supported Linux/WSL environment rather than silently switching index formats.

## 2. Establish configuration and safe defaults

Create `.env.example` and copy it to `.env`. Read `OPENAI_API_KEY`, model names, data directory, chunk sizes, retrieval weights, reranker settings, confidence threshold, upload limit, CORS origins, and API key from environment variables. Never place secrets in source code or commit `.env`. Require an API key of at least 32 characters and non-wildcard CORS origins in production.

For local Streamlit credentials, use `.streamlit/secrets.toml` (ignored by version control); the dashboard also supports environment-based keys for containers.

Use the local filesystem for source files and FAISS snapshots. Require an API key in production, bound document uploads, limit CORS origins, and keep health checks free of secrets.

## 3. Build document loading and chunking

Implement a loader for Markdown, plain text, HTML, and PDF. Convert content to text, strip HTML script/style elements, preserve headings and PDF page numbers, and reject unknown or invalid inputs.

Implement three named strategies:

1. `fixed`: recursive character-size splitting with overlap.
2. `recursive`: heading-aware section splitting with paragraph/sentence boundaries.
3. `semantic`: sentence grouping based on adjacent embedding similarity, bounded by the maximum chunk size.

Record source, source digest, chunk index, section, page, strategy, and character count for every chunk.

## 4. Persist source files and synchronized indexes

Keep raw uploaded bytes in a content-addressed archive so they can be reprocessed. Embed chunks with `text-embedding-3-small`; normalize vectors and use a FAISS inner-product index as cosine retrieval.

Before committing an incoming chunk, compare its vector against the retained corpus and the other accepted chunks. Skip it when cosine similarity exceeds the configured deduplication threshold. Save chunk metadata, vectors, FAISS index, and document inventory in a fresh generation directory; only point `CURRENT` at that generation after every file is written successfully. Rebuild BM25 from those exact chunk records at process startup.

Add a re-index command that verifies each current document's raw-file digest, reloads the original source, and reruns the same chunking/embedding path without requiring an upload. Build/cache BM25 from the active chunk snapshot on the first sparse or hybrid query for that generation.

## 5. Add hybrid retrieval and reranking

For each query, retrieve dense and BM25 top-k results. Fuse their ranked chunk IDs with configurable weighted Reciprocal Rank Fusion, send the top candidates through a cross-encoder, and keep the configured final top-k. Preserve per-stage scores for inspection. Keep dense-only and sparse-only modes available for comparison.

## 6. Add grounded answers and citation checks

Create a deterministic OpenAI chat helper. Tell the model to treat retrieved text as evidence rather than instructions, answer only from evidence, cite every factual claim with a numbered chunk reference, and explicitly abstain when evidence is insufficient.

Parse each claim and cited chunk separately. Ask an LLM judge whether the complete claim is supported, flag invalid or unsupported citations, and calculate claim-level citation coverage. Ask a separate structured judge whether the answer addresses all parts of the question.

## 7. Wire the LangGraph workflow

Define the shared graph state as a Pydantic `BaseModel`. Use small sequential nodes for retrieval, generation/abstention, citation verification, and confidence scoring. Compile with `MemorySaver` and pass a stable `thread_id` in every invocation; retain a bounded recent-turn history so follow-up answers can use earlier context. Use the same state and service from API and evaluation code.

Confidence is a weighted composite of retrieval confidence (40%), verified claim coverage (35%), and answer completeness (25%). Keep the component scores in the response. Below the retrieval threshold, do not call the generator; return a clear no-answer response and the closest documents to check manually.

## 8. Expose the service and dashboard

Create FastAPI endpoints:

- `GET /healthz`
- `GET /v1/documents`
- `POST /v1/ingest`
- `POST /v1/ask`
- `POST /v1/compare`

Validate inputs, restrict uploads to supported formats and a bounded size, authenticate with `X-API-Key` when configured, and document request and response schemas through OpenAPI. Build a Streamlit dashboard that asks questions, displays ranked chunks/citation checks/confidence, and compares hybrid with dense-only retrieval.

## 9. Seed, test, and evaluate

Add a small non-sensitive sample corpus and a hand-written golden dataset of at least 50 questions. Include simple lookups, multi-document reasoning, ambiguous prompts, and questions whose answers are absent. Test chunk provenance, RRF math, deduplication, persistence, index alignment, and citation failures with deterministic fake embeddings and judges.

Run the local checks:

```powershell
pytest
rag-pipeline-seed
rag-pipeline-reindex
python -m rag_pipeline.evaluation.run
```

The evaluation runner should compare fixed, recursive, and semantic chunkers using correctness, faithfulness, retrieval relevance, and citation accuracy. Use an LLM judge for correctness and faithfulness; report its real outputs rather than claiming unmeasured percentages.

## 10. Containerize and demonstrate

Build an API image and a dashboard service with Docker Compose. Mount a persistent volume for raw documents and the versioned FAISS index, pass keys only through environment configuration, and expose API and dashboard ports. Seed after starting the API:

```powershell
docker compose up --build
docker compose exec api rag-pipeline-seed
```

For a short demo, ingest a new runbook, ask a direct lookup, ask a multi-hop question, ask something absent from the corpus, then compare hybrid and dense-only retrieval. Show the citation verification and confidence breakdown. Capture the full evaluation report before making any interview claim about quality.

## Rebuild checklist

- [ ] `.env` is local and `OPENAI_API_KEY` is configured.
- [ ] The data directory is writable and persistent.
- [ ] Sample ingestion reports a nonzero chunk count.
- [ ] `GET /healthz` and `/docs` load.
- [ ] A supported question returns numbered citations and a confidence breakdown.
- [ ] An unanswerable question produces a structured abstention.
- [ ] Dense/hybrid comparison returns both result sets.
- [ ] Unit tests pass and all three evaluation strategies complete.
- [ ] Production mode has an API key, restricted CORS, and a persistent data volume.
