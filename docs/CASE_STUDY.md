# Case Study: Grounded Hybrid Search for Internal Documentation

## Problem

Internal technical documents contain both conceptual explanations and exact identifiers such as configuration keys and error codes. Dense-only retrieval can miss exact terms, while keyword-only search can miss semantically related passages. A useful answer must also show which source supports it and decline when evidence is weak.

## Solution

This project indexes Markdown, plain text, HTML, and PDF documents. It stores content-addressed raw uploads and immutable, versioned FAISS snapshots; it builds BM25 from the same chunk set. Fixed-size, heading-aware recursive, and embedding-similarity semantic chunking are switchable. Cosine-similarity deduplication avoids storing near-identical chunks.

Queries use dense and sparse top-k retrieval, configurable weighted Reciprocal Rank Fusion, a cross-encoder second pass, and a grounded OpenAI answer with numbered evidence. A judge checks each claim-citation pair and answer completeness. Retrieval, citation coverage, and completeness appear separately in the confidence breakdown. Low-confidence questions receive an explicit abstention.

The workflow uses a Pydantic `BaseModel` as LangGraph state and `MemorySaver` for in-process conversation checkpoints, bounded to 20 turns. FastAPI exposes ingestion, asking, comparison, and document inventory endpoints; Streamlit exposes ranked chunks, citation checks, and hybrid-versus-dense comparison.

## Evaluation

The checked-in synthetic corpus has 50 hand-authored questions: direct lookups, multi-hop questions, and five questions with no supported answer. Run:

```powershell
python -m rag_pipeline.evaluation.run
```

The runner evaluates all 50 questions with fixed, recursive, and semantic chunking. It reports answer correctness and faithfulness from the configured LLM judge, expected-source retrieval relevance, citation accuracy, and citation coverage in both JSON and Markdown.

**Benchmark results are intentionally not prefilled.** Run the full suite with the desired model, reranker, and corpus to obtain reproducible numbers before making any claim about faithfulness, citation accuracy, or which strategy performs best. The bundled documents are synthetic and are not a substitute for an organization's real evaluation set.

## Four-minute demo walkthrough

1. Start the API and dashboard, then ingest a new Markdown runbook through `POST /v1/ingest` or `rag-pipeline-seed`.
2. Ask a direct lookup such as “How long do OAuth access tokens last?” and show the answer, source chunk, verified citation, and confidence dimensions.
3. Ask a multi-part question that joins policies, then inspect the ranked evidence and per-claim citation outcomes.
4. Ask an absent question, such as a private phone number, and show that the system abstains rather than inventing a fact.
5. Enable “Compare hybrid retrieval with dense-only retrieval” and explain when exact keyword matching changes the retrieved evidence.
6. Point to `data/evaluation/results/chunking-comparison.md` after running the golden suite; discuss actual observed metrics, not estimates.

## Tradeoffs and next production steps

FAISS keeps a demo self-contained and local, following the interview rules' preference over the project brief's optional ChromaDB/Qdrant choice. Generation and embeddings remain hosted APIs. `MemorySaver` is process-local and is intended for a single demo instance; a multi-worker deployment should use a durable, shared checkpointer. Add organization-specific identity integration, rate limiting, audit policy, retention controls, backup monitoring, and a managed vector service before processing real confidential documents at scale.
