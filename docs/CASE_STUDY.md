# Case Study: Grounded Hybrid Search for Internal Documentation

## Problem

Internal technical documents contain both conceptual explanations and exact identifiers such as configuration keys and error codes. Dense-only retrieval can miss exact terms, while keyword-only search can miss semantically related passages. A useful answer must also show which source supports it and decline when evidence is weak.

## Solution

This project indexes Markdown, plain text, HTML, and PDF documents. It stores content-addressed raw uploads and immutable, versioned FAISS snapshots; it builds BM25 from the same chunk set. Fixed-size, heading-aware recursive, and embedding-similarity semantic chunking are switchable. Cosine-similarity deduplication avoids storing near-identical chunks.

Queries use dense and sparse top-k retrieval, configurable weighted Reciprocal Rank Fusion, a cross-encoder second pass, and a grounded OpenAI answer with numbered evidence. A judge checks each claim-citation pair and answer completeness. Retrieval, citation coverage, and completeness appear separately in the confidence breakdown. Low-confidence questions receive an explicit abstention.

The workflow uses a Pydantic `BaseModel` as LangGraph state and keeps recent conversation turns bounded. PostgreSQL-backed deployments persist conversation checkpoints; local/test operation uses `MemorySaver`. FastAPI exposes ingestion, asking, comparison, workspace, and document APIs. The React dashboard provides Keycloak sign-in, workspace-scoped chat, citation/evidence review, hybrid-versus-dense comparison, document upload, and workspace administration.

## Evaluation

The checked-in synthetic corpus has 50 hand-authored questions: direct lookups, multi-hop questions, and five questions with no supported answer. Run:

```powershell
python -m rag_pipeline.evaluation.run
```

The runner evaluates all 50 questions with fixed, recursive, and semantic chunking. It reports answer correctness and faithfulness from the configured LLM judge, expected-source retrieval relevance, citation accuracy, and citation coverage in both JSON and Markdown.

**Benchmark results are intentionally not prefilled.** Run the full suite with the desired model, reranker, and corpus to obtain reproducible numbers before making any claim about faithfulness, citation accuracy, or which strategy performs best. The bundled documents are synthetic and are not a substitute for an organization's real evaluation set.

## Four-minute demo walkthrough

1. Start the Compose stack and sign in to the dashboard, then ingest a new Markdown runbook from Documents or run `rag-pipeline-seed`.
2. Ask a direct lookup such as “How long do OAuth access tokens last?” and show the answer, source chunk, verified citation, and confidence dimensions.
3. Ask a multi-part question that joins policies, then inspect the ranked evidence and per-claim citation outcomes.
4. Ask an absent question, such as a private phone number, and show that the system abstains rather than inventing a fact.
5. Compare hybrid retrieval with dense-only retrieval from Chat and explain when exact keyword matching changes the retrieved evidence.
6. Point to `data/evaluation/results/chunking-comparison.md` after running the golden suite; discuss actual observed metrics, not estimates.

## Tradeoffs and next production steps

Local FAISS/BM25 supports native development, while Compose can use Qdrant and PostgreSQL for shared retrieval and durable state. Generation and embeddings use hosted APIs. The project includes Keycloak authentication, workspace authorization, Redis rate limiting/cache, and audited platform-admin access, but local integration and production deployment still require validation. Add organization-specific retention, backup/recovery, observability, and measured capacity controls before processing confidential documents at scale.
