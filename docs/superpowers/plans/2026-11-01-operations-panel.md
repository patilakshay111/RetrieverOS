# Read-only Operations Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Streamlit Operations panel showing `/readyz` dependency states and a verified dry-run reindex report.

**Architecture:** Keep public readiness behavior unchanged. Add an authenticated GET endpoint that requires `platform_admin`, verifies archived source digests without ingesting, and returns only safe metadata. The Streamlit UI fetches both reports using its existing API credentials and offers no mutation controls.

**Tech Stack:** Python, FastAPI, Streamlit, Requests, pytest.

**Spec:** Approved Part 5 Operations UI request in the current task.

## Global Constraints

- Never execute CLI or shell commands from Streamlit.
- Never expose document contents or secrets in the dry-run endpoint.
- Do not add any reindex action or modify/delete index data.
- Preserve unrelated and pre-existing worktree changes; do not run Docker integration.

## Review Focus

- A non-platform-admin identity must receive 403; missing authentication must retain the existing authentication response.
- Missing manifests and digest mismatches must fail explicitly without returning source bytes or sensitive details.
- Successful dry-run output must state that it is read-only and contain verified source metadata only.
- The UI must report unavailable dependency states and dry-run errors accessibly without exposing response bodies.

---

### Task 1: Add the read-only Operations report

**Files:**
- Modify: `src/rag_pipeline/api.py` (new admin-only report GET endpoint; do not change `/readyz`)
- Modify: `src/rag_pipeline/ui.py` (render readiness and dry-run metadata)
- Test: `tests/test_core.py`

**Interfaces:**
- Endpoint: `GET /v1/admin/operations/reindex-dry-run`, authenticated through the existing `authenticate` dependency and restricted to `VerifiedIdentity.platform_admin`.
- Response: dry-run/read-only status plus verified per-document source name, digest, byte count, and chunking strategy; never source bytes, credentials, owner identifiers, or role values.
- UI: fetches `/readyz` and the report using existing request headers; has no reindex mutation action.

- [ ] Write API tests for admin authorization, verified metadata-only output, and absence of ingest calls; include digest failure behavior.
- [ ] Run those focused tests and confirm they fail before implementation.
- [ ] Implement the minimal report endpoint without changing readiness logic or invoking the CLI.
- [ ] Add the accessible Operations panel, including both healthy and unavailable readiness states and explicit request errors.
- [ ] Run focused API tests and the applicable UI test; then run the relevant existing test suite.
