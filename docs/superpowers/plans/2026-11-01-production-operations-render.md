# Production Operations and Render Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-safe observability, security/evaluation quality gates, reliable local operations, and a documented Render validation after local completion.

**Architecture:** Instrument FastAPI and its dependencies with redacted OpenTelemetry; use deterministic security/offline tests as required CI gates, keep paid model judging separately controlled, and validate Docker Compose before preparing a Render deployment test. The test target is 100 registered accounts; load/capacity claims wait for explicit concurrency and latency objectives.

**Tech Stack:** OpenTelemetry Python, pytest, GitHub Actions, Docker Compose, Render deployment configuration, documented external OIDC/data-service settings.

**Spec:** [2026-11-01-multiuser-web-dashboard-design.md](../specs/2026-11-01-multiuser-web-dashboard-design.md)

## Global Constraints

- Never emit prompts, documents, answers, API keys, OIDC tokens, or personal claims in logs or trace attributes.
- External telemetry export is opt-in until retention/export policy is selected.
- CI runs tests before image build; no deployment may run when CI fails.
- Model-judge quality gates must be reproducible, reviewed, and explicitly cost-controlled.
- Docker Compose is the local integration path; Render validation is after full local smoke tests.
- Keep one API service with a durable attached disk for the initial Render test; shared object storage is required before horizontal API scaling.
- No Kubernetes or Helm configuration.
- 100 registered accounts is not a concurrency or throughput claim.

## Review Focus

- Malformed or nested sensitive fields must be redacted before export; test representative request/exception payloads.
- Telemetry exporter misconfiguration must not leak data or silently report success; test disabled, invalid, and reachable exporter cases.
- CI must fail on Python/frontend test, migration, Compose, or image-build failure; test workflow dependency ordering.
- Evaluation data should not be sent to a paid external model in normal PR CI; test default workflow has no model credentials or calls.
- Render health/readiness and callbacks must use public HTTPS URLs and exact origins; test production configuration rejects local HTTP and wildcard CORS.

---

### Task 1: Add redacted tracing and structured operational metrics

**Files:**
- Create: `src/rag_pipeline/observability.py`
- Modify: `src/rag_pipeline/config.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `pyproject.toml`
- Test: `tests/test_core.py`

**Interfaces:**
- Produce `configure_observability(settings)` with explicit service name, exporter opt-in, sampling, and allowlisted attributes.
- Request spans may include route, status, duration, and correlation ID only; do not attach request/response bodies or authentication headers.

- [ ] **Step 1: Write failing redaction tests** for nested prompts, document text, bearer tokens, API keys, exception values, and safe route/status attributes.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k observability`; confirm current instrumentation does not satisfy the expectations.
- [ ] **Step 3: Implement** instrumentation and allowlist/redaction before exporter handoff; keep exporting disabled by default.
- [ ] **Step 4: Run** focused tests and inspect captured span attributes for forbidden fields.
- [ ] **Step 5: Commit** instrumentation and settings.

### Task 2: Add security and RAG evaluation quality gates

**Files:**
- Modify: `tests/test_core.py`
- Modify: `src/rag_pipeline/evaluation/run.py`
- Create: `tests/test_security_boundaries.py`
- Modify: `.github/workflows/tests.yml`

- [ ] **Step 1: Write failing tests** for prompt injection as untrusted evidence, cross-user/workspace retrieval/cache leakage, ACL changes, and model-judge opt-in behavior.
- [ ] **Step 2: Run** `python -m pytest tests/test_security_boundaries.py`; confirm each test fails until behavior is implemented.
- [ ] **Step 3: Implement** deterministic isolation/security tests and a separate manually triggered or explicitly credentialed evaluation job; do not add paid LLM calls to ordinary PR checks.
- [ ] **Step 4: Run** offline suite and workflow lint/parser; confirm CI runs tests before image build and no secret is required for default tests.
- [ ] **Step 5: Commit** security tests and workflow changes.

### Task 3: Verify Compose as the production-like local stack

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `.env.example`
- Modify: `scripts/init_local_env.py`
- Modify: `README.md`
- Modify: `docs/PROJECT_RECREATION_GUIDE.md`

- [ ] **Step 1: Add integration smoke tests/checks** for service health, Alembic ordering, secret generation refusing overwrite, and two-user isolation through the browser/API.
- [ ] **Step 2: Run** `docker compose --env-file .env.example config --quiet` and check `.env.example` contains placeholders only.
- [ ] **Step 3: Implement** compose health/readiness, persistent data volumes, resource/runtime security defaults, and a reproducible local smoke-test script; no Kubernetes additions.
- [ ] **Step 4: Run** Compose build and smoke test when Docker Engine is available. If unavailable, record that limitation and do not claim stack validation.
- [ ] **Step 5: Commit** verified local operations and docs.

### Task 4: Prepare and run the deferred Render validation

**Files:**
- Create or modify: `render.yaml` only if a verified blueprint supports the selected external services
- Create: `docs/RENDER_DEPLOYMENT_AND_VALIDATION.md`
- Modify: `.env.example` only for non-secret variable names/defaults

- [ ] **Step 1: Confirm** exact Render plan/service limits, public dashboard/API origins, persistent PostgreSQL/vector/cache choices, external HTTPS OIDC issuer, Google OAuth redirect URI, SMTP, and secret values with the owner before deployment.
- [ ] **Step 2: Add a configuration validation test** rejecting missing HTTPS issuer, wildcard CORS, missing storage/encryption settings, or local callback URLs for production.
- [ ] **Step 3: Implement** only a Render blueprint/config that matches the verified architecture; never commit secrets. If a required service cannot be hosted or configured safely on Render, document the blocker rather than substitute a mock.
- [ ] **Step 4: Run** after local Compose passes: deploy through the owner's Render account, verify health/readiness, login/register, key storage, workspace ACLs, conversations, retrieval, and audit/trace redaction; record observed limits and failures.
- [ ] **Step 5: Commit** deployment docs/config and measured test results; do not claim production readiness if any acceptance criteria remain unverified.

### Task 5: Define and run capacity testing only after targets are specified

**Files:**
- Create: `tests/load/` scenario and runner
- Create: `docs/LOAD_TEST_RESULTS.md`
- Modify: `.private/PRODUCTION_READINESS_DECISIONS.md`

- [ ] **Step 1: Obtain owner-approved peak concurrency, requests/second, upload/query distribution, latency percentiles, and error-rate target for 100 registered accounts.**
- [ ] **Step 2: Add a repeatable load scenario** with synthetic users/documents and no production credentials embedded in fixtures.
- [ ] **Step 3: Run** against the approved Render test deployment and capture service/resource/test-tool configuration.
- [ ] **Step 4: Report** actual p50/p95/p99 latency, throughput, errors, resource use, and bottlenecks; distinguish account count from simultaneous load.
- [ ] **Step 5: Commit** the measured report and update the readiness decisions; make no unsupported scaling claims.
