# Engineering Handbook

## Code review
Every production change requires one approval from a code owner and passing unit tests. Changes to authentication, authorization, encryption, or key handling require two security-team approvals. Reviewers should check backward compatibility, failure handling, and observability before approval.

## Branch and release workflow
Feature branches use the prefix `feature/` and bug-fix branches use `fix/`. Pull requests target `main`. A release is created from a reviewed commit after the continuous integration workflow passes. Production releases are deployed on Tuesdays and Thursdays between 10:00 and 14:00 UTC.

## Continuous integration
The CI workflow runs formatting, static type checks, unit tests, dependency vulnerability scanning, and a container build. A failed required check blocks merging. The team supports Python 3.11 and 3.12. The build cache is retained for seven days.

## Python conventions
Use Python 3.11 or newer, type annotate public interfaces, and prefer small functions with one responsibility. Read configuration from environment variables, never commit credentials, and use structured logging rather than print statements in services.

## Service ownership
The Identity Platform team owns the identity gateway. The Data Foundations team owns the document ingestion service. The Search Platform team owns the shared search cluster. Ownership is recorded in the service catalog and reviewed quarterly.

## API conventions
Internal APIs use versioned paths beginning with `/v1`. Requests must have a bounded body size and a request identifier. Clients should use exponential backoff with jitter for retryable 429 and 503 responses, and must not retry other 4xx responses.

## Observability
Services emit JSON logs with a request ID, service name, and severity. Metrics are exported every 30 seconds. An alert should link to the relevant dashboard and runbook. Do not log access tokens, passwords, private keys, or document contents.
