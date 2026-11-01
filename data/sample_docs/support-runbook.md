# Support Runbook

## Severity levels
SEV1 means a critical customer-facing service is unavailable or customer data may be exposed. SEV2 means a major feature is degraded for multiple customers. SEV3 means a limited feature is impaired with a workaround. SEV4 is a low-impact question or cosmetic defect.

## Escalation
For a SEV1, page the primary on-call engineer immediately and notify the incident commander within five minutes. For SEV2, page the owning team's on-call within 15 minutes. For SEV3 and SEV4, create a support ticket and assign the owning team by the next business day.

## Incident updates
The incident commander posts a customer-safe status update every 30 minutes during a SEV1 or SEV2. Updates include impact, mitigation, and the next update time. Do not include personal data, credentials, or speculative root causes.

## Rollback
The service owner may roll back to the last healthy release when an active deployment causes a SEV1 or SEV2. Record the deployment identifier and rollback time in the incident channel. Verify recovery with the service dashboard before resolving the incident.

## Search service troubleshooting
For error `SEARCH-429`, check the request-rate dashboard and apply exponential backoff with jitter. For `SEARCH-503`, check the cluster health panel and escalate to Search Platform on-call if the service remains unhealthy for five minutes. Never increase client retries without checking the retry budget.

## Document ingestion troubleshooting
For error `INGEST-413`, confirm the upload is no larger than 10 MB and split large files. For `INGEST-422`, verify the file is a supported text, Markdown, HTML, or PDF document and that its content can be decoded. Re-ingest after correcting the source file.
