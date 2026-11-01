# Security Policy

## Authentication
Service-to-service requests use short-lived OAuth 2.0 access tokens. Tokens expire after 15 minutes. Services validate token issuer, audience, signature, and expiry on every request. Long-lived personal access tokens are prohibited in production.

## Secrets
Production secrets are stored in the managed secrets vault and rotated every 90 days. Applications retrieve secrets through workload identity at startup; secrets must not be copied into container images, source control, logs, or developer chat.

## Data classification
Internal documents are classified as Public, Internal, Confidential, or Restricted. Restricted data may only be processed by approved services in the production region. Logs must not contain document text or personal data. Retention for query logs is 30 days.

## Incident response
Report a suspected security incident to the on-call security engineer within 15 minutes using the incident hotline. Preserve relevant timestamps and request IDs. Do not delete, alter, or publicly discuss evidence. The incident commander owns external communications.

## Dependency management
Critical dependency vulnerabilities must be triaged within 24 hours and remediated within 72 hours. High vulnerabilities must be triaged within three business days. Exceptions require a documented risk acceptance from the security lead and expire after 30 days.

## Access reviews
Production access is granted through the access portal and requires a manager approval. Privileged access is time-bound to four hours and protected by multifactor authentication. Team owners review active production access every quarter.
