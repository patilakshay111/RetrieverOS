# Platform Operations

## Deployment
Production deployment uses a canary for 10 minutes, then increases traffic to 25%, 50%, and 100% if error rate and latency remain within the service SLO. The release manager can pause or roll back at each stage. The release window is Tuesday and Thursday from 10:00 to 14:00 UTC.

## Backup and recovery
The document index is backed up every six hours and retained for 30 days. Recovery point objective is six hours and recovery time objective is two hours. The service owner must run a restore drill every quarter and record the result in the operations log.

## Capacity
The Search Platform team reviews index storage and query latency weekly. Keep at least 25% free disk capacity. Alert when p95 query latency exceeds 800 milliseconds for ten minutes or when disk usage exceeds 75%.

## Availability targets
The internal documentation API has a monthly availability target of 99.9%. Planned maintenance is excluded when announced at least 48 hours in advance. The measurement window uses successful health checks and excludes client-side network failures.

## Data residency
Production document indexes and backups remain in the configured production region. Cross-region copies require approval from the data protection officer and security lead. Development environments may only use synthetic or explicitly approved redacted content.

## On-call handoff
The outgoing engineer writes a handoff note before the shift ends. The note lists active incidents, recent deployments, known risks, and the next scheduled maintenance. The incoming engineer acknowledges the handoff in the on-call channel.
