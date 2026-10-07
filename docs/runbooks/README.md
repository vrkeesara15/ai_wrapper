# Runbooks

Each runbook says who acts (role from the "Team and operating model" slide), what to check first, the
steps, how to verify, and what to record. Commands use the `aiready` CLI (groups: `serve config token audit
policy registry eval ingest sizing cost bundle incident`). The underlying HTTP endpoints are given where
the CLI is not available. In Kubernetes, run the CLI inside a platform pod so that it has the configuration,
the wiring overlay and a service identity:

```bash
kubectl -n aiready exec -it deploy/observability -- aiready <group> <command> ...
```

| Runbook | Trigger / alert |
|---|---|
| [incident-response.md](incident-response.md) | Suspected misuse, data leak, compromised key, guardrail block spike, `AIReadyErrorRateHigh` |
| [model-rollout-rollback.md](model-rollout-rollback.md) | New model or adapter version, `AIReadyQualityScoreDrop`, `AIReadyTimeToFirstTokenOverBudget` |
| [add-gpu-node.md](add-gpu-node.md) | `AIReadyModelQueueTime`, `AIReadyRequestsWaiting`, `AIReadyGPUMemoryHigh`, sizing worksheet says so |
| [restore-vector-index.md](restore-vector-index.md) | Qdrant data loss or corruption, standby site activation |
| [rotate-signing-keys.md](rotate-signing-keys.md) | Scheduled rotation, suspected key exposure |
| [dlq-replay.md](dlq-replay.md) | `AIReadyDeadLetters` |
| [audit-chain-verification.md](audit-chain-verification.md) | `AIReadyAuditVerificationFailed`, audit or regulator request |

Every runbook ends with the same rule: record the actions in the incident or change ticket, then turn each
failure into a new test case (golden set, safety suite or policy test).
