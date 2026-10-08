# Operating model

How the roles on the *Team and operating model* slide map to the platform: what each one owns, which platform
roles they hold, and the tools and runbooks they use. One person can hold two roles on a first engagement;
nobody skips the platform role.

| Team role | Platform roles | Owns during the build | Owns once running | Main tools |
|---|---|---|---|---|
| Solution architect | `platform-admin` (read), `security-admin` (review) | Design, the client's security review (`docs/governance.md`) | Quarterly review of cost, quality and roadmap | dashboard, `/v1/chargeback`, `aiready sizing worksheet`, `aiready cost breakeven` |
| Platform engineer | `platform-admin` | Cluster, GPU serving, packaging (`deploy/`) | On-call, upgrades, capacity | Helm chart, `aiready bundle`, runbooks: add GPU node, model rollout/rollback, restore index |
| Integration engineer | `ai-engineer`, `ingestion-admin` | API facade, MCP server, identity bridge | New tools and connectors | `aiready app init/check`, `config/integration.yaml`, `config/tools.yaml` |
| AI engineer | `ai-engineer`, `reviewer` | Retrieval, prompts, orchestration, evaluation set | Evaluation upkeep, model and prompt changes | `aiready eval run/gate/routing`, prompt registry, review queue, `aiready.finetune` |
| Data engineer | `ingestion-admin` | Ingestion pipelines and access-tag mapping | Pipeline health, new sources | `aiready ingest run/runs/verify/dlq`, lineage API, pipeline metrics |
| Client: product owner | `user` | Use cases, acceptance | Feedback review priorities | review queue labels, golden-set candidates |
| Client: subject experts | `reviewer` | Golden questions, expected sources | Labelling low-scored answers | `config/evals/`, review queue |
| Client: security lead | `security-admin`, `access-reviewer`, `model-approver` | Policies, egress, key custody | Access reviews, incident response, model approvals | `aiready policy test`, `aiready incident ...`, `aiready audit verify/export`, access reviews |

## Recurring duties

| Cadence | Duty | How |
|---|---|---|
| Every release | Evaluation gate, policy tests, signed images | CI (`.github/workflows/ci.yml`, `release-gate.yml`) |
| Daily | Audit chain verification; memory purge; ingestion sync | Helm CronJobs (`aiready audit verify --all-chains`, memory `purge_expired`, ingestion sync) |
| Weekly | Failure review: label low scores, turn failures into test cases | review queue -> `/v1/review/golden-candidates` -> `config/evals/golden/` |
| Monthly | Chargeback by team; budget review | `/v1/chargeback?format=csv`, gateway budgets |
| Quarterly | Access review sign-off; cost/quality/roadmap review; key rotation | `POST /v1/access-reviews`, `aiready token rotate`, `aiready audit rotate-key` |
| On demand | Incidents | `docs/runbooks/incident-response.md` |

Headcount, on-call hours and response times are set per contract.
