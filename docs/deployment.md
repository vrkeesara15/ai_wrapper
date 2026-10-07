# Deploying AI-Ready

AI-Ready runs **beside** the existing application. The application is not modified: it gets an assistant
widget, an identity bridge and an API facade, and everything else runs in zones of its own. The architecture
is the same at every stage. Only the footprint changes.

| | Stage 1 - Pilot | Stage 2 - Production | Stage 3 - Platform |
|---|---|---|---|
| Runtime | Containers on a single host (`deploy/compose`) | Kubernetes with autoscaling (`deploy/helm/aiready`, `values-production.yaml`) | Same chart, plus `values-platform.yaml` |
| Models | One GPU node (vLLM + TEI) or a private endpoint, or the mock server | GPU pool A (chat, plus a second tier for harder tasks), GPU pool B (router, guard, embeddings, rerank) | One GPU pool shared across teams; idle tiers scale to zero; domain adapters |
| Redundancy | None (one host) | Gateway at least 3 replicas, PDBs, 3 zones, managed DB/Redis, 3 vector-DB replicas | As stage 2, with larger ceilings |
| Scaling | Manual | HPA on CPU, KEDA on queue depth (embedding, ingestion workers, agent runs, model pools) | KEDA scale-to-zero (`idleReplicas: 0`) for idle GPU tiers and ingestion workers |
| Move on when | Evaluation scores hold and people use it every week | A second team or application asks for access | Capacity follows measured demand |

Sizing: never quote node counts before a load test. Follow the "Capacity and sizing method": measure demand
(peak users x requests per user per minute x tokens), measure the tokens per second one GPU node actually
sustains on the client's hardware, then take nodes for load = demand / supply, plus one node so a failure is
not an outage, plus headroom. Repeat for each pool. `aiready sizing worksheet` fills in the worksheet, and
`aiready cost breakeven` compares self-hosting with an API.

## Repository map

| Path | What it is |
|---|---|
| `Dockerfile`, `.dockerignore` | One image for every service: multi-stage, Python 3.12 slim, uid 10001, read-only root, healthcheck, OCI labels |
| `Makefile` | `install test lint run-pilot build sign bundle helm-lint eval-gate ...` (`make help`) |
| `deploy/compose/` | Stage 1: `docker-compose.yml`, `.env.example`, `init-secrets.sh`, Postgres init SQL |
| `deploy/config-overlays/{compose,k8s}/services.yaml` | Service wiring overlays (`AIREADY_CONFIG_OVERLAY`), model endpoints, Kubernetes audit sink |
| `deploy/helm/aiready/` | Stages 2 and 3: the chart, `values-pilot.yaml`, `values-production.yaml`, `values-platform.yaml` |
| `deploy/terraform/` | Cloud topology on AWS, with a table of Azure and GCP equivalents |
| `deploy/onprem/` | On-prem topology: signed air-gapped release bundles (build, verify, import) and backup/restore scripts |
| `deploy/observability/` | OTel collector (strips personal data), Prometheus with alert rules, Tempo, Loki, Grafana |
| `deploy/security/` | Squid egress allow-list, Gatekeeper and Kyverno admission policies, Vault policies |
| `.github/workflows/` | `ci.yml`, `release-gate.yml`, `deploy.yml` |
| `docs/runbooks/` | Operational runbooks |

## Zone model

The zones come from the "Security and governance" slide. They are enforced as Docker networks in stage 1
and as NetworkPolicies (default deny) in stages 2 and 3. Every pod is labelled `aiready.io/zone`.

| Zone | Contents | Allowed in | Allowed out |
|---|---|---|---|
| app | assistant API/widget, identity bridge, API facade (and, in the pilot, the sample legacy app) | ingress controller, existing app CIDRs, platform | platform; existing app CIDRs only for pods labelled `aiready.io/reaches-app` |
| platform | gateway (+ admin), guardrails, governance, embedding, rerank, search, rag, memory, query, agents, MCP, ingestion, observability, OTel collector, batch jobs | app, platform | platform, model, data (in-cluster + `dataCidrs`), monitoring namespace; the egress proxy only for pods labelled `aiready.io/egress` |
| model | vLLM / TEI model servers | platform only | **nothing**: no DNS, no internet. Weights come from the mounted, checksummed model store |
| data | Qdrant (and Redis in the pilot); RDS / ElastiCache / on-prem DB outside the cluster | platform | data peers (replication) |
| egress | Squid egress proxy | platform pods with `egress: true` | internet on port 443 only, for allow-listed domains only. Private ranges and metadata endpoints are blocked |

A public model route needs **two** reviewed changes: a governance policy entry naming the data classes that
may leave (`config/policies/egress.yaml`), and an entry in the proxy allow-list (`egressProxy.allowlist` /
`deploy/security/egress-proxy/allowlist.txt`). Both are empty by default.

Pods follow the restricted Pod Security Standard: `runAsNonRoot`, read-only root filesystem, all
capabilities dropped, seccomp `RuntimeDefault`, and `automountServiceAccountToken: false`. Each workload has
its own ServiceAccount, and each platform pod mounts only the secret keys it lists in `components.<name>.secrets`.
Label the namespace `pod-security.kubernetes.io/enforce=restricted`, and apply `deploy/security/kyverno`
(or Gatekeeper) so that only signed images from the internal registry are admitted.

## Stage 1 - Pilot (single host)

Prerequisites: Docker with Compose v2. For `make run-pilot-gpu` you also need one NVIDIA GPU node with the
NVIDIA container toolkit.

```bash
make run-pilot            # creates deploy/compose/.env and secrets/, builds the image, starts everything
make run-demo             # also starts the sample legacy app (examples/legacy_app) on :8000
make run-pilot-gpu        # vLLM chat model + TEI embeddings/reranker instead of the mock model server
```

* Published ports bind to `127.0.0.1` (`AIREADY_BIND`): gateway :8080, assistant :8130, identity bridge
  :8150, Grafana :3000, admin :8081 (loopback only).
* GPU profile: the model zone has no outbound network, so put the weights in `MODEL_STORE_DIR`
  (`./models/{chat,embed,rerank}`) first. Weights for a site come from the same signed bundle used on-prem
  (`deploy/onprem/build-bundle.sh --models ...`).
* Private endpoint instead of a GPU: set `AIREADY_MODEL_CHAT_URL` in `.env` to the endpoint. Then add its
  domain to the egress allow-list and add the matching policy entry.
* State: Postgres (with pgvector), Redis and Qdrant on volumes. Secrets are files in
  `deploy/compose/secrets/`, mounted at `/run/secrets`.
* Each service writes its own audit hash chain under `/var/lib/aiready/audit/<service>.jsonl`. Run
  `aiready audit verify` to check them.
* Pilot caveat: the containers that publish ports sit on the app-zone bridge, which has a route out. Restrict
  that bridge's outbound traffic in the host firewall (`DOCKER-USER` chain). Stage 2 enforces the zones
  strictly.

## Stage 2 - Production (Kubernetes)

1. **Infrastructure.** Cloud: `deploy/terraform/aws` creates a private VPC across 3 AZs with no public
   addresses, EKS with a private endpoint, node groups `cpu`, `gpu-a`, `gpu-b` and `batch` (spot), KMS keys,
   Secrets Manager, ECR, RDS Postgres, ElastiCache Redis, S3 and VPC endpoints. The optional Bedrock endpoint
   is off unless you enable it. On-prem: three control-plane nodes, CPU workers, GPU workers A and B, batch
   workers, Vault with HSM, and an internal registry and model store (`deploy/onprem/README.md`).
2. **Cluster add-ons:** NVIDIA GPU operator (time-slicing or MIG on `gpu-b`), an internal ingress controller
   (`ingress.className`), KEDA, kube-prometheus-stack, External Secrets Operator or the Vault agent injector,
   Kyverno or Gatekeeper.
3. **Model store:** a ReadOnlyMany PVC (`modelServing.modelStore.existingClaim`) holding `<model>/<revision>/`
   with `SHA256SUMS`. `import-bundle.sh` writes it on-prem; in the cloud, an S3 sync job writes it. The
   `verify-weights` init container refuses to start a server whose weights fail the checksum.
4. **Secrets:** `secrets.mode=externalSecrets` (Terraform names them `aiready/<key>`), `vault` (agent
   injector renders `/run/secrets/<key>`; Transit for signing), or `existingSecret`.
5. **Install:**
   ```bash
   helm upgrade --install aiready deploy/helm/aiready -n aiready --create-namespace \
     -f deploy/helm/aiready/values-production.yaml \
     --set global.imageRegistry=<registry> --set image.digest=sha256:<signed digest>
   ```
   Edit `state.*`, `networkPolicy.appCidrs` and `networkPolicy.dataCidrs`, `keda.redisAddress`, the IRSA
   annotations and the ingress hosts for the site.
6. **Configuration** is the repository `config/` directory, packaged into the chart and mounted at
   `/etc/aiready/config`. The k8s overlay is merged last. Services reload changes without a restart. A
   change goes through git, then CI, then `helm upgrade`. Use `config.existingConfigMap` for a GitOps-managed
   copy.

Autoscaling: the gateway, guardrails and rag scale on CPU (HPA). Embedding scales on `aiready_queue_depth`.
Ingestion workers scale on the `aiready:q:ingestion.documents` list, and agents on `aiready:q:agent_runs`.
Model pools scale on `vllm:num_requests_waiting`. Bulk ingestion is queued behind live queries, and every
pool keeps at least one replica warm (except stage 3 tiers marked `idleReplicas: 0`).

Batch CronJobs run on the batch pool: `ingestion-sync`, `eval-nightly`, `audit-verify` (hourly),
`memory-purge`, `backup-postgres`, `backup-qdrant` (daily snapshots) and `backup-config`. Back up to the
`batch.backupTarget` PVC, and replicate it off-site.

## Stage 3 - Platform

Layer `values-platform.yaml` over `values-production.yaml`:

* **Shared GPU pool.** Every team's routes reach the same model Deployments through the gateway. Budgets,
  policies and chargeback separate the teams; the infrastructure is shared.
* **Scale to zero.** The large tier has `idleReplicas: 0`; KEDA wakes it when requests queue for it.
  Ingestion workers also go to 0 when there is no backlog.
* **Adapters.** These are LoRA adapters trained by the fine-tuning Job (`fineTuning.enabled`,
  `fineTuning.runId`) on the off-peak pool. An adapter passes the release gate and gets human approval in
  the adapter registry, then it is served on the frozen base model (`adapters:`). Rollback to the base model
  is one routing change.

## Release pipeline

`ci.yml` runs ruff, pytest, the policy tests (`aiready policy test config/policies/tests`), checks the
deployment assets (YAML, shellcheck, helm lint, kubeconform, compose config, terraform validate), then the
image build, a trivy scan, the SBOM and the cosign signature. `release-gate.yml` runs `aiready eval run` and
`aiready eval gate`, which fails when the score drops, a safety case fails, or the cost or latency budget is
exceeded. `deploy.yml` rolls out in this order: staging (a full copy with test data and an in-cluster eval),
then **shadow** (the next release in `aiready-next`, receiving mirrored real requests whose replies are
discarded, and with no route to the existing app), then **canary** (ingress-nginx sends `canaryWeight`% of
requests to it), then **production**. Each stage after staging needs manual approval. Rollback is one change:
delete the next release, or run `helm rollback`.

## Topologies

* **Cloud** (`deploy/terraform`): one region, three zones, a private network with no public addresses, and
  a managed control plane. CPU pool, GPU pool A (LLM), GPU pool B (embedding/rerank), and a batch pool on
  spare capacity. Managed key management, secrets, registry and logs. The optional private model endpoint is
  opened by policy.
* **On-prem** (`deploy/onprem`): no route to the internet. Updates arrive as signed bundles on an agreed
  schedule and are imported one way: scan, verify, then push to the internal registry and model store. A
  standby cluster at a second site is restored from snapshots, and backups are kept off-site.

## Operations

See `docs/runbooks/`: incident response, model rollout and rollback, adding a GPU node, restoring the vector
index, rotating signing keys, replaying dead letters, and audit chain verification.
