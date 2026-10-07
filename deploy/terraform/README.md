# Cloud topology: Terraform

Provisions the "Deployment topology: cloud" slide in the client's own cloud account: one region, three
availability zones, a private network with no public addresses, a Kubernetes cluster with a managed control
plane and four node pools, and the managed state services, all encrypted with keys the client holds.

AWS is the primary implementation (`aws/`). Azure and GCP equivalents are listed below; the Helm chart in
`deploy/helm/aiready` is cloud-neutral and only needs the node labels/taints and secret names to match.

## Layout

```
aws/
  versions.tf            providers (aws ~> 5.80, random, tls) and a commented S3 backend example
  main.tf                wires the modules together
  variables.tf           every input, commented
  outputs.tf             endpoints, ARNs, IRSA roles and a helm_values_hint map
  terraform.tfvars.example  stage 2 baseline plus stage 1 / stage 3 notes
  modules/
    kms/                 three CMKs with rotation: eks (Secrets), data (RDS/S3/Redis/Secrets/ECR), logs
    observability/       CloudWatch log groups (control plane, application, audit, flow logs) with KMS + retention
    network/             VPC, 3 private /20 subnets, VPC endpoints, flow logs, optional egress tier with NAT
    eks/                 private-only EKS, KMS envelope encryption, control-plane logs, IRSA, node groups, add-ons
    data/                RDS PostgreSQL 16, ElastiCache Redis (TLS + AUTH), S3 documents / weights / backups
    secrets/             Secrets Manager entries named after /run/secrets/<name> files
    ecr/                 immutable, scan-on-push, KMS-encrypted repositories
    irsa/                least-privilege roles for External Secrets, data readers, backups, optional Bedrock
```

## Usage

```bash
cd deploy/terraform/aws
cp terraform.tfvars.example terraform.tfvars      # edit
terraform init                                    # configure the backend block in versions.tf first
terraform plan -out plan.tfplan
terraform apply plan.tfplan
terraform output -json helm_values_hint           # feed into deploy/helm/aiready/values-production.yaml
```

The API endpoint is private only: run Terraform's Kubernetes steps and `helm` from inside the VPC (VPN,
bastion via SSM, or a self-hosted CI runner in the private subnets).

## Zone model on AWS

| Zone | Where it lands |
|---|---|
| Application zone | The existing app's VPC, peered / via transit gateway; reaches only the internal load balancer (`app_network_cidrs`). |
| AI platform zone | `cpu` node group (label `aiready.io/pool=cpu`). |
| Model zone | `gpu_a` (`aiready.io/pool=gpu-a`) and `gpu_b` (`aiready.io/pool=gpu-b`), tainted `nvidia.com/gpu=true:NoSchedule`. No outbound route; Kubernetes NetworkPolicies (VPC CNI policy enforcement is enabled) block egress. |
| Data zone | RDS, ElastiCache, S3, all KMS-encrypted, reachable only from the cluster security group. |
| Outside model API | Closed by default. `enable_egress_nat = true` creates egress subnets + NAT used only by the `egress` node group (taint `aiready.io/egress=true:NoSchedule`) where the egress proxy runs. `enable_private_model_endpoint = true` adds a Bedrock VPC endpoint instead of any internet route. |

Batch work (ingestion workers, evaluation runs, fine-tuning) runs on the `batch` spot node group (label
`aiready.io/pool=batch`, taint `aiready.io/batch=true:NoSchedule`).

## Stage mapping

| Stage | What to set |
|---|---|
| 1 Pilot | Usually `deploy/compose` on one host. If the pilot uses a private endpoint instead of a GPU: `stage = "pilot"`, CPU group only, `enable_private_model_endpoint = true`, single-AZ DB, one Redis node. |
| 2 Production | Defaults: 3+ CPU nodes for a redundant gateway, GPU A for the chat tier (second model tier), GPU B for embedding/rerank/small models, multi-AZ DB and Redis. |
| 3 Platform | `stage = "platform"`, `gpu_a.min_size = 0` so idle GPU capacity scales to zero (KEDA in the chart scales pods to zero, the cluster autoscaler / Karpenter removes nodes), larger batch pool for adapter training. |

Node counts are never quoted before a load test: compute them with `aiready sizing` (see `docs/deployment.md`).

## Secrets

`modules/secrets` creates `<name>/token_signing_key`, `<name>/audit_hmac_key`, `<name>/db_password`,
`<name>/redis_auth_token`, `<name>/service_token_key`, `<name>/qdrant_api_key` (generated) and `<name>/llm_api_key` (empty; filled out of band only if a public or
private-endpoint route is approved). External Secrets Operator (role output `irsa_role_arns.external_secrets`)
syncs them into a Kubernetes Secret that the chart mounts at `/run/secrets/<name>` (chart `secrets.mode=externalSecrets`, `remoteKeyPrefix: <name>/`; keys must match `secrets.keys`). Rotation writes new
versions outside Terraform (`ignore_changes` on the value), see `docs/runbooks/rotate-signing-keys.md`.

## Notes and assumptions

* pgvector: RDS PostgreSQL ships the extension; the platform's migration runs `CREATE EXTENSION vector`. No
  parameter group change is needed. If the chosen vector store is Qdrant, it runs in-cluster with EBS volumes.
* Root volumes use EBS default encryption (AWS-managed key) to avoid granting the autoscaling service-linked
  role on the data key; switch to the data CMK if the client requires it.
* IRSA service-account names (`ingestion`, `ingestion-worker`, `model-loader`, `backup`, `gateway`) must match
  the chart's ServiceAccounts; change `data_reader_service_accounts` / `backup_service_accounts` if renamed.
* GPU nodes use the `AL2023_x86_64_NVIDIA` AMI; install the NVIDIA device plugin (or GPU operator) so
  `nvidia.com/gpu` resources are advertised.
* Cluster autoscaler discovery tags are set on node groups; Karpenter is a drop-in alternative for stage 3.

## Azure and GCP equivalents

| AWS (this module) | Azure | GCP |
|---|---|---|
| VPC, private subnets | VNet, private subnets | VPC, private subnets (Private Google Access) |
| EKS (private endpoint) | AKS private cluster | GKE private cluster |
| Managed node groups (GPU A/B, batch) | AKS node pools (NC/ND series) | GKE node pools (A2/G2/A3) |
| KMS CMKs | Key Vault keys (Managed HSM) | Cloud KMS (Cloud HSM) |
| Secrets Manager | Key Vault secrets | Secret Manager |
| ECR | Azure Container Registry | Artifact Registry |
| RDS PostgreSQL | Azure Database for PostgreSQL Flexible Server | Cloud SQL for PostgreSQL / AlloyDB |
| ElastiCache Redis | Azure Cache for Redis | Memorystore for Redis |
| S3 | Blob Storage | Cloud Storage (GCS) |
| VPC endpoints | Private Link / private endpoints | Private Service Connect |
| Bedrock VPC endpoint (private model endpoint) | Azure OpenAI private endpoint | Vertex AI via Private Service Connect |
| CloudWatch Logs | Azure Monitor / Log Analytics | Cloud Logging |
| Spot capacity | Spot VMs (spot node pools) | Spot VMs |
| NAT gateway (egress tier) | NAT Gateway / Azure Firewall | Cloud NAT |
| IRSA | Workload Identity (Entra ID) | Workload Identity Federation |
