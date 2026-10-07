# On-prem deployment: air-gapped release bundles

This directory implements the **"Deployment topology: on-prem"** slide: a client data centre with **no
route to the internet**. Everything a release needs crosses into the site once, as a signed bundle,
through a one-way import. Nothing on site ever pulls from a public registry, model hub or package index.

```
 Connected build host            Transfer               Air-gapped site
 ───────────────────            ────────               ───────────────────────────────────────────────
 build-bundle.sh   ──► bundle.tar + .sha256 ──► scan & verify station ──► internal registry (images)
  images, chart,        (removable media or        verify-bundle.sh       model store (weights)
  config, weights,       a one-way data diode)     + site AV/CVE scan     internal chart repo
  SBOMs, SHA256SUMS,                                     │
  cosign signatures                               import-bundle.sh ──► helm upgrade (internal ingress only)
```

The CLI wraps the same scripts: `aiready bundle build`, `aiready bundle verify`, `aiready bundle import`.

## Files

| File | Purpose |
|---|---|
| `build-bundle.sh` | Connected side. Packs images (`skopeo` OCI archives, else `docker save`), the Helm chart, `config/` + `deploy/config-overlays/`, model weights with a per-file sha256 manifest, SBOMs (`syft`, SPDX JSON per image), `bundle.json`, `SHA256SUMS`, cosign signatures. |
| `verify-bundle.sh` | Air-gapped side. Checks the cosign signature of `SHA256SUMS` and `bundle.json` (refuses unsigned bundles unless `--allow-unsigned`), every file checksum, that no unlisted file was slipped in, and the hash of every weight file inside every model archive. Any mismatch: non-zero exit, nothing imported. |
| `import-bundle.sh` | Extracts into a private directory, runs `verify-bundle.sh` on exactly those files, then loads images into `INTERNAL_REGISTRY` (optionally re-signing them with the site key for admission control), unpacks weights into `MODEL_STORE` (path or `s3://` with `S3_ENDPOINT`, e.g. MinIO) as `<name>/<revision>/` with `manifest.json` and `SHA256SUMS` beside the weights, copies the chart to the internal chart repo, prints the `helm upgrade` command and appends a line to the import log (who, when, version, bundle sha256, outcome). |
| `images.txt` | Third-party images shipped in every bundle. The platform image (`${REGISTRY:-ghcr.io/aiready}/aiready:${VERSION}`) is added automatically. |
| `models.example.yaml` | Input list of approved model weights (name, source dir, revision, licence). |
| `lib/` | Shared shell helpers and `models.py` (pack / verify / unpack weights; standard library + PyYAML on the build host). |
| `backup/` | Backup and restore scripts for the state database, vector index and configuration (see below). |

### Example

```bash
# connected build host
COSIGN_KEY=hashivault://aiready-release ./build-bundle.sh --version 1.4.0 --models models.yaml --out dist/

# air-gapped side (use the site's already-trusted copy of the scripts, not the one inside the bundle)
./verify-bundle.sh --bundle /media/aiready-bundle-1.4.0.tar --key /etc/aiready/cosign.pub
INTERNAL_REGISTRY=registry.dc.internal/aiready MODEL_STORE=s3://models S3_ENDPOINT=https://minio.dc.internal \
  ./import-bundle.sh --bundle /media/aiready-bundle-1.4.0.tar --sign-images /etc/aiready/site-cosign.key
helm upgrade --install aiready /var/lib/aiready/charts/aiready-<chart-version>.tgz -n aiready \
  -f values-production.yaml -f values-site.yaml \
  --set global.imageRegistry=registry.dc.internal/aiready --set image.tag=1.4.0
```

## Updates: signed bundles on an agreed schedule

* Updates arrive **only** as signed bundles, on a schedule agreed with the client (for example monthly,
  plus out-of-band security fixes): `[to be agreed]`.
* Each bundle is one version of all three registries from the release pipeline slide: images, model
  weights, and chart/config. It has already passed the release gate (`aiready eval gate`) and CI's image
  scan before it is built. The site scans it again on the verify station with its own tooling.
* Rollback is one change: the previous chart and images remain in the internal registry and chart repo,
  so `helm rollback aiready <revision>` (or re-applying the previous version's values) needs no new
  import. Model weights are stored per revision, so the previous model stays in the store.
* The import log (`/var/log/aiready/bundle-imports.log`, protect with `chattr +a`) is forwarded to the
  client's security monitoring alongside the platform's audit log.

## Key management

| Key | Where it lives | Used for |
|---|---|---|
| Release signing key (cosign) | HSM or Vault transit on the vendor/build side (`COSIGN_KEY=hashivault://...` or `pkcs11:...`). Never on disk in CI. | `cosign sign-blob` over `SHA256SUMS` and `bundle.json` |
| Release public key | Handed to the client **out of band** (in person, or published fingerprint confirmed by phone) and installed at `/etc/aiready/cosign.pub`. Never taken from inside a bundle. | `verify-bundle.sh` |
| Site image-signing key | Client Vault with HSM (on-prem topology slide: "Vault with HSM"). | `import-bundle.sh --sign-images`: re-signs imported images by digest so Gatekeeper/Kyverno (`deploy/security/`) admit only images from the internal registry signed by the site. |
| Backup encryption key | age or GPG keypair; public key in the backup CronJob config, private key in Vault, fetched only for a restore. | `backup/*.sh` |

Signatures are made without uploading to a public transparency log (`COSIGN_TLOG=0`, the default), and
verified with `--insecure-ignore-tlog` accordingly: an air-gapped site cannot reach a public Rekor, and
release metadata should not be published there. Set `COSIGN_TLOG=1` if the client runs a private Rekor.

Rotating the release key: generate a new key in the HSM, deliver the new public key out of band, ship one
bundle signed with the old key that announces the change, then switch. See `docs/runbooks/` for signing
key rotation of the platform's own token and audit keys.

## Mapping to the on-prem topology slide

| Slide element | Implementation |
|---|---|
| One-way import: release bundle, scan and verify, internal registry | `build-bundle.sh` → transfer → `verify-bundle.sh` → `import-bundle.sh` |
| Kubernetes, **three control-plane nodes** | Client-provided cluster (kubeadm, RKE2, OpenShift...) with 3 control-plane nodes; the chart is distribution-neutral. |
| Ingress: **internal only** | Chart ingress with an internal class and no public address (`values-production.yaml`). |
| CPU workers: platform services | Gateway (3+ replicas), guardrails, governance, search, RAG, agents, MCP, etc. on the CPU node pool. |
| GPU workers A: LLM serving | Chat model pool (vLLM) with `nvidia.com/gpu`, node selector/taints for pool A. |
| GPU workers B: small models | Router + guard model pool and the embedding + rerank pool. |
| GPU scheduler: queues and sharing | NVIDIA GPU operator plus a queueing scheduler (Kueue/Volcano) for batch and fine-tuning jobs; KEDA scales on queue depth. |
| Batch workers: ingestion, evals | Ingestion workers, eval CronJobs, fine-tuning Jobs on the batch node pool. |
| Monitoring: metrics and traces | `deploy/observability/` (OTel collector with PII masking, Prometheus, Tempo, Loki, Grafana). |
| Secrets and keys: Vault with HSM | Chart's Vault agent injector / External Secrets toggle; keys held by the client. |
| Storage: database cluster primary + standby | PostgreSQL (with pgvector) primary and streaming-replica standby, e.g. CloudNativePG or Patroni. |
| Vector database: three replicas | Qdrant distributed mode, replication factor 3 for collections. |
| Object storage: weights, snapshots | MinIO (S3 API) or NFS; `MODEL_STORE` and `BACKUP_TARGET`. |
| Second site: standby cluster restored from snapshots | Procedure below. |
| Off-site backups: database, index, config | `backup/` scripts targeting the off-site location. |

## Backups (`backup/`)

All scripts are self-contained (they share `backup/_common.sh`), installed in the platform image under
`/opt/aiready/backup`, and run by the chart's backup CronJobs. Targets are a directory (an NFS mount) or
`s3://bucket/prefix` (`S3_ENDPOINT` for MinIO). Every artefact gets a `.sha256` sidecar; set
`BACKUP_AGE_RECIPIENT` (or `BACKUP_GPG_RECIPIENT`) to encrypt in a stream so no plaintext lands on disk.

| Script | What | Default schedule | Default retention |
|---|---|---|---|
| `backup-postgres.sh` | `pg_dump -Fc` of the state DB (registries, run state, budgets, audit chain); libpq `PG*` env | nightly 01:00 | 14 days local, `[to be agreed]` off-site |
| `backup-qdrant.sh` | Snapshot API: snapshot every collection, download, delete server copy (run per peer with `--label` in distributed mode) | daily 02:00 | 14 days |
| `backup-config.sh` | Config dir + overlay + Helm values (`--helm-release` captures effective values); never secrets | daily, and before every import | 30 days |
| `restore-postgres.sh` | `pg_restore --clean --if-exists`, checksum + decrypt first; needs `--yes` | on demand | |
| `restore-qdrant.sh` | Upload (`POST .../snapshots/upload?priority=snapshot`) or recover from location (`PUT .../snapshots/recover`); needs `--yes` | on demand | |

Off-site: the same CronJobs write a second copy to the off-site target (second site object store or
tape gateway). Restore tests run `[quarterly, to be agreed]`; a backup that has never been restored is not
counted as a backup.

**Recovery objectives:** RPO `[to be agreed]`, RTO `[to be agreed]`. The defaults above give an RPO of up to
24 hours for the vector index (rebuildable from sources by re-ingestion) and near-zero for the database
when streaming replication to the second site is enabled.

## Standby cluster at the second site

The second site runs a standby Kubernetes cluster with the same bundle version imported (every bundle is
imported at both sites). It is restored from snapshots, not kept hot, except for the database replica.

1. **Database.** Preferred: a streaming replica of the primary at the second site (`pg_basebackup` to seed
   it, then WAL streaming; CloudNativePG `replica` cluster or Patroni standby cluster). Promote it on
   failover. Fallback: `restore-postgres.sh --source <latest off-site dump> --yes`.
2. **Audit chain.** Run `aiready audit verify` against the restored database before opening traffic.
3. **Vector index.** For each collection: `restore-qdrant.sh --collection <c> --source <latest snapshot> --yes`.
   Then replay ingestion from the snapshot time (`aiready ingest` from the change log / sources) so the
   index catches up with the source systems.
4. **Configuration.** Unpack the latest `aiready-config-*.tar.gz`, compare with git, and apply it as the
   config ConfigMap/overlay; secrets come from the second site's Vault (keys replicated per the client's
   Vault DR setup, never from backups).
5. **Models.** Already present: weights are imported at both sites with every bundle. The init container
   re-verifies `SHA256SUMS` on each GPU node before a model server starts.
6. **Cutover.** `helm upgrade --install` with the site values, wait for `/readyz` on gateway and model
   pools, run the smoke eval (`aiready eval run --suite smoke`), then switch the internal DNS record (or
   the global load balancer) for the gateway and assistant hostnames to the second site.
7. **Fail back** by reversing replication once the primary site is healthy; keep the incident record.
