# Restoring the vector index from a snapshot

**Owner:** platform engineer, with the data engineer. The vector index is derived data: it can always be
rebuilt from sources by re-ingestion (lineage links every chunk to its source record). A snapshot restore
is faster, and it restores the exact state at snapshot time.

## Decide: restore or re-ingest

* **Restore** when a collection is lost or corrupt, or when activating the standby site. RPO = time since the
  last daily snapshot (`backup-qdrant` CronJob).
* **Re-ingest** when the snapshot is older than the source changes you care about, or the embedding model
  changed. Use `aiready ingest run --source <name> --full`.
* Usually you do both: restore, then run an incremental ingestion to catch up.

## Steps

1. Stop writers so that the restore is not overwritten:
   ```bash
   kubectl -n aiready scale deploy/ingestion-worker --replicas=0     # KEDA: pause first
   kubectl -n aiready annotate scaledobject ingestion-worker autoscaling.keda.sh/paused-replicas="0" --overwrite
   ```
2. Find the snapshot on the backup target (PVC / off-site copy): `qdrant/<collection>-<timestamp>.snapshot`,
   with its `.sha256` sidecar. Decrypt it first if it is encrypted (age/gpg key from Vault).
3. Restore each collection from a pod in the platform zone (the backup tools are in the image):
   ```bash
   kubectl -n aiready run qdrant-restore --rm -it --restart=Never --image=<platform image> \
     --overrides='{"spec":{"automountServiceAccountToken":false}}' -- \
     /opt/aiready/backup/restore-qdrant.sh --collection <collection> --source /backups/qdrant/<file>.snapshot --yes
   ```
   (Mount the `aiready-backups` PVC, or use `--location <url>` for a snapshot the Qdrant peers can reach.) In
   a 3-peer cluster, restore on each peer that holds shards, or restore on one peer and let replication
   catch up (replication factor > 1).
4. Check the counts against the document registry:
   `aiready ingest verify --collection <collection>` compares chunk counts and samples lineage.
5. Resume the writers: remove the paused annotation, then `aiready ingest run --all-sources --incremental`.
6. Run a smoke evaluation: `aiready eval run claims --release restore-<date>`. Retrieval scores should match
   the last scorecard.

Record: the snapshot timestamp, the collections restored, the counts before and after, and the catch-up
ingestion run id.
