# Dead-letter queue replay

**Owner:** data engineer. **Alert:** `AIReadyDeadLetters` (any message in a `*:dlq` queue).

A message is dead-lettered after `max_attempts` retries (`ingestion.queue.max_attempts`). Permanent errors,
such as an unparseable file or a refused access-tag mapping, go there at once. Replaying without fixing the
cause only fills the DLQ again.

## Steps

1. List the dead letters and their last errors:
   ```bash
   aiready ingest dlq list [--topic ingestion.documents]      # or GET /v1/dlq on the ingestion service
   ```
2. Classify:
   * **Transient** (timeouts, embedding pool saturated, source temporarily down): replay once the dependency
     is healthy.
   * **Data** (a corrupt or unsupported file): fix it at the source or exclude it in the source config. Do not
     replay.
   * **Access** (no access tags, or an unknown group mapping): fix the mapping in `config/ingestion.yaml`.
     Documents without access tags are never indexed, by design.
   * **Bug:** open a ticket, add the document to the ingestion tests, and release a fix first.
3. Replay:
   ```bash
   aiready ingest dlq replay [--topic ingestion.documents] [--limit N]   # or POST /v1/dlq/replay
   ```
   Replayed messages go back on the work queue. KEDA scales the workers on the queue length.
4. Verify: the `aiready_queue_depth{queue=~".*dlq"}` gauge returns to 0, and
   `aiready_pipeline_items_total{outcome="loaded"}` rises. `aiready ingest runs --last` shows the run.

Record: the message count by class, the causes fixed, and the replay result.
