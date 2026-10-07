# Audit chain verification

**Owner:** platform engineer; the result goes to the client's security lead.
**Schedule:** the `audit-verify` CronJob runs hourly. A failure fires `AIReadyAuditVerificationFailed`
(critical).

The audit log is append-only and tamper-evident: each record carries the HMAC (key `audit_hmac_key`, held by
the client) of its content plus the previous record's hash. Verification recomputes the chain. A gap, an
edited record or a reordered record breaks it at a specific sequence number.

## Routine verification

```bash
aiready audit verify --all-chains              # every chain file / sink configured as system of record
aiready audit verify --chain <file> --since 2026-10-01T00:00:00Z
```
In stage 1 there is one chain per service under `/var/lib/aiready/audit/`. In Kubernetes, verify the
system-of-record sink (the durable store or SIEM copy). The stdout copy is for monitoring only.

## When verification fails

1. **Treat it as a security incident** until proven otherwise ([incident-response.md](incident-response.md)).
   Preserve the evidence: copy the chain files or export the range (`aiready audit export --since ... --until ...`)
   to write-once storage before anything else.
2. Find the break: the verifier reports the first bad sequence number, and the expected and found hashes.
3. Tell the causes apart:
   * **Truncation or a lost tail** (pod restart with local storage, a full disk): the chain ends early but is
     consistent. Compare with the SIEM copy, which received each event as it happened.
   * **Key mismatch** (a rotation without the linking record, or a wrong key mounted): every record after a
     point fails. Check `audit.key_rotated` records and the mounted `audit_hmac_key` version.
   * **Modified or removed records:** a single record fails, or the sequence numbers jump. This is
     tampering until proven otherwise. Identify who had write access to the storage in that window.
4. Never "repair" a chain in place. Start a new chain segment with a signed `audit.chain_reset` record that
   references the incident (`aiready audit reset-chain --incident INC-<n>`), and keep the broken segment as
   evidence.
5. Report to the client's security lead with: the range affected, the cause, and the evidence location.

## For auditors and regulators

`aiready audit export --since <t1> --until <t2> --format jsonl --verify` produces the records plus the
verification report, and the record hashes let a third party re-verify the export with the HMAC key held by
the client.
