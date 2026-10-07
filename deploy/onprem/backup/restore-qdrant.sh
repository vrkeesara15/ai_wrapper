#!/usr/bin/env bash
# Restore one Qdrant collection from a snapshot made by backup-qdrant.sh.
#
# Two modes:
#   upload    (--source FILE|s3://...) checksum, decrypt, then POST the snapshot to
#             /collections/{c}/snapshots/upload?priority=snapshot (multipart upload).
#   location  (--location URL) PUT /collections/{c}/snapshots/recover so Qdrant fetches the snapshot
#             itself from an internal URL or file:// path (large snapshots, no client upload).
# priority=snapshot makes the snapshot's data win over whatever the collection currently holds.
# See docs/runbooks for the full "restore vector index from snapshot" procedure.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/_common.sh
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --collection NAME (--source FILE|s3://... | --location URL) --yes [--url URL] [--checksum SHA256]

  --collection   Collection to restore (required)
  --source       Snapshot file from backup-qdrant.sh (.snapshot[.age|.gpg]), local or s3://
  --location     URL Qdrant can read itself (http(s):// or file:///...)
  --checksum     Expected sha256 for --location mode (Qdrant verifies it)
  --url          Qdrant HTTP endpoint (default: \$QDRANT_URL or http://qdrant:6333)
  --yes          Confirm that the collection's current contents will be replaced
Env: QDRANT_API_KEY, AGE_IDENTITY_FILE for .age, S3_ENDPOINT
EOF
}

COLLECTION="" SOURCE="" LOCATION="" CHECKSUM="" URL="${QDRANT_URL:-http://qdrant:6333}" YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection) COLLECTION="${2:?}"; shift 2 ;;
    --source) SOURCE="${2:?}"; shift 2 ;;
    --location) LOCATION="${2:?}"; shift 2 ;;
    --checksum) CHECKSUM="${2:?}"; shift 2 ;;
    --url) URL="${2:?}"; shift 2 ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
[[ -n "$COLLECTION" ]] || { usage >&2; die "--collection is required"; }
[[ "$COLLECTION" =~ ^[A-Za-z0-9._-]+$ ]] || die "unexpected collection name: $COLLECTION"
[[ -n "$SOURCE" || -n "$LOCATION" ]] || { usage >&2; die "--source or --location is required"; }
[[ "$YES" == 1 ]] || die "refusing to replace collection $COLLECTION without --yes"
have curl || die "curl not installed"
URL="${URL%/}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/aiready-qdrantr.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
HEADERS="$WORK/headers"
: > "$HEADERS"
chmod 600 "$HEADERS"
if [[ -n "${QDRANT_API_KEY:-}" ]]; then printf 'api-key: %s\n' "$QDRANT_API_KEY" > "$HEADERS"; fi

if [[ -n "$LOCATION" ]]; then
  body="$(python3 -c 'import json,sys; d={"location": sys.argv[1], "priority": "snapshot"}
if sys.argv[2]: d["checksum"] = sys.argv[2]
print(json.dumps(d))' "$LOCATION" "$CHECKSUM")"
  log "recovering $COLLECTION from $LOCATION"
  curl -fsS -H @"$HEADERS" -H 'Content-Type: application/json' -X PUT \
    "$URL/collections/$COLLECTION/snapshots/recover?wait=true" -d "$body" >/dev/null
else
  FILE="$(fetch "$SOURCE" "$WORK")"
  [[ -f "$FILE" ]] || die "snapshot not found: $SOURCE"
  check_sidecar "$FILE"
  case "$FILE" in
    *.age|*.gpg) decrypt_to_stdout "$FILE" > "$WORK/restore.snapshot"; PLAIN="$WORK/restore.snapshot" ;;
    *) PLAIN="$FILE" ;;
  esac
  log "uploading snapshot into $COLLECTION"
  curl -fsS -H @"$HEADERS" -X POST -F "snapshot=@$PLAIN" \
    "$URL/collections/$COLLECTION/snapshots/upload?priority=snapshot&wait=true" >/dev/null
fi
count="$(curl -fsS -H @"$HEADERS" "$URL/collections/$COLLECTION" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"].get("points_count"))')"
log "collection $COLLECTION restored: points_count=$count"
log "Next: compare points_count with the ingestion lineage, then replay ingestion from the snapshot time (aiready ingest)."
