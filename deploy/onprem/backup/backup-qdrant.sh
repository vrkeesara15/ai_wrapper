#!/usr/bin/env bash
# Snapshot every Qdrant collection (the vector index) and store the snapshots at the backup target.
#
# Uses the Qdrant snapshot API: GET /collections, POST /collections/{c}/snapshots?wait=true, download,
# then DELETE the server-side copy so snapshots do not fill the vector database's volume. Snapshots
# hold document chunks, so they are encrypted in a stream when a recipient is configured.
# Distributed Qdrant keeps shards per peer: run once per peer (--url http://qdrant-N.qdrant-headless:6333)
# with --label so file names stay distinct. Scheduled by the chart's daily backup CronJob ("daily
# snapshots" on the cloud topology slide).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/_common.sh
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--url URL] [--target DIR|s3://...] [--collections a,b] [--retention-days N] [--label NAME]

  --url              Qdrant HTTP endpoint (default: \$QDRANT_URL or http://qdrant:6333)
  --target           Backup destination (default: \$BACKUP_TARGET or /backups/qdrant)
  --collections      Comma-separated subset (default: all collections)
  --retention-days   Prune older snapshots at the target, 0 keeps all (default: \$BACKUP_RETENTION_DAYS or 14)
  --label            Extra name component, e.g. the peer name in a distributed cluster
  --keep-remote      Do not delete the snapshot from the Qdrant server after download
Env: QDRANT_API_KEY (sent as the api-key header), BACKUP_AGE_RECIPIENT / BACKUP_GPG_RECIPIENT, S3_ENDPOINT
EOF
}

URL="${QDRANT_URL:-http://qdrant:6333}" TARGET="${BACKUP_TARGET:-/backups/qdrant}" COLLECTIONS=""
RETENTION="${BACKUP_RETENTION_DAYS:-14}" LABEL="" KEEP_REMOTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="${2:?}"; shift 2 ;;
    --target) TARGET="${2:?}"; shift 2 ;;
    --collections) COLLECTIONS="${2:?}"; shift 2 ;;
    --retention-days) RETENTION="${2:?}"; shift 2 ;;
    --label) LABEL="${2:?}"; shift 2 ;;
    --keep-remote) KEEP_REMOTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
have curl || die "curl not installed"
have python3 || die "python3 not installed"
URL="${URL%/}"

# The API key is passed through a header file so it never shows up in the process list.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/aiready-qdrant.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
HEADERS="$WORK/headers"
: > "$HEADERS"
chmod 600 "$HEADERS"
if [[ -n "${QDRANT_API_KEY:-}" ]]; then printf 'api-key: %s\n' "$QDRANT_API_KEY" > "$HEADERS"; fi
q() { curl -fsS -H @"$HEADERS" "$@"; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

if [[ -z "$COLLECTIONS" ]]; then
  COLLECTIONS="$(q "$URL/collections" | json "','.join(c['name'] for c in d['result']['collections'])")"
fi
[[ -n "$COLLECTIONS" ]] || { log "no collections to back up"; exit 0; }

IFS=',' read -r -a LIST <<< "$COLLECTIONS"
TS="$(timestamp)"
for c in "${LIST[@]}"; do
  [[ "$c" =~ ^[A-Za-z0-9._-]+$ ]] || die "unexpected collection name: $c"
  log "snapshotting collection $c"
  name="$(q -X POST "$URL/collections/$c/snapshots?wait=true" | json "d['result']['name']")"
  prefix="aiready-qdrant-${LABEL:+$LABEL-}${c}-"
  file="$WORK/${prefix}${TS}.snapshot$(encrypt_suffix)"
  q "$URL/collections/$c/snapshots/$name" | encrypt_stream > "$file"
  [[ -s "$file" ]] || die "downloaded snapshot for $c is empty"
  write_checksum "$file"
  upload "$file" "$TARGET"
  rm -f "$file" "$file.sha256"
  if [[ "$KEEP_REMOTE" == 0 ]]; then
    q -X DELETE "$URL/collections/$c/snapshots/$name?wait=true" >/dev/null
  fi
  prune "$TARGET" "$prefix" "$RETENTION"
done
log "qdrant backup complete (${#LIST[@]} collection(s))"
