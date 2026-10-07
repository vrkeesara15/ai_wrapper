#!/usr/bin/env bash
# Back up the state database (registries, run state, budgets, audit chain) with pg_dump custom format.
#
# Connection comes from the standard libpq variables (PGHOST, PGPORT, PGUSER, PGDATABASE, and
# PGPASSWORD or PGPASSFILE mounted from the secret store), so no credential appears on the command
# line. The dump is streamed through optional encryption, checksummed, copied to the target and old
# copies are pruned. Scheduled by the chart's backup CronJob (nightly by default); the off-site copy
# is the same script pointed at the second site's target.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/_common.sh
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--target DIR|s3://bucket/prefix] [--retention-days N] [--database NAME]

  --target           Backup destination (default: \$BACKUP_TARGET, else /backups/postgres)
  --retention-days   Delete backups older than N days at the target, 0 keeps all (default: \$BACKUP_RETENTION_DAYS or 14)
  --database         Database to dump (default: \$PGDATABASE or aiready)
Env: PGHOST PGPORT PGUSER PGPASSWORD/PGPASSFILE, BACKUP_AGE_RECIPIENT or BACKUP_GPG_RECIPIENT, S3_ENDPOINT
EOF
}

TARGET="${BACKUP_TARGET:-/backups/postgres}" RETENTION="${BACKUP_RETENTION_DAYS:-14}" DB="${PGDATABASE:-aiready}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?}"; shift 2 ;;
    --retention-days) RETENTION="${2:?}"; shift 2 ;;
    --database) DB="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
have pg_dump || die "pg_dump not installed"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/aiready-pg.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
PREFIX="aiready-pg-${DB}-"
FILE="$WORK/${PREFIX}$(timestamp).dump$(encrypt_suffix)"

log "dumping database $DB from ${PGHOST:-localhost}"
pg_dump --format=custom --compress=6 --no-password --dbname "$DB" | encrypt_stream > "$FILE"
[[ -s "$FILE" ]] || die "dump is empty"
write_checksum "$FILE"
upload "$FILE" "$TARGET"
prune "$TARGET" "$PREFIX" "$RETENTION"
log "postgres backup complete: $(basename "$FILE")"
