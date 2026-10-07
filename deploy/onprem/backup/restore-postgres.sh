#!/usr/bin/env bash
# Restore the state database from a backup made by backup-postgres.sh.
#
# Used for the standby cluster at the second site and for recovery. The checksum is verified and the
# dump decrypted in a stream before pg_restore runs. --clean --if-exists drops objects first, so the
# target ends up exactly like the backup; --yes is required because this overwrites data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/_common.sh
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --source FILE|s3://... --yes [--database NAME] [--jobs N]

  --source     Backup file (.dump, .dump.age, .dump.gpg), local or s3:// (required)
  --database   Target database (default: \$PGDATABASE or aiready)
  --jobs       Parallel restore jobs (default 4; ignored for encrypted input)
  --yes        Confirm that the target database will be overwritten
Env: PGHOST PGPORT PGUSER PGPASSWORD/PGPASSFILE, AGE_IDENTITY_FILE for .age, S3_ENDPOINT
EOF
}

SOURCE="" DB="${PGDATABASE:-aiready}" JOBS=4 YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="${2:?}"; shift 2 ;;
    --database) DB="${2:?}"; shift 2 ;;
    --jobs) JOBS="${2:?}"; shift 2 ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
[[ -n "$SOURCE" ]] || { usage >&2; die "--source is required"; }
[[ "$YES" == 1 ]] || die "refusing to overwrite database $DB without --yes"
have pg_restore || die "pg_restore not installed"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/aiready-pgr.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
FILE="$(fetch "$SOURCE" "$WORK")"
[[ -f "$FILE" ]] || die "backup not found: $SOURCE"
check_sidecar "$FILE"

log "restoring $(basename "$FILE") into $DB on ${PGHOST:-localhost}"
case "$FILE" in
  *.age|*.gpg)
    # Encrypted dumps restore from a stream (pg_restore cannot run parallel jobs on a pipe).
    decrypt_to_stdout "$FILE" | pg_restore --clean --if-exists --no-owner --no-password --exit-on-error --dbname "$DB"
    ;;
  *)
    pg_restore --clean --if-exists --no-owner --no-password --exit-on-error --jobs "$JOBS" --dbname "$DB" "$FILE"
    ;;
esac
log "restore complete. Next: run 'aiready audit verify' to confirm the audit chain is intact."
