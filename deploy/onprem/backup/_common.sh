# shellcheck shell=bash
# Shared helpers for the backup / restore scripts. Sourced, never executed.
#
# The scripts are installed together in the platform image under /opt/aiready/backup and run from the
# Helm chart's backup CronJobs (or by hand during a restore). Targets are either a directory (an NFS
# mount for the off-site copy) or s3://bucket/prefix (MinIO / S3; S3_ENDPOINT selects the endpoint).
#
# Encryption: set BACKUP_AGE_RECIPIENT (age public key) or BACKUP_GPG_RECIPIENT to encrypt in a
# stream, so no plaintext dump touches disk. The private key stays in Vault/HSM and is only fetched
# for a restore.

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

timestamp() { date -u +%Y%m%dT%H%M%SZ; }

sha256_file() {
  if have sha256sum; then sha256sum "$1" | awk '{print $1}'
  elif have shasum; then shasum -a 256 "$1" | awk '{print $1}'
  else die "neither sha256sum nor shasum is installed"; fi
}

# Write "<hash>  <basename>" next to a file.
write_checksum() { ( cd "$(dirname "$1")" && printf '%s  %s\n' "$(sha256_file "$(basename "$1")")" "$(basename "$1")" > "$(basename "$1").sha256" ); }

aws_s3() {
  have aws || die "aws CLI required for s3:// targets"
  if [[ -n "${S3_ENDPOINT:-}" ]]; then aws --endpoint-url "$S3_ENDPOINT" s3 "$@"; else aws s3 "$@"; fi
}

# Encrypt stdin to stdout when a recipient is configured; prints the file suffix with --suffix.
encrypt_suffix() {
  if [[ -n "${BACKUP_AGE_RECIPIENT:-}" ]]; then echo ".age"
  elif [[ -n "${BACKUP_GPG_RECIPIENT:-}" ]]; then echo ".gpg"
  else echo ""; fi
}
encrypt_stream() {
  if [[ -n "${BACKUP_AGE_RECIPIENT:-}" ]]; then
    have age || die "age not installed"; age -r "$BACKUP_AGE_RECIPIENT"
  elif [[ -n "${BACKUP_GPG_RECIPIENT:-}" ]]; then
    have gpg || die "gpg not installed"; gpg --batch --yes --trust-model always --encrypt --recipient "$BACKUP_GPG_RECIPIENT"
  else
    cat
  fi
}
# Decrypt a file to stdout based on its suffix. AGE_IDENTITY_FILE holds the age private key.
decrypt_to_stdout() {
  case "$1" in
    *.age) have age || die "age not installed"; [[ -n "${AGE_IDENTITY_FILE:-}" ]] || die "AGE_IDENTITY_FILE not set"
           age -d -i "$AGE_IDENTITY_FILE" "$1" ;;
    *.gpg) have gpg || die "gpg not installed"; gpg --batch --decrypt "$1" ;;
    *) cat "$1" ;;
  esac
}

# Copy a local file (and its .sha256) to the target.
upload() {
  local file="$1" target="$2"
  if [[ "$target" == s3://* ]]; then
    aws_s3 cp --only-show-errors "$file" "${target%/}/"
    aws_s3 cp --only-show-errors "$file.sha256" "${target%/}/"
  else
    mkdir -p "$target"
    cp "$file" "$file.sha256" "$target/"
  fi
  log "stored $(basename "$file") in $target"
}

# Fetch a backup (local path or s3:// URL) into a directory; prints the local path.
fetch() {
  local src="$1" dir="$2"
  if [[ "$src" == s3://* ]]; then
    aws_s3 cp --only-show-errors "$src" "$dir/"
    aws_s3 cp --only-show-errors "$src.sha256" "$dir/" 2>/dev/null || true
    printf '%s' "$dir/$(basename "$src")"
  else
    printf '%s' "$src"
  fi
}

# Check a file against its .sha256 sidecar when one exists.
check_sidecar() {
  local file="$1"
  if [[ -f "$file.sha256" ]]; then
    [[ "$(sha256_file "$file")" == "$(awk '{print $1}' "$file.sha256")" ]] || die "checksum mismatch for $file"
    log "checksum ok for $(basename "$file")"
  else
    log "no checksum sidecar for $(basename "$file"); continuing"
  fi
}

# Date N days ago as YYYY-MM-DD (GNU date or BSD date).
days_ago() {
  date -u -d "$1 days ago" +%Y-%m-%d 2>/dev/null || date -u -v "-$1d" +%Y-%m-%d
}

# Delete backups older than N days whose names start with PREFIX.
prune() {
  local target="$1" prefix="$2" days="$3" cutoff
  [[ "$days" -gt 0 ]] || return 0
  if [[ "$target" == s3://* ]]; then
    cutoff="$(days_ago "$days")"
    aws_s3 ls "${target%/}/" | while read -r d _t _size name; do
      [[ "$name" == "$prefix"* && "$d" < "$cutoff" ]] || continue
      aws_s3 rm --only-show-errors "${target%/}/$name"
      log "pruned $name"
    done
  else
    [[ -d "$target" ]] || return 0
    find "$target" -maxdepth 1 -type f -name "${prefix}*" -mtime "+$days" -print -delete | while read -r f; do
      log "pruned $(basename "$f")"
    done
  fi
}
