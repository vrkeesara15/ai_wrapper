#!/usr/bin/env bash
# Back up the platform configuration: the config directory, the active overlay and the Helm values.
#
# Configuration (routes, policies, budgets, prompts, tool schemas) is versioned in git, but the copy
# actually running on site is what a standby cluster must be restored to, so it is archived next to
# the database and index backups. Secrets are never included: they live in Vault and /run/secrets.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/_common.sh
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--config-dir DIR] [--overlay-dir DIR] [--values FILE]... [--helm-release NAME] [--namespace NS] [--target DIR|s3://...] [--retention-days N]

  --config-dir     Base config (default: \$AIREADY_CONFIG_DIR or /etc/aiready/config)
  --overlay-dir    Overlay (default: \$AIREADY_CONFIG_OVERLAY, skipped when unset)
  --values FILE    Helm values file to include (repeatable)
  --helm-release   Also capture 'helm get values --all' for this release (needs helm + cluster access)
  --namespace      Namespace of the release (default: aiready)
  --target         Backup destination (default: \$BACKUP_TARGET or /backups/config)
  --retention-days Prune older archives, 0 keeps all (default: \$BACKUP_RETENTION_DAYS or 30)
EOF
}

CONFIG_DIR="${AIREADY_CONFIG_DIR:-/etc/aiready/config}" OVERLAY_DIR="${AIREADY_CONFIG_OVERLAY:-}"
VALUES=() RELEASE="" NAMESPACE="aiready" TARGET="${BACKUP_TARGET:-/backups/config}" RETENTION="${BACKUP_RETENTION_DAYS:-30}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir) CONFIG_DIR="${2:?}"; shift 2 ;;
    --overlay-dir) OVERLAY_DIR="${2:?}"; shift 2 ;;
    --values) VALUES+=("${2:?}"); shift 2 ;;
    --helm-release) RELEASE="${2:?}"; shift 2 ;;
    --namespace) NAMESPACE="${2:?}"; shift 2 ;;
    --target) TARGET="${2:?}"; shift 2 ;;
    --retention-days) RETENTION="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
[[ -d "$CONFIG_DIR" ]] || die "config directory not found: $CONFIG_DIR"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/aiready-cfg.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
STAGE="$WORK/config-backup"
mkdir -p "$STAGE/values"
# -L dereferences ConfigMap symlinks (..data/) so the archive holds the real files.
cp -RL "$CONFIG_DIR" "$STAGE/base"
if [[ -n "$OVERLAY_DIR" && -d "$OVERLAY_DIR" ]]; then cp -RL "$OVERLAY_DIR" "$STAGE/overlay"; fi
for v in ${VALUES[@]+"${VALUES[@]}"}; do cp "$v" "$STAGE/values/"; done
if [[ -n "$RELEASE" ]]; then
  have helm || die "--helm-release needs helm"
  helm get values "$RELEASE" --namespace "$NAMESPACE" --all -o yaml > "$STAGE/values/$RELEASE-effective.yaml"
fi
# Hidden ConfigMap bookkeeping and anything that looks like key material is left out.
find "$STAGE" \( -name '..*' -o -name '*.key' -o -name '*.pem' -o -name '.env' \) -prune -exec rm -rf {} +
{
  echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "config_dir=$CONFIG_DIR"
  echo "overlay_dir=${OVERLAY_DIR:-none}"
} > "$STAGE/BACKUP-INFO"

PREFIX="aiready-config-"
FILE="$WORK/${PREFIX}$(timestamp).tar.gz$(encrypt_suffix)"
tar -czf - -C "$WORK" config-backup | encrypt_stream > "$FILE"
write_checksum "$FILE"
upload "$FILE" "$TARGET"
prune "$TARGET" "$PREFIX" "$RETENTION"
log "config backup complete: $(basename "$FILE")"
