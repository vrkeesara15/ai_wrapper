#!/usr/bin/env bash
# Verify a release bundle BEFORE anything from it is imported.
#
# Runs on the air-gapped side (scan-and-verify station). Order matters: the signature proves
# SHA256SUMS came from the release key; SHA256SUMS then proves every file; the model manifest then
# proves every weight file inside each archive. Any failure exits non-zero and nothing is imported.
# Use the copy of this script already trusted on site, not the one inside the bundle being checked.
# Wrapped by `aiready bundle verify`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --bundle PATH [options]

  --bundle PATH       Bundle .tar file or an already extracted bundle directory (required)
  --key FILE          cosign public key (default: \$COSIGN_PUBLIC_KEY, else /etc/aiready/cosign.pub)
  --extract-to DIR    Where to extract a .tar bundle (default: a new temporary directory)
  --allow-unsigned    Accept a bundle without signatures. Only for test environments.
  -h, --help          Show this help

On success the last line on stdout is: VERIFIED <bundle directory>
EOF
}

BUNDLE="" KEY="${COSIGN_PUBLIC_KEY:-/etc/aiready/cosign.pub}" EXTRACT_TO="" ALLOW_UNSIGNED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:?}"; shift 2 ;;
    --key) KEY="${2:?}"; shift 2 ;;
    --extract-to) EXTRACT_TO="${2:?}"; shift 2 ;;
    --allow-unsigned) ALLOW_UNSIGNED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
[[ -n "$BUNDLE" ]] || { usage >&2; die "--bundle is required"; }
[[ -e "$BUNDLE" ]] || die "bundle not found: $BUNDLE"

# ------------------------------------------------------------------ unpack
if [[ -f "$BUNDLE" ]]; then
  if [[ -f "$BUNDLE.sha256" ]]; then
    expected="$(awk '{print $1}' "$BUNDLE.sha256")"
    [[ "$(sha256_file "$BUNDLE")" == "$expected" ]] || die "outer archive checksum mismatch ($BUNDLE.sha256)"
    log "outer archive checksum ok"
  fi
  # Reject absolute paths and parent-directory entries before extracting anything.
  if tar -tf "$BUNDLE" | grep -qE '(^/|(^|/)\.\.(/|$))'; then
    die "archive contains absolute or '..' paths; refusing"
  fi
  tops="$(tar -tf "$BUNDLE" | cut -d/ -f1 | sort -u)"
  [[ "$(printf '%s\n' "$tops" | wc -l | tr -d ' ')" == 1 ]] || die "archive must contain exactly one top-level directory"
  EXTRACT_TO="${EXTRACT_TO:-$(mktemp -d "${TMPDIR:-/tmp}/aiready-bundle.XXXXXX")}"
  mkdir -p "$EXTRACT_TO"
  tar -xf "$BUNDLE" -C "$EXTRACT_TO" --no-same-owner
  DIR="$EXTRACT_TO/$tops"
else
  DIR="$(cd "$BUNDLE" && pwd)"
fi
[[ -f "$DIR/SHA256SUMS" && -f "$DIR/bundle.json" ]] || die "not a release bundle: $DIR"

# ------------------------------------------------------------------ signatures
if [[ -f "$DIR/SHA256SUMS.sig" ]]; then
  have cosign || die "bundle is signed but cosign is not installed on this host"
  [[ -f "$KEY" ]] || die "cosign public key not found: $KEY (distributed out of band)"
  for f in SHA256SUMS bundle.json; do
    [[ -f "$DIR/$f.sig" ]] || die "missing signature $f.sig"
    # shellcheck disable=SC2046  # flags are intentionally word-split
    cosign verify-blob $(cosign_verify_flags) --key "$KEY" --signature "$DIR/$f.sig" "$DIR/$f" >/dev/null 2>&1 \
      || die "SIGNATURE INVALID for $f: do not import this bundle"
  done
  log "signatures valid (key: $KEY)"
elif [[ "$ALLOW_UNSIGNED" == 1 ]]; then
  warn "############################################################"
  warn "#  BUNDLE IS NOT SIGNED. Origin cannot be proven.          #"
  warn "#  Continuing only because --allow-unsigned was given.     #"
  warn "#  Never use this for a production import.                #"
  warn "############################################################"
else
  die "bundle is not signed (no SHA256SUMS.sig); refusing. Use --allow-unsigned only for test environments."
fi

# ------------------------------------------------------------------ checksums
(
  cd "$DIR"
  sha256_check SHA256SUMS || die "checksum mismatch: bundle is corrupt or tampered"
  # Every file present must be listed: nothing may be slipped in beside the signed list.
  listed="$(awk '{ $1=""; sub(/^ +/, ""); print }' SHA256SUMS | LC_ALL=C sort)"
  present="$(find . -type f ! -name SHA256SUMS ! -name 'SHA256SUMS.sig' ! -name 'bundle.json.sig' \
    | sed 's|^\./||' | LC_ALL=C sort)"
  if [[ "$listed" != "$present" ]]; then
    diff <(printf '%s\n' "$listed") <(printf '%s\n' "$present") >&2 || true
    die "bundle contents differ from SHA256SUMS (unlisted or missing files)"
  fi
)
log "all $(wc -l < "$DIR/SHA256SUMS" | tr -d ' ') files match SHA256SUMS"

# ------------------------------------------------------------------ model weights, file by file
if [[ -f "$DIR/models/manifest.json" ]]; then
  "$(python_bin)" "$SCRIPT_DIR/lib/models.py" verify --dir "$DIR/models" || die "model weights failed verification"
fi

log "bundle $(basename "$DIR") verified"
echo "VERIFIED $DIR"
