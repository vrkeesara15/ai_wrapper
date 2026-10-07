#!/usr/bin/env bash
# Import a verified release bundle into the air-gapped site: images into the internal registry,
# weights into the model store, the chart into the internal chart repository.
#
# Verification is not optional: this script extracts the bundle into a private directory, runs
# verify-bundle.sh on exactly those files, and imports only from them (no check-then-swap window).
# Every attempt, successful or not, is appended to the import log. Wrapped by `aiready bundle import`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --bundle FILE.tar [options]

  --bundle FILE          Release bundle archive (required)
  --key FILE             cosign public key (default: \$COSIGN_PUBLIC_KEY or /etc/aiready/cosign.pub)
  --registry HOST[/NS]   Internal registry (default: \$INTERNAL_REGISTRY, required)
  --model-store DEST     Filesystem path or s3://bucket/prefix (default: \$MODEL_STORE, required
                         when the bundle carries weights). S3 uses \$S3_ENDPOINT for MinIO etc.
  --chart-repo DIR       Internal chart repository directory (default: \$CHART_REPO or /var/lib/aiready/charts)
  --namespace NS         Namespace used in the printed helm command (default: aiready)
  --log FILE             Append-only import log (default: /var/log/aiready/bundle-imports.log)
  --sign-images KEY      Re-sign imported images with the site cosign key (admission control)
  --allow-unsigned       Passed to verify-bundle.sh. Test environments only.
  -h, --help             Show this help
EOF
}

BUNDLE="" KEY="${COSIGN_PUBLIC_KEY:-/etc/aiready/cosign.pub}" REGISTRY="${INTERNAL_REGISTRY:-}"
STORE="${MODEL_STORE:-}" CHART_REPO="${CHART_REPO:-/var/lib/aiready/charts}" NAMESPACE="aiready" LOG_FILE="/var/log/aiready/bundle-imports.log"
SIGN_KEY="" ALLOW_UNSIGNED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:?}"; shift 2 ;;
    --key) KEY="${2:?}"; shift 2 ;;
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --model-store) STORE="${2:?}"; shift 2 ;;
    --chart-repo) CHART_REPO="${2:?}"; shift 2 ;;
    --namespace) NAMESPACE="${2:?}"; shift 2 ;;
    --log) LOG_FILE="${2:?}"; shift 2 ;;
    --sign-images) SIGN_KEY="${2:?}"; shift 2 ;;
    --allow-unsigned) ALLOW_UNSIGNED=(--allow-unsigned); shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
[[ -f "$BUNDLE" ]] || { usage >&2; die "--bundle must be an existing .tar file"; }
[[ -n "$REGISTRY" ]] || die "--registry or INTERNAL_REGISTRY is required"
REGISTRY="${REGISTRY%/}"

BUNDLE_SHA="$(sha256_file "$BUNDLE")"
VERSION="unknown"
STATUS="failed"
mkdir -p "$(dirname "$LOG_FILE")"
record() {
  # One line per attempt: when, who, where, what, outcome. Opened in append mode only; on site,
  # protect the file with `chattr +a` and forward it to security monitoring.
  printf '%s\tuser=%s\thost=%s\tversion=%s\tbundle_sha256=%s\tregistry=%s\tstatus=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SUDO_USER:-$(id -un)}" "$(hostname)" "$VERSION" "$BUNDLE_SHA" \
    "$REGISTRY" "$STATUS" >> "$LOG_FILE"
}
WORK="$(mktemp -d "${TMPDIR:-/tmp}/aiready-import.XXXXXX")"
cleanup() { record; rm -rf "$WORK"; }
trap cleanup EXIT

# ------------------------------------------------------------------ verify (mandatory)
result="$("$SCRIPT_DIR/verify-bundle.sh" --bundle "$BUNDLE" --key "$KEY" --extract-to "$WORK" ${ALLOW_UNSIGNED[@]+"${ALLOW_UNSIGNED[@]}"} | tail -n 1)"
[[ "$result" == VERIFIED\ * ]] || die "verification did not succeed; nothing imported"
DIR="${result#VERIFIED }"
VERSION="$("$(python_bin)" -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$DIR/bundle.json")"
log "importing aiready $VERSION"

# ------------------------------------------------------------------ images
while IFS=$'\t' read -r file transport src target; do
  dest="$REGISTRY/$target"
  if have skopeo; then
    skopeo copy --quiet "$transport:$DIR/$file" "docker://$dest"
  else
    have docker || die "need skopeo or docker to load images"
    [[ "$transport" == docker-archive ]] || die "$file is an OCI archive; skopeo is required to import it"
    docker load -q -i "$DIR/$file" >/dev/null
    docker tag "$src" "$dest"
    docker push -q "$dest" >/dev/null
  fi
  if [[ -n "$SIGN_KEY" ]]; then
    have cosign || die "--sign-images needs cosign"
    # Sign by digest so the signature cannot be moved to another image under the same tag.
    if have skopeo; then
      digest="$(skopeo inspect --format '{{.Digest}}' "docker://$dest")"
    else
      digest="$(docker inspect --format '{{index .RepoDigests 0}}' "$dest" | sed 's/.*@//')"
    fi
    # shellcheck disable=SC2046
    cosign sign $(cosign_sign_flags) --key "$SIGN_KEY" "${dest%:*}@$digest" >/dev/null
  fi
  log "  image $src -> $dest"
done < "$DIR/images/index.tsv"

# ------------------------------------------------------------------ model weights
model_count="$("$(python_bin)" -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("models", [])))' "$DIR/models/manifest.json")"
endpoint=()
if [[ "$model_count" -gt 0 ]]; then
  [[ -n "$STORE" ]] || die "bundle carries $model_count model(s): --model-store or MODEL_STORE is required"
  if [[ "$STORE" == s3://* ]]; then
    have aws || die "aws CLI required for an s3:// model store"
    staging="$WORK/model-store"
    "$(python_bin)" "$SCRIPT_DIR/lib/models.py" unpack --dir "$DIR/models" --dest "$staging"
    if [[ -n "${S3_ENDPOINT:-}" ]]; then endpoint=(--endpoint-url "$S3_ENDPOINT"); fi
    aws ${endpoint[@]+"${endpoint[@]}"} s3 cp --only-show-errors --recursive "$staging/" "${STORE%/}/"
  else
    mkdir -p "$STORE"
    "$(python_bin)" "$SCRIPT_DIR/lib/models.py" unpack --dir "$DIR/models" --dest "$STORE"
  fi
  # Keep the bundle-level manifest too, named by version, so the store records what each release shipped.
  if [[ "$STORE" == s3://* ]]; then
    aws ${endpoint[@]+"${endpoint[@]}"} s3 cp --only-show-errors "$DIR/models/manifest.json" "${STORE%/}/manifests/$VERSION.json"
  else
    mkdir -p "$STORE/manifests" && cp "$DIR/models/manifest.json" "$STORE/manifests/$VERSION.json"
  fi
  log "  $model_count model(s) -> $STORE (each with manifest.json and SHA256SUMS beside the weights)"
fi

# ------------------------------------------------------------------ chart
chart="$(find "$DIR/charts" -name '*.tgz' | head -n 1)"
[[ -n "$chart" ]] || die "no chart in bundle"
mkdir -p "$CHART_REPO"
cp "$chart" "$CHART_REPO/"
if have helm; then helm repo index "$CHART_REPO" >/dev/null; fi
chart_ref="$CHART_REPO/$(basename "$chart")"
STATUS="imported"
log "import complete. Roll out with (after reviewing values for this site):"
cat <<EOF

  helm upgrade --install aiready $chart_ref \\
    --namespace $NAMESPACE --create-namespace \\
    -f values-production.yaml -f values-site.yaml \\
    --set global.imageRegistry=$REGISTRY \\
    --set image.tag=$VERSION

EOF
