#!/usr/bin/env bash
# Build a signed, self-describing release bundle for an air-gapped (on-prem) site.
#
# Runs on the CONNECTED build host. The bundle is the only thing that crosses into the client data
# centre (one-way import), so it carries everything a release needs: container images, the Helm
# chart, configuration, approved model weights with per-file hashes, SBOMs, a checksum list and
# signatures. Wrapped by `aiready bundle build`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=SCRIPTDIR/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --version VERSION [options]

  --version VERSION        Release version (required), e.g. 1.4.0
  --out DIR                Output directory (default: ./dist)
  --models FILE            Model manifest (see models.example.yaml); omit to ship no weights
  --platform-image REF     Platform image (default: \${REGISTRY:-ghcr.io/aiready}/aiready:VERSION)
  --images FILE            Third-party image list (default: $SCRIPT_DIR/images.txt)
  --no-pull                Use images already present locally (docker mode)
  -h, --help               Show this help

Environment:
  REGISTRY        Source registry for the platform image (default ghcr.io/aiready)
  COSIGN_KEY      cosign private key reference (file, hashivault://, awskms://, pkcs11:...).
                  If cosign is installed and COSIGN_KEY is set the bundle is signed.
  COSIGN_TLOG     1 to upload signatures to a transparency log (default 0: offline/private)
  PYTHON          Python interpreter with PyYAML (default python3)

Image transport: skopeo (oci-archive) if installed, otherwise docker pull + docker save.
SBOMs: one SPDX JSON per image if syft is installed.
EOF
}

VERSION="" OUT="./dist" MODELS="" PLATFORM_IMAGE="" IMAGES_FILE="$SCRIPT_DIR/images.txt" PULL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --models) MODELS="${2:?}"; shift 2 ;;
    --platform-image) PLATFORM_IMAGE="${2:?}"; shift 2 ;;
    --images) IMAGES_FILE="${2:?}"; shift 2 ;;
    --no-pull) PULL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
[[ -n "$VERSION" ]] || { usage >&2; die "--version is required"; }
[[ "$VERSION" =~ ^[A-Za-z0-9._+-]+$ ]] || die "invalid version: $VERSION"
PLATFORM_IMAGE="${PLATFORM_IMAGE:-${REGISTRY:-ghcr.io/aiready}/aiready:${VERSION}}"
[[ -f "$IMAGES_FILE" ]] || die "image list not found: $IMAGES_FILE"

NAME="aiready-bundle-${VERSION}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
STAGE="$OUT/$NAME"
[[ ! -e "$STAGE" ]] || die "$STAGE already exists; remove it or choose another --out"
mkdir -p "$STAGE"/{images,charts,config,models,sbom}

# ------------------------------------------------------------------ images
IMAGES=("$PLATFORM_IMAGE")
while IFS= read -r ref; do IMAGES+=("$ref"); done < <(grep -vE '^[[:space:]]*(#|$)' "$IMAGES_FILE" | awk '{print $1}')

if have skopeo; then TRANSPORT="oci-archive"; else TRANSPORT="docker-archive"; have docker || die "need skopeo or docker"; fi
log "packing ${#IMAGES[@]} images with $TRANSPORT"
: > "$STAGE/images/index.tsv"
for ref in "${IMAGES[@]}"; do
  file="images/$(safe_name "$ref").tar"
  target="$(internal_path "$ref")"
  if [[ "$TRANSPORT" == oci-archive ]]; then
    skopeo copy --quiet "docker://$ref" "oci-archive:$STAGE/$file:$target"
  else
    if [[ "$PULL" == 1 ]]; then docker pull --quiet "$ref" >/dev/null; fi
    docker save -o "$STAGE/$file" "$ref"
  fi
  # file <TAB> transport <TAB> source reference <TAB> path inside the internal registry
  printf '%s\t%s\t%s\t%s\n' "$file" "$TRANSPORT" "$ref" "$target" >> "$STAGE/images/index.tsv"
  if have syft; then
    syft --quiet "$TRANSPORT:$STAGE/$file" -o "spdx-json=$STAGE/sbom/$(safe_name "$ref").spdx.json"
  fi
  log "  $ref -> $file"
done
have syft || warn "syft not installed: bundle has no SBOMs (the import-side scanner must generate them)"

# ------------------------------------------------------------------ helm chart
CHART_DIR="$REPO_ROOT/deploy/helm/aiready"
[[ -f "$CHART_DIR/Chart.yaml" ]] || die "Helm chart not found at $CHART_DIR"
if have helm; then
  helm package "$CHART_DIR" --app-version "$VERSION" --destination "$STAGE/charts" >/dev/null
else
  warn "helm not installed: shipping the chart as a plain tar.gz of the chart directory"
  COPYFILE_DISABLE=1 tar -czhf "$STAGE/charts/aiready-chart-${VERSION}.tgz" -C "$(dirname "$CHART_DIR")" aiready
fi

# ------------------------------------------------------------------ configuration
if [[ -d "$REPO_ROOT/config" ]]; then
  cp -R "$REPO_ROOT/config" "$STAGE/config/base"
else
  warn "no config/ directory in the repository"
fi
if [[ -d "$REPO_ROOT/deploy/config-overlays" ]]; then
  cp -R "$REPO_ROOT/deploy/config-overlays" "$STAGE/config/overlays"
fi
# Secrets never travel in a bundle: they are created on site in Vault / the secret store.
if find "$STAGE/config" -type f \( -name '*.key' -o -name '*.pem' -o -name '.env' \) | grep -q .; then
  die "config contains key/.env files; secrets must never be bundled"
fi

# ------------------------------------------------------------------ model weights
if [[ -n "$MODELS" ]]; then
  [[ -f "$MODELS" ]] || die "model manifest not found: $MODELS"
  "$(python_bin)" "$SCRIPT_DIR/lib/models.py" pack --models "$MODELS" --out "$STAGE/models"
else
  warn "no --models given: bundle carries no weights (the model store keeps the current ones)"
  printf '{"format": 1, "models": []}\n' > "$STAGE/models/manifest.json"
fi

# ------------------------------------------------------------------ helper scripts travel with the bundle
mkdir -p "$STAGE/tools/lib" "$STAGE/tools/backup"
cp "$SCRIPT_DIR"/verify-bundle.sh "$SCRIPT_DIR"/import-bundle.sh "$STAGE/tools/"
cp "$SCRIPT_DIR"/lib/common.sh "$SCRIPT_DIR"/lib/models.py "$STAGE/tools/lib/"
cp "$SCRIPT_DIR"/backup/*.sh "$STAGE/tools/backup/" 2>/dev/null || true

# ------------------------------------------------------------------ bundle manifest
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
"$(python_bin)" - "$STAGE" "$VERSION" "$GIT_COMMIT" "$TRANSPORT" <<'PY'
import json, os, sys, time
stage, version, commit, transport = sys.argv[1:5]
images = []
with open(os.path.join(stage, "images", "index.tsv")) as f:
    for line in f:
        file, tr, src, target = line.rstrip("\n").split("\t")
        images.append({"file": file, "transport": tr, "source": src, "target": target})
models = json.load(open(os.path.join(stage, "models", "manifest.json")))
doc = {
    "name": "aiready",
    "version": version,
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_commit": commit,
    "image_transport": transport,
    "images": images,
    "charts": sorted(os.listdir(os.path.join(stage, "charts"))),
    "models": [{k: m[k] for k in ("name", "revision", "licence", "archive", "archive_sha256")}
               for m in models.get("models", [])],
    "sboms": sorted(os.listdir(os.path.join(stage, "sbom"))),
}
json.dump(doc, open(os.path.join(stage, "bundle.json"), "w"), indent=2, sort_keys=True)
PY

# ------------------------------------------------------------------ checksums (cover every file)
(
  cd "$STAGE"
  find . -type f ! -name SHA256SUMS ! -name '*.sig' | sed 's|^\./||' | LC_ALL=C sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha256_file "$f")" "$f"
  done > ../.SHA256SUMS.tmp && mv ../.SHA256SUMS.tmp SHA256SUMS
)
log "SHA256SUMS lists $(wc -l < "$STAGE/SHA256SUMS" | tr -d ' ') files"

# ------------------------------------------------------------------ signatures
if have cosign && [[ -n "${COSIGN_KEY:-}" ]]; then
  # shellcheck disable=SC2046  # flags are intentionally word-split
  cosign sign-blob $(cosign_sign_flags) --key "$COSIGN_KEY" --output-signature "$STAGE/SHA256SUMS.sig" "$STAGE/SHA256SUMS" >/dev/null
  # shellcheck disable=SC2046
  cosign sign-blob $(cosign_sign_flags) --key "$COSIGN_KEY" --output-signature "$STAGE/bundle.json.sig" "$STAGE/bundle.json" >/dev/null
  log "signed SHA256SUMS and bundle.json with cosign"
  # Image signatures: CI already signs each image in the source registry (cosign sign). Those
  # signatures are bound to the source registry, so the import side re-signs the imported images with
  # the site key (import-bundle.sh --sign-images) for admission control (Gatekeeper/Kyverno).
else
  warn "UNSIGNED BUNDLE: cosign not installed or COSIGN_KEY not set. verify-bundle.sh will refuse it unless --allow-unsigned."
fi

# ------------------------------------------------------------------ final archive
COPYFILE_DISABLE=1 tar -cf "$OUT/$NAME.tar" -C "$OUT" "$NAME"
( cd "$OUT" && printf '%s  %s\n' "$(sha256_file "$NAME.tar")" "$NAME.tar" > "$NAME.tar.sha256" )
log "bundle ready: $OUT/$NAME.tar ($(du -h "$OUT/$NAME.tar" | awk '{print $1}'))"
log "carry $NAME.tar and $NAME.tar.sha256 across; the cosign PUBLIC key is distributed separately"
echo "$OUT/$NAME.tar"
