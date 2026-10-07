# shellcheck shell=bash
# Shared helpers for the release-bundle scripts (build / verify / import).
# Sourced, never executed. Kept free of non-POSIX tools beyond bash, tar and a sha256 tool so it
# runs on the connected build host (often macOS) and on the air-gapped import host (Linux).

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
warn() { printf '[%s] WARNING: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# sha256 of one file, hash only. GNU coreutils on Linux, shasum on macOS.
sha256_file() {
  if have sha256sum; then
    sha256sum "$1" | awk '{print $1}'
  elif have shasum; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is installed"
  fi
}

# Check a "<hash>  <path>" list (read relative to the current directory).
sha256_check() {
  if have sha256sum; then
    sha256sum --quiet -c "$1"
  elif have shasum; then
    shasum -a 256 --quiet -c "$1"
  else
    die "neither sha256sum nor shasum is installed"
  fi
}

# Turn an image reference into a file-system safe name: ghcr.io/a/b:1.0 -> ghcr.io_a_b_1.0
safe_name() { printf '%s' "$1" | tr '/:@' '___'; }

# Repository path inside the internal registry: drop the source registry host if the first path
# component looks like one (contains a dot or a port, or is localhost).
#   ghcr.io/huggingface/text-embeddings-inference:1.5 -> huggingface/text-embeddings-inference:1.5
#   redis:7.4-alpine                                  -> redis:7.4-alpine
internal_path() {
  local ref="$1" first rest
  first="${ref%%/*}"
  if [[ "$ref" == */* ]] && { [[ "$first" == *.* ]] || [[ "$first" == *:* ]] || [[ "$first" == localhost ]]; }; then
    rest="${ref#*/}"
    printf '%s' "$rest"
  else
    printf '%s' "$ref"
  fi
}

python_bin() {
  if [[ -n "${PYTHON:-}" ]]; then printf '%s' "$PYTHON"; return; fi
  if have python3; then printf 'python3'; return; fi
  die "python3 is required (set PYTHON=/path/to/python)"
}

# cosign transparency log: an air-gapped site cannot reach a public Rekor, and signing payloads must
# not be published there either. COSIGN_TLOG=1 opts back in (e.g. a private Rekor instance).
cosign_sign_flags()   { if [[ "${COSIGN_TLOG:-0}" == 1 ]]; then echo "--yes"; else echo "--yes --tlog-upload=false"; fi; }
cosign_verify_flags() { if [[ "${COSIGN_TLOG:-0}" == 1 ]]; then echo ""; else echo "--insecure-ignore-tlog=true"; fi; }
