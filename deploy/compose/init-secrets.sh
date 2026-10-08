#!/usr/bin/env bash
# Create the pilot's secret files (once). Existing files are never overwritten, so keys survive restarts.
# In stages 2/3 these come from the client's vault (External Secrets Operator or Vault agent) instead.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/secrets"
mkdir -p "$dir"
chmod 700 "$dir"

gen() {
  local name="$1" bytes="${2:-48}"
  local file="$dir/$name"
  if [[ -s "$file" ]]; then
    return 0
  fi
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 "$bytes" | tr -d '\n' >"$file"
  else
    head -c "$bytes" /dev/urandom | base64 | tr -d '\n' >"$file"
  fi
  # Readable by the container uid (10001) through the bind mount; not by other host users.
  chmod 644 "$file"
  echo "created secrets/$name"
}

gen token_signing_key 64
gen audit_hmac_key 64
gen service_token_key 64
gen grafana_admin_password 24
# No vendor API in the pilot by default: an empty key keeps public routes closed.
[[ -e "$dir/llm_api_key" ]] || { : >"$dir/llm_api_key"; chmod 644 "$dir/llm_api_key"; echo "created secrets/llm_api_key (empty)"; }

# Qdrant reads its API key from the environment, so it lives in .env (generated once, never overwritten).
env_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
if [[ -f "$env_file" ]] && grep -qE '^QDRANT_API_KEY=$' "$env_file"; then
  key="$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  sed -i.bak "s/^QDRANT_API_KEY=$/QDRANT_API_KEY=${key}/" "$env_file" && rm -f "$env_file.bak"
  echo "generated QDRANT_API_KEY in .env"
fi
