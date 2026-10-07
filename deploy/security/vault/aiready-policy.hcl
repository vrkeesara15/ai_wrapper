# Vault policy for AI-Ready services ("Keys and secrets: vault, keys held by the client").
#
# Bound to the Kubernetes auth role `aiready` (service account `aiready` in namespace `aiready`):
#   vault write auth/kubernetes/role/aiready \
#     bound_service_account_names=aiready bound_service_account_namespaces=aiready \
#     policies=aiready ttl=1h
# The Vault agent injector (chart: secrets.mode=vault) renders each secret to /run/secrets/<name>,
# which is exactly where aiready.core.config.resolve_secret looks.

# Read-only access to the platform's own secrets (KV v2).
path "secret/data/aiready/platform/*" {
  capabilities = ["read"]
}
path "secret/metadata/aiready/platform/*" {
  capabilities = ["list", "read"]
}

# Token signing uses Transit so the private key never leaves Vault (HSM-backed on-prem). The gateway and
# identity bridge sign/verify; nobody can export the key.
path "transit/sign/aiready-token-signing" {
  capabilities = ["update"]
}
path "transit/verify/aiready-token-signing" {
  capabilities = ["update"]
}
path "transit/keys/aiready-token-signing" {
  capabilities = ["read"]
}

# Audit-chain HMAC key: services only compute HMACs; rotation is an operator action.
path "transit/hmac/aiready-audit" {
  capabilities = ["update"]
}

# Short-lived database credentials (dynamic secrets) instead of a static password.
path "database/creds/aiready-app" {
  capabilities = ["read"]
}
path "database/creds/aiready-readonly" {
  capabilities = ["read"]
}

# Explicitly nothing else.
path "sys/*" {
  capabilities = ["deny"]
}
