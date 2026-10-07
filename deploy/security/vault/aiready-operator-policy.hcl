# Vault policy for platform operators: rotate keys and manage secrets, but never read plaintext key material.
# Used by the runbooks docs/runbooks/rotate-signing-keys.md and incident-response.md.

path "transit/keys/aiready-token-signing/rotate" {
  capabilities = ["update"]
}
path "transit/keys/aiready-token-signing/config" {
  capabilities = ["update"]   # min_decryption_version / min_encryption_version to retire old keys
}
path "transit/keys/aiready-audit/rotate" {
  capabilities = ["update"]
}
path "transit/keys/*" {
  capabilities = ["read", "list"]
}
path "secret/data/aiready/platform/*" {
  capabilities = ["create", "update", "read"]
}
path "secret/metadata/aiready/platform/*" {
  capabilities = ["list", "read", "delete"]
}
# Revoke all leases for a compromised role during an incident.
path "sys/leases/revoke-prefix/database/creds/aiready-*" {
  capabilities = ["update", "sudo"]
}
