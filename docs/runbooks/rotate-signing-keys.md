# Rotating signing keys

**Owner:** platform engineer with the client's security lead. **Schedule:** token signing key every 90
days, service token key every 90 days, audit HMAC key every year. Rotate immediately on suspected exposure
(see [incident-response.md](incident-response.md)).

Keys are held by the client: in Vault (Transit, HSM-backed on-prem), AWS Secrets Manager/KMS, or the
pilot's `deploy/compose/secrets/`. Services read `/run/secrets/<name>` and reload them without a restart
(the config store resolves `${secret:...}` again on reload). If a service does not pick the new key up,
restart it with `kubectl rollout restart`.

## Token signing key (`token_signing_key`) - user tokens issued by the identity bridge

Overlap rotation, so that no signed-in user is cut off:
1. Generate the new key and publish it **as an accepted verification key** alongside the current one:
   `aiready token rotate --stage add` (Vault Transit: `vault write -f transit/keys/aiready-token-signing/rotate`).
   Verifiers now accept both keys.
2. Wait for one reload interval, then switch signing to the new key: `aiready token rotate --stage activate`.
3. Wait for the maximum token lifetime (short-lived tokens; see `security.tokens.ttl`), then retire the old
   key: `aiready token rotate --stage retire` (Vault: raise `min_decryption_version`).
4. Check: `aiready token verify --sample` against a new token, and confirm there is no rise in 401s on the
   gateway dashboard.

On **exposure**, skip the overlap: activate the new key and retire the old one at once. Every user then
signs in again through the existing application, which is unaffected.

## Service token key (`service_token_key`)

Use the same three stages: `aiready token rotate --key service --stage add|activate|retire`. Service tokens
only prove which service relays a call (they carry no data rights), but rotate them as carefully as the user
key.

## Audit HMAC key (`audit_hmac_key`)

The audit chain must stay verifiable across the rotation:
1. `aiready audit verify` must pass before you rotate.
2. `aiready audit rotate-key` writes a signed `audit.key_rotated` record that links the chains, then
   switches to the new key.
3. Keep the old key (read-only, in Vault) for as long as the retention period lasts, so that historic
   records stay verifiable.
4. `aiready audit verify --all-chains` passes after the rotation.

## Release signing key (cosign)

The private key is in the HSM/Vault and the public key is distributed out of band. Rotating it requires the
new public key in the Kyverno policy (`deploy/security/kyverno/policies.yaml`) **before** the first image is
signed with the new key. On-prem sites must also receive the new public key before the next bundle.

Record: the key name, the stage timestamps, the operator, and the verification output.
