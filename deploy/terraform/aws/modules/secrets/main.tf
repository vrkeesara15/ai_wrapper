# Secrets Manager entries, named after the files the services read from /run/secrets/<name>
# (see aiready.core.config.resolve_secret). External Secrets Operator (or the Vault agent) syncs them into
# the cluster. Generated values never appear in variables; llm_api_key is created empty and filled out of
# band only when a public / private-endpoint model route is approved by policy.

resource "random_password" "token_signing_key" {
  length  = 64
  special = false
}

resource "random_password" "audit_hmac_key" {
  length  = 64
  special = false
}

# RDS forbids / @ " and spaces in master passwords.
resource "random_password" "db" {
  length           = 40
  special          = true
  override_special = "!#$%^&*()-_=+[]{}<>:?"
}

# ElastiCache AUTH: 16-128 printable characters, no @ " /.
resource "random_password" "redis" {
  length  = 64
  special = false
}

resource "random_password" "service_token_key" {
  length  = 64
  special = false
}

resource "random_password" "qdrant_api_key" {
  length  = 48
  special = false
}

locals {
  generated = {
    service_token_key = random_password.service_token_key.result
    qdrant_api_key    = random_password.qdrant_api_key.result
    token_signing_key = random_password.token_signing_key.result
    audit_hmac_key    = random_password.audit_hmac_key.result
    db_password       = random_password.db.result
    redis_auth_token  = random_password.redis.result
  }

  all_names = toset(concat(keys(local.generated), ["llm_api_key"], var.extra_secret_names))
}

resource "aws_secretsmanager_secret" "this" {
  for_each = local.all_names

  name                    = "${var.name}/${each.key}"
  description             = "AI-Ready secret ${each.key} (mounted as /run/secrets/${each.key})"
  kms_key_id              = var.data_kms_key_arn
  recovery_window_in_days = var.recovery_window_days
  tags                    = merge(var.tags, { "aiready.io/secret" = each.key })
}

resource "aws_secretsmanager_secret_version" "generated" {
  for_each = local.generated

  secret_id     = aws_secretsmanager_secret.this[each.key].id
  secret_string = each.value

  # Rotation (aiready token / audit runbooks) writes new versions outside Terraform.
  lifecycle {
    ignore_changes = [secret_string]
  }
}
