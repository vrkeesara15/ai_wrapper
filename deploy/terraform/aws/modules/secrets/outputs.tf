output "secret_arns" {
  description = "Secret ARNs keyed by secret name."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}

output "db_password" {
  description = "Generated database master password."
  value       = random_password.db.result
  sensitive   = true
}

output "redis_auth_token" {
  description = "Generated Redis AUTH token."
  value       = random_password.redis.result
  sensitive   = true
}
