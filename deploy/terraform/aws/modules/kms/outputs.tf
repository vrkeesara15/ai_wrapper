output "eks_key_arn" {
  description = "KMS key for Kubernetes Secrets envelope encryption."
  value       = aws_kms_key.this["eks"].arn
}

output "data_key_arn" {
  description = "KMS key for data at rest."
  value       = aws_kms_key.this["data"].arn
}

output "logs_key_arn" {
  description = "KMS key for CloudWatch log groups."
  value       = aws_kms_key.this["logs"].arn
}
