output "role_arns" {
  description = "Role ARNs keyed by purpose (annotate service accounts with eks.amazonaws.com/role-arn)."
  value       = { for k, r in aws_iam_role.this : k => r.arn }
}
