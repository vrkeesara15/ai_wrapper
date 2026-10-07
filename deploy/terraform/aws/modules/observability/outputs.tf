output "flow_log_group_arn" {
  description = "Log group receiving VPC flow logs."
  value       = aws_cloudwatch_log_group.this["flow_logs"].arn
}

output "application_log_group_name" {
  description = "Log group for service logs exported by the collector."
  value       = aws_cloudwatch_log_group.this["application"].name
}

output "audit_log_group_name" {
  description = "Log group for the audit stream."
  value       = aws_cloudwatch_log_group.this["audit"].name
}

output "control_plane_log_group_name" {
  description = "EKS control-plane log group."
  value       = aws_cloudwatch_log_group.this["control_plane"].name
}
