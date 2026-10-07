# CloudWatch log groups, created up front so they carry the client's KMS key and a retention rule.
# The EKS control-plane group uses the exact name EKS writes to, so EKS reuses it instead of creating
# an unencrypted one. Service logs normally go through the OpenTelemetry collector (deploy/observability);
# the application group is the CloudWatch export target for that collector.

locals {
  groups = {
    control_plane = { name = "/aws/eks/${var.cluster_name}/cluster", retention = var.log_retention_days }
    application   = { name = "/aiready/${var.name}/application", retention = var.log_retention_days }
    audit         = { name = "/aiready/${var.name}/audit", retention = var.audit_log_retention_days }
    flow_logs     = { name = "/aiready/${var.name}/vpc-flow-logs", retention = var.log_retention_days }
  }
}

resource "aws_cloudwatch_log_group" "this" {
  for_each = local.groups

  name              = each.value.name
  retention_in_days = each.value.retention
  kms_key_id        = var.logs_kms_key_arn
  tags              = merge(var.tags, { Name = each.value.name })
}
