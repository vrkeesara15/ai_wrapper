# Three customer-managed keys so access can be granted and revoked per purpose:
#   eks  - envelope encryption of Kubernetes Secrets
#   data - RDS, S3, ElastiCache, Secrets Manager, ECR
#   logs - CloudWatch log groups (needs the logs service principal in the key policy)

data "aws_iam_policy_document" "base" {
  statement {
    sid       = "AccountAdministration"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${var.account_id}:root"]
    }
  }
}

data "aws_iam_policy_document" "logs" {
  source_policy_documents = [data.aws_iam_policy_document.base.json]

  statement {
    sid       = "CloudWatchLogs"
    actions   = ["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["logs.${var.region}.amazonaws.com"]
    }
    condition {
      test     = "ArnLike"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values   = ["arn:aws:logs:${var.region}:${var.account_id}:log-group:*"]
    }
  }
}

locals {
  keys = {
    eks  = { description = "${var.name}: EKS secrets envelope encryption", policy = data.aws_iam_policy_document.base.json }
    data = { description = "${var.name}: data at rest (RDS, S3, Redis, Secrets Manager, ECR)", policy = data.aws_iam_policy_document.base.json }
    logs = { description = "${var.name}: CloudWatch logs", policy = data.aws_iam_policy_document.logs.json }
  }
}

resource "aws_kms_key" "this" {
  for_each = local.keys

  description             = each.value.description
  enable_key_rotation     = true
  deletion_window_in_days = var.deletion_window_days
  policy                  = each.value.policy
  tags                    = merge(var.tags, { Name = "${var.name}-${each.key}" })
}

resource "aws_kms_alias" "this" {
  for_each = local.keys

  name          = "alias/${var.name}-${each.key}"
  target_key_id = aws_kms_key.this[each.key].key_id
}
