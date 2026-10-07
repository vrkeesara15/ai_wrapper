# IAM roles for Kubernetes service accounts. No shared data-rights account: each role is bound to named
# service accounts and grants only what that workload needs.
#   external_secrets - read the AI-Ready secrets (sync into the cluster)
#   data_reader      - read documents and model weights (ingestion, model loader init containers)
#   backup           - write snapshots to the backups bucket (backup CronJobs)
#   private_model    - invoke Bedrock through the VPC endpoint (only when enabled; gateway only)

locals {
  oidc_host = replace(var.oidc_provider_url, "https://", "")

  subjects = {
    external_secrets = ["system:serviceaccount:${var.external_secrets_namespace}:${var.external_secrets_service_account}"]
    data_reader      = [for sa in var.data_reader_service_accounts : "system:serviceaccount:${var.namespace}:${sa}"]
    backup           = [for sa in var.backup_service_accounts : "system:serviceaccount:${var.namespace}:${sa}"]
    private_model    = ["system:serviceaccount:${var.namespace}:gateway"]
  }

  roles = var.enable_private_model_endpoint ? local.subjects : { for k, v in local.subjects : k => v if k != "private_model" }
}

data "aws_iam_policy_document" "assume" {
  for_each = local.roles

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = each.value
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "this" {
  for_each = local.roles

  name               = "${var.name}-${replace(each.key, "_", "-")}"
  assume_role_policy = data.aws_iam_policy_document.assume[each.key].json
  tags               = var.tags
}

data "aws_iam_policy_document" "external_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret", "secretsmanager:ListSecretVersionIds"]
    resources = values(var.secret_arns)
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = [var.data_kms_key_arn]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["secretsmanager.${var.region}.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "data_reader" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [var.documents_bucket_arn, var.weights_bucket_arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${var.documents_bucket_arn}/*", "${var.weights_bucket_arn}/*"]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = [var.data_kms_key_arn]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["s3.${var.region}.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "backup" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [var.backups_bucket_arn]
  }
  statement {
    actions   = ["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload"]
    resources = ["${var.backups_bucket_arn}/*"]
  }
  statement {
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [var.data_kms_key_arn]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["s3.${var.region}.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "private_model" {
  statement {
    actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = var.private_model_arns
  }
}

locals {
  policies = {
    external_secrets = data.aws_iam_policy_document.external_secrets.json
    data_reader      = data.aws_iam_policy_document.data_reader.json
    backup           = data.aws_iam_policy_document.backup.json
    private_model    = data.aws_iam_policy_document.private_model.json
  }
}

resource "aws_iam_role_policy" "this" {
  for_each = local.roles

  name   = replace(each.key, "_", "-")
  role   = aws_iam_role.this[each.key].id
  policy = local.policies[each.key]
}
