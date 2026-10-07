variable "name" {
  description = "Name prefix."
  type        = string
}

variable "region" {
  description = "Region (kms:ViaService conditions)."
  type        = string
}

variable "oidc_provider_arn" {
  description = "EKS OIDC provider ARN."
  type        = string
}

variable "oidc_provider_url" {
  description = "EKS OIDC issuer URL."
  type        = string
}

variable "namespace" {
  description = "Namespace of the AI-Ready release."
  type        = string
}

variable "external_secrets_namespace" {
  description = "External Secrets Operator namespace."
  type        = string
}

variable "external_secrets_service_account" {
  description = "External Secrets Operator service account."
  type        = string
}

variable "data_reader_service_accounts" {
  description = "Service accounts allowed to read documents and weights."
  type        = list(string)
}

variable "backup_service_accounts" {
  description = "Service accounts allowed to write backups."
  type        = list(string)
}

variable "secret_arns" {
  description = "Secrets Manager ARNs readable by External Secrets."
  type        = map(string)
}

variable "data_kms_key_arn" {
  description = "Data KMS key."
  type        = string
}

variable "documents_bucket_arn" {
  description = "Documents bucket ARN."
  type        = string
}

variable "weights_bucket_arn" {
  description = "Model weights bucket ARN."
  type        = string
}

variable "backups_bucket_arn" {
  description = "Backups bucket ARN."
  type        = string
}

variable "enable_private_model_endpoint" {
  description = "Create the Bedrock invoke role for the gateway service account."
  type        = bool
  default     = false
}

variable "private_model_arns" {
  description = "Bedrock foundation-model ARNs the gateway may invoke (approved in the model registry)."
  type        = list(string)
  default     = ["arn:aws:bedrock:*::foundation-model/*"]
}

variable "tags" {
  description = "Tags."
  type        = map(string)
  default     = {}
}
