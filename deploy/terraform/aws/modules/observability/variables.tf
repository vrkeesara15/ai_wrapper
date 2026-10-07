variable "name" {
  description = "Name prefix."
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name (control-plane log group name)."
  type        = string
}

variable "logs_kms_key_arn" {
  description = "KMS key encrypting the log groups."
  type        = string
}

variable "log_retention_days" {
  description = "Retention for service, control-plane and flow logs."
  type        = number
}

variable "audit_log_retention_days" {
  description = "Retention for the audit log group."
  type        = number
}

variable "tags" {
  description = "Tags."
  type        = map(string)
  default     = {}
}
