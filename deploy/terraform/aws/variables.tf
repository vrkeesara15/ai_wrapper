# ---------------------------------------------------------------------------- general

variable "name" {
  description = "Name prefix for every resource (e.g. aiready-prod). Lower case, letters, digits and dashes."
  type        = string
  default     = "aiready"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}$", var.name))
    error_message = "name must be 2-31 characters: lower case letters, digits and dashes, starting with a letter."
  }
}

variable "region" {
  description = "AWS region. One region, three availability zones (see the cloud topology slide)."
  type        = string
  default     = "eu-west-1"
}

variable "stage" {
  description = "Scaling stage: pilot (stage 1 on a private endpoint), production (stage 2) or platform (stage 3)."
  type        = string
  default     = "production"

  validation {
    condition     = contains(["pilot", "production", "platform"], var.stage)
    error_message = "stage must be one of pilot, production, platform."
  }
}

variable "tags" {
  description = "Extra tags merged onto every resource (cost centre, owner, data classification)."
  type        = map(string)
  default     = {}
}

# ---------------------------------------------------------------------------- network

variable "vpc_cidr" {
  description = "CIDR of the platform VPC. Must not overlap the existing application's network if they are peered."
  type        = string
  default     = "10.40.0.0/16"
}

variable "availability_zones" {
  description = "Exactly three AZs. Empty means the first three available AZs in the region."
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.availability_zones) == 0 || length(var.availability_zones) == 3
    error_message = "Provide exactly three availability zones, or none to pick automatically."
  }
}

variable "enable_egress_nat" {
  description = <<-EOT
    Create the single controlled route out: an egress subnet per AZ with a NAT gateway, used ONLY by the
    egress proxy node group. Workload subnets never get a default route. Leave false for a fully closed
    network (models served in-cluster, images from ECR via endpoints).
  EOT
  type        = bool
  default     = false
}

variable "enable_private_model_endpoint" {
  description = "Create a Bedrock runtime interface endpoint (a vendor model inside this account, opened by policy)."
  type        = bool
  default     = false
}

variable "app_network_cidrs" {
  description = "CIDRs of the existing application (peered / transit gateway) allowed to reach the internal load balancer."
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------- eks

variable "kubernetes_version" {
  description = "EKS control plane version."
  type        = string
  default     = "1.31"
}

variable "admin_principal_arns" {
  description = "IAM role/user ARNs granted cluster-admin through EKS access entries (platform engineers, CI deploy role)."
  type        = list(string)
  default     = []
}

variable "node_groups" {
  description = <<-EOT
    Managed node groups. Keys must stay cpu, gpu_a, gpu_b, batch (labels and taints are fixed by the module so
    they match the Helm chart). gpu_a.min_size may be 0 for stage 3 scale-to-zero.
  EOT
  type = map(object({
    instance_types = list(string)
    min_size       = number
    desired_size   = number
    max_size       = number
    disk_size_gb   = optional(number, 100)
  }))
  default = {
    cpu = {
      instance_types = ["m6i.2xlarge"]
      min_size       = 3
      desired_size   = 3
      max_size       = 9
    }
    gpu_a = {
      instance_types = ["g5.12xlarge"]
      min_size       = 1
      desired_size   = 1
      max_size       = 4
      disk_size_gb   = 500
    }
    gpu_b = {
      instance_types = ["g5.2xlarge"]
      min_size       = 1
      desired_size   = 1
      max_size       = 2
      disk_size_gb   = 200
    }
    batch = {
      instance_types = ["m6i.2xlarge", "m6a.2xlarge", "m5.2xlarge"]
      min_size       = 0
      desired_size   = 0
      max_size       = 10
    }
  }

  validation {
    condition     = alltrue([for k in keys(var.node_groups) : contains(["cpu", "gpu_a", "gpu_b", "batch"], k)])
    error_message = "node_groups keys must be a subset of cpu, gpu_a, gpu_b, batch."
  }
}

variable "egress_node_instance_type" {
  description = "Instance type for the small egress-proxy node group (only when enable_egress_nat = true)."
  type        = string
  default     = "t3.medium"
}

# ---------------------------------------------------------------------------- data

variable "db_instance_class" {
  description = "RDS instance class for the state database (registry, run state, budgets)."
  type        = string
  default     = "db.r6g.large"
}

variable "db_allocated_storage_gb" {
  description = "Initial storage; autoscaling up to 4x."
  type        = number
  default     = 100
}

variable "db_engine_version" {
  description = "PostgreSQL major (or major.minor) version. pgvector is installed per database with CREATE EXTENSION vector."
  type        = string
  default     = "16"
}

variable "db_multi_az" {
  description = "Standby in a second AZ. true for stage 2/3; false is acceptable for a stage 1 pilot."
  type        = bool
  default     = true
}

variable "db_backup_retention_days" {
  description = "Automated backup retention for RDS."
  type        = number
  default     = 14
}

variable "deletion_protection" {
  description = "Protect the database and Redis from accidental deletion. Turn off only to tear down a pilot."
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "ElastiCache node type (limits, semantic cache, queues)."
  type        = string
  default     = "cache.r7g.large"
}

variable "redis_num_nodes" {
  description = "Nodes in the replication group (>= 2 enables automatic failover and multi-AZ)."
  type        = number
  default     = 2
}

variable "backup_retention_days" {
  description = "Days kept in the backups bucket before expiry (index snapshots, DB dumps, config)."
  type        = number
  default     = 90
}

# ---------------------------------------------------------------------------- operations

variable "log_retention_days" {
  description = "CloudWatch retention for service and control-plane logs."
  type        = number
  default     = 30
}

variable "audit_log_retention_days" {
  description = "CloudWatch retention for the audit log stream (set to the client's retention rule)."
  type        = number
  default     = 400
}

variable "ecr_repositories" {
  description = "ECR repositories (immutable tags, scanned on push)."
  type        = list(string)
  default     = ["aiready", "aiready-legacy-app", "egress-proxy", "vllm", "text-embeddings-inference"]
}

variable "kubernetes_namespace" {
  description = "Namespace the Helm chart is installed into (used in IRSA trust policies)."
  type        = string
  default     = "aiready"
}

variable "external_secrets_namespace" {
  description = "Namespace of the External Secrets Operator."
  type        = string
  default     = "external-secrets"
}

variable "external_secrets_service_account" {
  description = "Service account of the External Secrets Operator."
  type        = string
  default     = "external-secrets"
}

variable "data_reader_service_accounts" {
  description = "Service accounts (in kubernetes_namespace) that read documents and model weights from S3."
  type        = list(string)
  default     = ["ingestion", "ingestion-worker", "model-loader"]
}

variable "backup_service_accounts" {
  description = "Service accounts (in kubernetes_namespace) that write snapshots to the backups bucket."
  type        = list(string)
  default     = ["backup"]
}
