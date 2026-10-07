variable "name" {
  description = "Cluster name."
  type        = string
}

variable "kubernetes_version" {
  description = "EKS version."
  type        = string
}

variable "vpc_id" {
  description = "VPC id."
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR (API access from inside the VPC)."
  type        = string
}

variable "private_subnet_ids" {
  description = "Workload subnets across three AZs."
  type        = list(string)
}

variable "egress_subnet_ids" {
  description = "Egress-proxy subnets (used only by the egress node group)."
  type        = list(string)
  default     = []
}

variable "enable_egress_node_group" {
  description = "Create the small tainted node group for the egress proxy in the egress subnets."
  type        = bool
  default     = false
}

variable "egress_node_instance_type" {
  description = "Instance type of the egress node group."
  type        = string
  default     = "t3.medium"
}

variable "app_network_cidrs" {
  description = "Existing application CIDRs allowed to reach the internal ingress on 443."
  type        = list(string)
  default     = []
}

variable "eks_kms_key_arn" {
  description = "KMS key for Secrets envelope encryption."
  type        = string
}

variable "node_groups" {
  description = "Node group sizes keyed by cpu, gpu_a, gpu_b, batch."
  type = map(object({
    instance_types = list(string)
    min_size       = number
    desired_size   = number
    max_size       = number
    disk_size_gb   = optional(number, 100)
  }))
}

variable "admin_principal_arns" {
  description = "IAM principals granted cluster admin."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags."
  type        = map(string)
  default     = {}
}
