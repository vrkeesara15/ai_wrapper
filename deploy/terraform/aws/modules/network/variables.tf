variable "name" {
  description = "Name prefix."
  type        = string
}

variable "region" {
  description = "AWS region (endpoint service names)."
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name (subnet discovery tags for internal load balancers)."
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR; /16 recommended (three /20 workload subnets plus small egress/NAT tiers)."
  type        = string
}

variable "azs" {
  description = "Three availability zones."
  type        = list(string)

  validation {
    condition     = length(var.azs) == 3
    error_message = "Exactly three availability zones are required."
  }
}

variable "enable_egress_nat" {
  description = "Create egress subnets with NAT gateways for the egress proxy only."
  type        = bool
  default     = false
}

variable "enable_private_model_endpoint" {
  description = "Create the bedrock-runtime interface endpoint."
  type        = bool
  default     = false
}

variable "flow_log_group_arn" {
  description = "CloudWatch log group for VPC flow logs."
  type        = string
}

variable "tags" {
  description = "Tags."
  type        = map(string)
  default     = {}
}
