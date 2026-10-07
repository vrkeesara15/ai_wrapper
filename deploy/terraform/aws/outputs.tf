output "vpc_id" {
  description = "Platform VPC."
  value       = module.network.vpc_id
}

output "private_subnet_ids" {
  description = "Workload subnets (no route to the internet)."
  value       = module.network.private_subnet_ids
}

output "egress_subnet_ids" {
  description = "Subnets used only by the egress proxy (empty when enable_egress_nat = false)."
  value       = module.network.egress_subnet_ids
}

output "cluster_name" {
  description = "EKS cluster name (aws eks update-kubeconfig --name ...)."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Private API endpoint of the EKS control plane (reachable only from inside the VPC / VPN)."
  value       = module.eks.cluster_endpoint
}

output "oidc_provider_arn" {
  description = "IRSA OIDC provider."
  value       = module.eks.oidc_provider_arn
}

output "node_group_labels" {
  description = "Node labels per pool; the Helm chart nodeSelectors use aiready.io/pool."
  value       = module.eks.node_group_labels
}

output "kms_key_arns" {
  description = "Customer-managed keys (held by the client)."
  value = {
    eks  = module.kms.eks_key_arn
    data = module.kms.data_key_arn
    logs = module.kms.logs_key_arn
  }
}

output "ecr_repository_urls" {
  description = "Container registry URLs; images are pushed here (signed) by the release pipeline."
  value       = module.ecr.repository_urls
}

output "db_endpoint" {
  description = "State database endpoint (host:port). Credentials are in Secrets Manager."
  value       = module.data.db_endpoint
}

output "redis_endpoint" {
  description = "Primary endpoint of the Redis replication group (TLS, auth token in Secrets Manager)."
  value       = module.data.redis_primary_endpoint
}

output "bucket_names" {
  description = "S3 buckets: documents, weights, backups."
  value       = module.data.bucket_names
}

output "secret_arns" {
  description = "Secrets Manager ARNs keyed by the secret file name expected under /run/secrets."
  value       = module.secrets.secret_arns
}

output "irsa_role_arns" {
  description = "Annotate service accounts with eks.amazonaws.com/role-arn using these."
  value       = module.irsa.role_arns
}

output "bedrock_endpoint_id" {
  description = "Bedrock runtime interface endpoint (null unless enable_private_model_endpoint = true)."
  value       = module.network.bedrock_endpoint_id
}

output "helm_values_hint" {
  description = "Values to feed into deploy/helm/aiready (values-production.yaml) for this account."
  value = {
    image_registry      = split("/", module.ecr.repository_urls["aiready"])[0]
    external_secrets    = { enabled = true, role_arn = module.irsa.role_arns["external_secrets"] }
    postgres_host       = module.data.db_address
    redis_host          = module.data.redis_primary_endpoint
    weights_bucket      = module.data.bucket_names["weights"]
    documents_bucket    = module.data.bucket_names["documents"]
    backups_bucket      = module.data.bucket_names["backups"]
    application_log_grp = module.observability.application_log_group_name
  }
}
