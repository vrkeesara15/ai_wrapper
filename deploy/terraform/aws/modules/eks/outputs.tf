output "cluster_name" {
  description = "Cluster name."
  value       = aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  description = "Private API endpoint."
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_ca" {
  description = "Cluster CA bundle (base64)."
  value       = aws_eks_cluster.this.certificate_authority[0].data
}

output "node_security_group_id" {
  description = "EKS-managed cluster security group attached to nodes (used to allow DB/Redis access)."
  value       = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
}

output "oidc_provider_arn" {
  description = "IRSA OIDC provider ARN."
  value       = aws_iam_openid_connect_provider.this.arn
}

output "oidc_provider_url" {
  description = "IRSA OIDC issuer URL."
  value       = aws_iam_openid_connect_provider.this.url
}

output "node_role_arn" {
  description = "Node IAM role."
  value       = aws_iam_role.node.arn
}

output "node_group_labels" {
  description = "Labels per node group."
  value       = { for k, g in aws_eks_node_group.this : k => g.labels }
}
