output "vpc_id" {
  description = "VPC id."
  value       = aws_vpc.this.id
}

output "private_subnet_ids" {
  description = "Workload subnets, one per AZ, no route to the internet."
  value       = aws_subnet.private[*].id
}

output "private_subnet_cidrs" {
  description = "Workload subnet CIDRs."
  value       = aws_subnet.private[*].cidr_block
}

output "egress_subnet_ids" {
  description = "Egress-proxy subnets (empty unless enable_egress_nat)."
  value       = aws_subnet.egress[*].id
}

output "s3_endpoint_id" {
  description = "S3 gateway endpoint id (buckets may restrict access to it)."
  value       = aws_vpc_endpoint.s3.id
}

output "endpoint_security_group_id" {
  description = "Security group of the interface endpoints."
  value       = aws_security_group.endpoints.id
}

output "bedrock_endpoint_id" {
  description = "Bedrock runtime endpoint id, or null."
  value       = var.enable_private_model_endpoint ? aws_vpc_endpoint.interface["bedrock-runtime"].id : null
}
