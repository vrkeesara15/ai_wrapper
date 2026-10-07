# AI-Ready cloud topology on AWS: one region, three AZs, private network with no public addresses,
# EKS with CPU / GPU A (LLM) / GPU B (embedding, rerank, small models) / batch (spot) pools, and the
# managed state services (RDS Postgres, ElastiCache Redis, S3) encrypted with client-held KMS keys.

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  azs = length(var.availability_zones) == 3 ? var.availability_zones : slice(data.aws_availability_zones.available.names, 0, 3)

  cluster_name = var.name

  tags = merge({
    "app.kubernetes.io/part-of" = "aiready"
    "aiready.io/stage"          = var.stage
    "managed-by"                = "terraform"
  }, var.tags)
}

module "kms" {
  source = "./modules/kms"

  name       = var.name
  region     = var.region
  account_id = data.aws_caller_identity.current.account_id
  tags       = local.tags
}

module "observability" {
  source = "./modules/observability"

  name                     = var.name
  cluster_name             = local.cluster_name
  logs_kms_key_arn         = module.kms.logs_key_arn
  log_retention_days       = var.log_retention_days
  audit_log_retention_days = var.audit_log_retention_days
  tags                     = local.tags
}

module "network" {
  source = "./modules/network"

  name                          = var.name
  region                        = var.region
  cluster_name                  = local.cluster_name
  vpc_cidr                      = var.vpc_cidr
  azs                           = local.azs
  enable_egress_nat             = var.enable_egress_nat
  enable_private_model_endpoint = var.enable_private_model_endpoint
  flow_log_group_arn            = module.observability.flow_log_group_arn
  tags                          = local.tags
}

module "secrets" {
  source = "./modules/secrets"

  name             = var.name
  data_kms_key_arn = module.kms.data_key_arn
  tags             = local.tags
}

module "ecr" {
  source = "./modules/ecr"

  name             = var.name
  repositories     = var.ecr_repositories
  data_kms_key_arn = module.kms.data_key_arn
  tags             = local.tags
}

module "eks" {
  source = "./modules/eks"

  name                      = local.cluster_name
  kubernetes_version        = var.kubernetes_version
  vpc_id                    = module.network.vpc_id
  vpc_cidr                  = var.vpc_cidr
  private_subnet_ids        = module.network.private_subnet_ids
  egress_subnet_ids         = module.network.egress_subnet_ids
  enable_egress_node_group  = var.enable_egress_nat
  egress_node_instance_type = var.egress_node_instance_type
  app_network_cidrs         = var.app_network_cidrs
  eks_kms_key_arn           = module.kms.eks_key_arn
  node_groups               = var.node_groups
  admin_principal_arns      = var.admin_principal_arns
  tags                      = local.tags

  # the control-plane log group is pre-created with KMS + retention; EKS would otherwise create its own
  depends_on = [module.observability]
}

module "data" {
  source = "./modules/data"

  name                     = var.name
  account_id               = data.aws_caller_identity.current.account_id
  vpc_id                   = module.network.vpc_id
  private_subnet_ids       = module.network.private_subnet_ids
  client_security_group_id = module.eks.node_security_group_id
  data_kms_key_arn         = module.kms.data_key_arn
  db_instance_class        = var.db_instance_class
  db_allocated_storage_gb  = var.db_allocated_storage_gb
  db_engine_version        = var.db_engine_version
  db_multi_az              = var.db_multi_az
  db_backup_retention_days = var.db_backup_retention_days
  db_password              = module.secrets.db_password
  redis_node_type          = var.redis_node_type
  redis_num_nodes          = var.redis_num_nodes
  redis_auth_token         = module.secrets.redis_auth_token
  backup_retention_days    = var.backup_retention_days
  deletion_protection      = var.deletion_protection
  tags                     = local.tags
}

module "irsa" {
  source = "./modules/irsa"

  name                             = var.name
  oidc_provider_arn                = module.eks.oidc_provider_arn
  oidc_provider_url                = module.eks.oidc_provider_url
  namespace                        = var.kubernetes_namespace
  external_secrets_namespace       = var.external_secrets_namespace
  external_secrets_service_account = var.external_secrets_service_account
  data_reader_service_accounts     = var.data_reader_service_accounts
  backup_service_accounts          = var.backup_service_accounts
  secret_arns                      = module.secrets.secret_arns
  data_kms_key_arn                 = module.kms.data_key_arn
  documents_bucket_arn             = module.data.bucket_arns["documents"]
  weights_bucket_arn               = module.data.bucket_arns["weights"]
  backups_bucket_arn               = module.data.bucket_arns["backups"]
  enable_private_model_endpoint    = var.enable_private_model_endpoint
  region                           = var.region
  tags                             = local.tags
}
