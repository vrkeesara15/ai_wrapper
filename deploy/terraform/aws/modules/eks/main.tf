# EKS with a private-only API endpoint, KMS envelope encryption of Secrets, control-plane logs, IRSA,
# and four managed node groups whose labels and taints are fixed here so they match the Helm chart:
#
#   pool    label aiready.io/pool   taint                              runs
#   cpu     cpu                     -                                  gateway, orchestration, search, MCP, guardrails
#   gpu_a   gpu-a                   nvidia.com/gpu=true:NoSchedule     chat model pool (LLM serving)
#   gpu_b   gpu-b                   nvidia.com/gpu=true:NoSchedule     embedding + rerank + small (router/guard) models
#   batch   batch                   aiready.io/batch=true:NoSchedule   ingestion workers, eval runs (spot)
#   egress  egress                  aiready.io/egress=true:NoSchedule  egress proxy only (optional)

data "aws_partition" "current" {}

locals {
  pool_settings = {
    cpu = {
      label         = "cpu"
      extra_labels  = {}
      ami_type      = "AL2023_x86_64_STANDARD"
      capacity_type = "ON_DEMAND"
      taints        = []
    }
    gpu_a = {
      label         = "gpu-a"
      extra_labels  = { "aiready.io/gpu-pool" = "chat", "nvidia.com/gpu.present" = "true" }
      ami_type      = "AL2023_x86_64_NVIDIA"
      capacity_type = "ON_DEMAND"
      taints        = [{ key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }]
    }
    gpu_b = {
      label         = "gpu-b"
      extra_labels  = { "aiready.io/gpu-pool" = "embed-small", "nvidia.com/gpu.present" = "true" }
      ami_type      = "AL2023_x86_64_NVIDIA"
      capacity_type = "ON_DEMAND"
      taints        = [{ key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }]
    }
    batch = {
      label         = "batch"
      extra_labels  = {}
      ami_type      = "AL2023_x86_64_STANDARD"
      capacity_type = "SPOT"
      taints        = [{ key = "aiready.io/batch", value = "true", effect = "NO_SCHEDULE" }]
    }
  }

  egress_group = var.enable_egress_node_group ? {
    egress = {
      instance_types = [var.egress_node_instance_type]
      min_size       = 2
      desired_size   = 2
      max_size       = 3
      disk_size_gb   = 50
    }
  } : {}

  egress_settings = {
    label         = "egress"
    extra_labels  = {}
    ami_type      = "AL2023_x86_64_STANDARD"
    capacity_type = "ON_DEMAND"
    taints        = [{ key = "aiready.io/egress", value = "true", effect = "NO_SCHEDULE" }]
  }

  groups = merge(var.node_groups, local.egress_group)

  group_settings = { for k, _ in local.groups : k => (k == "egress" ? local.egress_settings : local.pool_settings[k]) }
}

# ---------------------------------------------------------------- IAM

data "aws_iam_policy_document" "cluster_assume" {
  statement {
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cluster" {
  name               = "${var.name}-eks-cluster"
  assume_role_policy = data.aws_iam_policy_document.cluster_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSClusterPolicy"
}

data "aws_iam_policy_document" "cluster_kms" {
  statement {
    actions   = ["kms:Encrypt", "kms:Decrypt", "kms:ListGrants", "kms:DescribeKey", "kms:CreateGrant"]
    resources = [var.eks_kms_key_arn]
  }
}

resource "aws_iam_role_policy" "cluster_kms" {
  name   = "secrets-encryption"
  role   = aws_iam_role.cluster.id
  policy = data.aws_iam_policy_document.cluster_kms.json
}

data "aws_iam_policy_document" "node_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "${var.name}-eks-node"
  assume_role_policy = data.aws_iam_policy_document.node_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "node" {
  for_each = toset([
    "AmazonEKSWorkerNodePolicy",
    "AmazonEKS_CNI_Policy",
    "AmazonEC2ContainerRegistryReadOnly",
    "AmazonSSMManagedInstanceCore",
  ])

  role       = aws_iam_role.node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/${each.key}"
}

# ---------------------------------------------------------------- networking

resource "aws_security_group" "cluster" {
  name        = "${var.name}-eks-cluster"
  description = "Additional EKS control-plane security group"
  vpc_id      = var.vpc_id
  tags        = merge(var.tags, { Name = "${var.name}-eks-cluster" })
}

resource "aws_vpc_security_group_ingress_rule" "cluster_api_vpc" {
  security_group_id = aws_security_group.cluster.id
  description       = "Kubernetes API from inside the VPC (operators via VPN / bastion, CI runners)"
  cidr_ipv4         = var.vpc_cidr
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

# Internal load balancer access from the existing application network (application zone -> platform zone).
resource "aws_vpc_security_group_ingress_rule" "nodes_from_app" {
  for_each = toset(var.app_network_cidrs)

  security_group_id = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
  description       = "Existing application to the internal ingress"
  cidr_ipv4         = each.key
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

# ---------------------------------------------------------------- cluster

resource "aws_eks_cluster" "this" {
  name     = var.name
  version  = var.kubernetes_version
  role_arn = aws_iam_role.cluster.arn

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  vpc_config {
    subnet_ids              = var.private_subnet_ids
    security_group_ids      = [aws_security_group.cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = false
  }

  encryption_config {
    resources = ["secrets"]
    provider {
      key_arn = var.eks_kms_key_arn
    }
  }

  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
  }

  tags = var.tags

  depends_on = [
    aws_iam_role_policy_attachment.cluster,
    aws_iam_role_policy.cluster_kms,
  ]
}

resource "aws_eks_access_entry" "admin" {
  for_each = toset(var.admin_principal_arns)

  cluster_name  = aws_eks_cluster.this.name
  principal_arn = each.key
  tags          = var.tags
}

resource "aws_eks_access_policy_association" "admin" {
  for_each = toset(var.admin_principal_arns)

  cluster_name  = aws_eks_cluster.this.name
  principal_arn = each.key
  policy_arn    = "arn:${data.aws_partition.current.partition}:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

  access_scope {
    type = "cluster"
  }

  depends_on = [aws_eks_access_entry.admin]
}

# ---------------------------------------------------------------- IRSA

data "tls_certificate" "oidc" {
  url = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "this" {
  url             = aws_eks_cluster.this.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.oidc.certificates[0].sha1_fingerprint]
  tags            = var.tags
}

locals {
  oidc_host = replace(aws_iam_openid_connect_provider.this.url, "https://", "")
}

data "aws_iam_policy_document" "ebs_csi_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.this.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = ["system:serviceaccount:kube-system:ebs-csi-controller-sa"]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ebs_csi" {
  name               = "${var.name}-ebs-csi"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

# ---------------------------------------------------------------- node groups

# Launch templates enforce IMDSv2 with a hop limit of 1 (pods cannot reach node credentials) and
# encrypted root volumes.
resource "aws_launch_template" "node" {
  for_each = local.groups

  name_prefix            = "${var.name}-${each.key}-"
  update_default_version = true

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = each.value.disk_size_gb
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags          = merge(var.tags, { Name = "${var.name}-${each.key}", "aiready.io/pool" = local.group_settings[each.key].label })
  }

  tags = var.tags
}

resource "aws_eks_node_group" "this" {
  for_each = local.groups

  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "${var.name}-${replace(each.key, "_", "-")}"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = each.key == "egress" ? var.egress_subnet_ids : var.private_subnet_ids
  instance_types  = each.value.instance_types
  ami_type        = local.group_settings[each.key].ami_type
  capacity_type   = local.group_settings[each.key].capacity_type

  scaling_config {
    min_size     = each.value.min_size
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
  }

  update_config {
    max_unavailable = 1
  }

  launch_template {
    id      = aws_launch_template.node[each.key].id
    version = aws_launch_template.node[each.key].latest_version
  }

  labels = merge({ "aiready.io/pool" = local.group_settings[each.key].label }, local.group_settings[each.key].extra_labels)

  dynamic "taint" {
    for_each = local.group_settings[each.key].taints
    content {
      key    = taint.value.key
      value  = taint.value.value
      effect = taint.value.effect
    }
  }

  tags = merge(var.tags, {
    "k8s.io/cluster-autoscaler/enabled"                             = "true"
    "k8s.io/cluster-autoscaler/${var.name}"                         = "owned"
    "k8s.io/cluster-autoscaler/node-template/label/aiready.io/pool" = local.group_settings[each.key].label
  })

  # The cluster autoscaler / Karpenter owns the live size; Terraform owns the bounds.
  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  depends_on = [aws_iam_role_policy_attachment.node]
}

# ---------------------------------------------------------------- add-ons

resource "aws_eks_addon" "vpc_cni" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "vpc-cni"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = var.tags

  # NetworkPolicy enforcement for the chart's default-deny + per-zone rules.
  configuration_values = jsonencode({ enableNetworkPolicy = "true" })
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "kube-proxy"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = var.tags
}

resource "aws_eks_addon" "coredns" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "coredns"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = var.tags

  depends_on = [aws_eks_node_group.this]
}

resource "aws_eks_addon" "ebs_csi" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "aws-ebs-csi-driver"
  service_account_role_arn    = aws_iam_role.ebs_csi.arn
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = var.tags

  depends_on = [aws_eks_node_group.this]
}
