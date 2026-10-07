# Private network with no public addresses. Workload subnets have no default route at all: images come
# from ECR and AWS APIs are reached through VPC endpoints. When enable_egress_nat is true, a separate
# egress subnet tier (one per AZ) gets a NAT gateway; only the egress proxy runs there, so the proxy's
# allow-list is the single route out (the "closed by default" outside-model route).

locals {
  private_cidrs = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i)]      # /20 each for pods (VPC CNI)
  egress_cidrs  = [for i in range(3) : cidrsubnet(var.vpc_cidr, 8, 48 + i)] # /24 each, egress proxy only
  public_cidrs  = [for i in range(3) : cidrsubnet(var.vpc_cidr, 8, 56 + i)] # /24 each, NAT gateways only

  interface_endpoints = toset(concat(
    ["ecr.api", "ecr.dkr", "sts", "logs", "secretsmanager", "kms", "ec2", "elasticloadbalancing", "autoscaling"],
    var.enable_private_model_endpoint ? ["bedrock-runtime"] : [],
  ))
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(var.tags, { Name = var.name })
}

# Lock down the default security group so nothing uses it implicitly.
resource "aws_default_security_group" "this" {
  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name}-default-closed" })
}

# ---------------------------------------------------------------- workload subnets (no route out)

resource "aws_subnet" "private" {
  count = 3

  vpc_id                  = aws_vpc.this.id
  cidr_block              = local.private_cidrs[count.index]
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = false

  tags = merge(var.tags, {
    Name                                        = "${var.name}-private-${var.azs[count.index]}"
    "aiready.io/tier"                           = "private"
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  })
}

resource "aws_route_table" "private" {
  count = 3

  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name}-private-${var.azs[count.index]}" })
}

resource "aws_route_table_association" "private" {
  count = 3

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# ---------------------------------------------------------------- optional egress tier

resource "aws_internet_gateway" "this" {
  count = var.enable_egress_nat ? 1 : 0

  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name}-igw" })
}

# Public subnets hold only the NAT gateways; no instance gets a public address.
resource "aws_subnet" "public_nat" {
  count = var.enable_egress_nat ? 3 : 0

  vpc_id                  = aws_vpc.this.id
  cidr_block              = local.public_cidrs[count.index]
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = false
  tags                    = merge(var.tags, { Name = "${var.name}-nat-${var.azs[count.index]}", "aiready.io/tier" = "nat" })
}

resource "aws_route_table" "public_nat" {
  count = var.enable_egress_nat ? 1 : 0

  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name}-nat" })
}

resource "aws_route" "public_nat_default" {
  count = var.enable_egress_nat ? 1 : 0

  route_table_id         = aws_route_table.public_nat[0].id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this[0].id
}

resource "aws_route_table_association" "public_nat" {
  count = var.enable_egress_nat ? 3 : 0

  subnet_id      = aws_subnet.public_nat[count.index].id
  route_table_id = aws_route_table.public_nat[0].id
}

resource "aws_eip" "nat" {
  count = var.enable_egress_nat ? 3 : 0

  domain = "vpc"
  tags   = merge(var.tags, { Name = "${var.name}-nat-${var.azs[count.index]}" })
}

resource "aws_nat_gateway" "this" {
  count = var.enable_egress_nat ? 3 : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public_nat[count.index].id
  tags          = merge(var.tags, { Name = "${var.name}-nat-${var.azs[count.index]}" })

  depends_on = [aws_internet_gateway.this]
}

resource "aws_subnet" "egress" {
  count = var.enable_egress_nat ? 3 : 0

  vpc_id                  = aws_vpc.this.id
  cidr_block              = local.egress_cidrs[count.index]
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = false
  tags                    = merge(var.tags, { Name = "${var.name}-egress-${var.azs[count.index]}", "aiready.io/tier" = "egress" })
}

resource "aws_route_table" "egress" {
  count = var.enable_egress_nat ? 3 : 0

  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name}-egress-${var.azs[count.index]}" })
}

resource "aws_route" "egress_default" {
  count = var.enable_egress_nat ? 3 : 0

  route_table_id         = aws_route_table.egress[count.index].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.this[count.index].id
}

resource "aws_route_table_association" "egress" {
  count = var.enable_egress_nat ? 3 : 0

  subnet_id      = aws_subnet.egress[count.index].id
  route_table_id = aws_route_table.egress[count.index].id
}

# ---------------------------------------------------------------- VPC endpoints

resource "aws_security_group" "endpoints" {
  name        = "${var.name}-vpc-endpoints"
  description = "HTTPS from inside the VPC to AWS service endpoints"
  vpc_id      = aws_vpc.this.id
  tags        = merge(var.tags, { Name = "${var.name}-vpc-endpoints" })
}

resource "aws_vpc_security_group_ingress_rule" "endpoints_https" {
  security_group_id = aws_security_group.endpoints.id
  description       = "HTTPS from the VPC"
  cidr_ipv4         = var.vpc_cidr
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

resource "aws_vpc_endpoint" "interface" {
  for_each = local.interface_endpoints

  vpc_id              = aws_vpc.this.id
  service_name        = "com.amazonaws.${var.region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.endpoints.id]
  private_dns_enabled = true
  tags                = merge(var.tags, { Name = "${var.name}-${each.key}" })
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = concat(aws_route_table.private[*].id, aws_route_table.egress[*].id)
  tags              = merge(var.tags, { Name = "${var.name}-s3" })
}

# ---------------------------------------------------------------- flow logs

data "aws_iam_policy_document" "flow_logs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["vpc-flow-logs.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "flow_logs" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogGroups", "logs:DescribeLogStreams"]
    resources = [var.flow_log_group_arn, "${var.flow_log_group_arn}:*"]
  }
}

resource "aws_iam_role" "flow_logs" {
  name               = "${var.name}-vpc-flow-logs"
  assume_role_policy = data.aws_iam_policy_document.flow_logs_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy" "flow_logs" {
  name   = "write-flow-logs"
  role   = aws_iam_role.flow_logs.id
  policy = data.aws_iam_policy_document.flow_logs.json
}

resource "aws_flow_log" "this" {
  vpc_id          = aws_vpc.this.id
  traffic_type    = "ALL"
  log_destination = var.flow_log_group_arn
  iam_role_arn    = aws_iam_role.flow_logs.arn
  tags            = merge(var.tags, { Name = "${var.name}-flow-logs" })
}
