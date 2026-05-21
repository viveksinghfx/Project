terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "vivek-tfstate-prod"
    key    = "eks/prod/terraform.tfstate"
    region = "ap-south-1"
  }
}

provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source          = "../../modules/vpc"
  vpc_name        = "vivek-eks-prod"
  vpc_cidr        = "10.0.0.0/16"
  azs             = ["ap-south-1a", "ap-south-1b"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.10.0/24", "10.0.11.0/24"]
  cluster_name    = "vivek-eks-prod"
  tags            = local.common_tags
}

module "eks" {
  source              = "../../modules/eks"
  cluster_name        = "vivek-eks-prod"
  kubernetes_version  = "1.30"
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.public_subnet_ids
  private_subnet_ids  = module.vpc.private_subnet_ids
  node_instance_types = ["t3.medium"]
  desired_nodes       = 2
  min_nodes           = 1
  max_nodes           = 5
  environment         = "prod"
  tags                = local.common_tags
}

locals {
  common_tags = {
    Project     = "vivek-eks-gitops"
    Environment = "prod"
    ManagedBy   = "terraform"
    Owner       = "vivek"
  }
}

variable "aws_region" { default = "ap-south-1" }
