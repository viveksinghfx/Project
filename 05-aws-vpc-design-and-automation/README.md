# AWS VPC Design & Automation

3-tier VPC design for production workloads — public, private, and database subnets
across two Availability Zones. Fully automated with Terraform.

## Architecture

```
┌─────────────────────────── AWS Region (ap-south-1) ─────────────────────────────┐
│                                                                                   │
│   VPC  10.0.0.0/16                                                                │
│                                                                                   │
│   ┌──────────────────────────┐    ┌──────────────────────────┐                   │
│   │   Availability Zone A    │    │   Availability Zone B    │                   │
│   │                          │    │                          │                   │
│   │  ┌────────────────────┐  │    │  ┌────────────────────┐  │                   │
│   │  │  Public Subnet     │  │    │  │  Public Subnet     │  │                   │
│   │  │  10.0.1.0/24       │  │    │  │  10.0.2.0/24       │  │                   │
│   │  │  [ ALB / NAT GW ]  │  │    │  │  [ ALB / NAT GW ]  │  │                   │
│   │  └────────────────────┘  │    │  └────────────────────┘  │                   │
│   │                          │    │                          │                   │
│   │  ┌────────────────────┐  │    │  ┌────────────────────┐  │                   │
│   │  │  Private Subnet    │  │    │  │  Private Subnet    │  │                   │
│   │  │  10.0.10.0/24      │  │    │  │  10.0.11.0/24      │  │                   │
│   │  │  [ App EC2 / EKS ] │  │    │  │  [ App EC2 / EKS ] │  │                   │
│   │  └────────────────────┘  │    │  └────────────────────┘  │                   │
│   │                          │    │                          │                   │
│   │  ┌────────────────────┐  │    │  ┌────────────────────┐  │                   │
│   │  │  DB Subnet         │  │    │  │  DB Subnet         │  │                   │
│   │  │  10.0.20.0/24      │  │    │  │  10.0.21.0/24      │  │                   │
│   │  │  [ RDS / ElastiC ] │  │    │  │  [ RDS Standby ]   │  │                   │
│   │  └────────────────────┘  │    │  └────────────────────┘  │                   │
│   └──────────────────────────┘    └──────────────────────────┘                   │
│                                                                                   │
│   Internet Gateway ──► Public RT    NAT Gateway ──► Private RT                   │
│   Terraform automates: VPC · Subnets · IGW · NAT · Route Tables · NACLs · SGs    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Stack

- **Infra**: Terraform (VPC, subnets, IGW, NAT GW, route tables, NACLs, security groups)
- **Cloud**: AWS

## Quick Start

```bash
cd terraform
terraform init
terraform apply -var-file="vars/prod.tfvars"
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
