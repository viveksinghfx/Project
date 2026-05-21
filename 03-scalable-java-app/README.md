# Scalable Java Application on AWS

End-to-end deployment of a Java (Spring Boot) application on AWS using Terraform for infrastructure
and Ansible for configuration. ALB routes traffic to an Auto Scaling Group; RDS Multi-AZ handles
the database layer.

## Architecture

```
  Internet
     │
     ▼
┌─────────┐        ┌──────────────────────────────────────────────┐
│  Route53│        │                  AWS VPC                      │
└────┬────┘        │                                               │
     │             │  Public Subnets                               │
     ▼             │  ┌──────────────────────────────────────┐    │
┌─────────┐        │  │           Application ALB            │    │
│   ALB   │◄───────┤  └──────────────────┬───────────────────┘    │
└─────────┘        │                     │                         │
                   │  Private Subnets    │                         │
                   │  ┌──────────────────▼───────────────────┐    │
                   │  │       Auto Scaling Group              │    │
                   │  │   ┌──────────┐   ┌──────────┐        │    │
                   │  │   │ App EC2  │   │ App EC2  │  ...   │    │
                   │  │   └─────┬────┘   └─────┬────┘        │    │
                   │  └─────────┼──────────────┼─────────────┘    │
                   │            │              │                   │
                   │  ┌─────────▼──────────────▼──────────────┐   │
                   │  │              RDS (Multi-AZ)            │   │
                   │  │          PostgreSQL Primary +           │   │
                   │  │             Standby Replica             │   │
                   │  └────────────────────────────────────────┘   │
                   └──────────────────────────────────────────────┘

  CI/CD: GitHub Actions ──► Ansible provision ──► Terraform infra
```

## Stack

- **Infra**: Terraform (ALB, ASG, RDS, VPC, IAM, Security Groups)
- **Config Management**: Ansible (app install, environment config)
- **Cloud**: AWS (EC2, ALB, ASG, RDS Multi-AZ, S3)

## Quick Start

```bash
# 1. Provision infrastructure
cd terraform
terraform init
terraform apply -var-file="envs/prod.tfvars"

# 2. Deploy application
cd ansible
ansible-playbook -i inventory/aws_ec2.yaml playbooks/deploy-app.yaml

# 3. Verify
curl http://<alb-dns>/actuator/health
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
