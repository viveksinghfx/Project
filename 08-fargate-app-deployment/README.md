# Fargate App Deployment on EKS

Deploys a containerised application to EKS Fargate using Helm. Infrastructure (VPC) is provisioned with Terraform. No EC2 worker nodes to manage — pods run serverless on Fargate.

## Architecture

```
┌────────────────────────────── AWS EKS (Fargate) ─────────────────────────────────┐
│                                                                                    │
│   kubectl / Helm                                                                   │
│        │                                                                           │
│        ▼                                                                           │
│   ┌─────────────┐         ┌────────────────────────────────────────────┐          │
│   │ Fargate     │         │              EKS Control Plane              │          │
│   │ Profile     │         │  (AWS-managed, no master nodes to operate)  │          │
│   └──────┬──────┘         └────────────────────────────────────────────┘          │
│          │                                                                         │
│          │  schedules pods onto                                                    │
│          ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐         │
│   │                    Fargate Nodes (serverless)                        │         │
│   │                                                                      │         │
│   │   ┌────────────────────┐       ┌────────────────────┐               │         │
│   │   │   Pod: game-2048   │       │   Pod: game-2048   │               │         │
│   │   │   (Helm chart)     │       │   (Helm chart)     │               │         │
│   │   └────────────────────┘       └────────────────────┘               │         │
│   └─────────────────────────────────────────────────────────────────────┘         │
│                          │                                                         │
│                   ┌──────▼──────┐                                                 │
│                   │  AWS ALB    │  ◄── aws-load-balancer-controller                │
│                   │  (Ingress)  │                                                  │
│                   └──────┬──────┘                                                 │
│                          │                                                         │
│                        Internet                                                   │
│                                                                                    │
│   Infra: Terraform (VPC)  ·  Helm (app)  ·  kubectl (Fargate profile + Ingress)   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## Stack

- **Infra**: Terraform — VPC, subnets, IGW, NAT
- **Orchestration**: EKS Fargate (serverless pods)
- **App Packaging**: Helm chart
- **Ingress**: AWS Load Balancer Controller

## Quick Start

```bash
# 1. Provision VPC
cd tf-vpc && terraform init && terraform apply

# 2. Create EKS cluster + Fargate profile
eksctl create cluster -f eks-fargate.yaml

# 3. Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system --set clusterName=vivek-fargate-cluster

# 4. Deploy app
helm install game-2048 ./helm/game-2048
kubectl get ingress -n game-2048
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
