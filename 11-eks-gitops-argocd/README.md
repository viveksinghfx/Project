# EKS GitOps Platform with ArgoCD

Production-grade GitOps CD platform on AWS EKS. GitHub Actions builds and pushes images to ECR; ArgoCD syncs Kustomize overlays to the cluster. Infrastructure is fully managed via Terraform.

## Architecture

```
  Developer
     │  git push
     ▼
  GitHub Repo
     │
     ├──► .github/workflows/ci.yaml
     │         │
     │         │  1. OIDC auth (no keys)
     │         ▼
     │    AWS STS ──► IAM Role ──► ECR
     │         │
     │         │  2. docker build + push
     │         │  3. kustomize edit set image (new SHA tag)
     │         │  4. git push (updates k8s/overlays/prod)
     │
     │  (GitOps loop)
     ▼
  ArgoCD (running in EKS)
     │  polls repo every 3 min / webhook
     │
     ├── detects image tag changed in kustomization.yaml
     │
     ▼
  EKS Cluster
  ┌─────────────────────────────────────────────┐
  │  Namespace: backend                          │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
  │  │  Pod     │  │  Pod     │  │  Pod     │  │
  │  │ (new SHA)│  │ (new SHA)│  │ (new SHA)│  │
  │  └──────────┘  └──────────┘  └──────────┘  │
  └─────────────────────────────────────────────┘
     │
     ▼
  Prometheus scrapes ──► Grafana dashboards
  Loki collects logs  ──► Alertmanager ──► Slack

  Infra provisioned by Terraform (EKS · VPC · IAM · ECR)
```

## Stack

- **Infra**: Terraform — EKS, VPC, IAM, ECR
- **Orchestration**: Kubernetes (EKS) + Helm
- **GitOps CD**: ArgoCD (App of Apps pattern)
- **CI**: GitHub Actions (OIDC → AWS, no long-lived keys)
- **Observability**: Prometheus + Grafana + Loki

## Quick Start

```bash
# 1. Provision cluster
cd terraform/envs/prod
terraform init && terraform apply -var-file="prod.tfvars"

# 2. Update kubeconfig
aws eks update-kubeconfig --name vivek-eks-prod --region ap-south-1

# 3. Bootstrap ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl apply -f argocd/app-of-apps.yaml
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
