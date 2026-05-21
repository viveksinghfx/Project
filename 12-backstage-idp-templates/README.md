# Internal Developer Platform — Backstage Software Templates

## Architecture

```
  Engineer opens Backstage UI
           │
           │  fills template form (service name, owner, env)
           ▼
  ┌──────────────────────────────────────────────────────┐
  │              Backstage Scaffolder                     │
  │                                                       │
  │  Template: new-microservice                           │
  │  Steps:                                               │
  │    1. fetch:template  ──► renders skeleton files      │
  │    2. publish:github  ──► creates GitHub repo         │
  │    3. catalog:register──► adds to Backstage catalog   │
  │    4. argocd:create   ──► creates ArgoCD Application  │
  └───────────┬───────────────────────────────────────────┘
              │
    ┌─────────┼──────────────┬─────────────────┐
    ▼         ▼              ▼                  ▼
  GitHub    Backstage     ArgoCD            IAM / ECR
  Repo      Catalog       Application       (Terraform)
  (+ CI/CD  (searchable,  (auto-syncs       provisioned
  workflow) ownership     k8s manifests     via GitHub
  baked in) tracked)      from repo)        Actions)

  Result: engineer gets a fully wired service in ~2 minutes
  with zero manual infra work.
```



Production IDP scaffolding using Backstage Software Templates. Engineers run a single
template to get a GitHub repo, ECR registry, EKS namespace, and GitHub Actions pipeline
— all wired together without touching Terraform or Kubernetes directly.

## What it provisions (per service)

- GitHub repository with CI/CD pre-configured
- AWS ECR repository (via Terraform)
- Kubernetes namespace + RBAC on EKS
- ArgoCD Application pointing at the new repo
- Prometheus ServiceMonitor for auto-scraping

## Templates

| Template | What it creates |
|---|---|
| `new-service` | Full microservice scaffold (repo + infra + CD) |
| `add-monitoring` | Grafana dashboard + alert rules for existing service |
| `eks-namespace` | Namespace, LimitRange, NetworkPolicy, RBAC |

## Running Locally

```bash
# Start Backstage dev server
cd backstage/
yarn install
yarn dev
```

## Template Usage via CLI (automation)

```bash
# Using Backstage scaffolder REST API
curl -X POST https://backstage.viveksingh.tech/api/scaffolder/v2/tasks \
  -H "Authorization: Bearer $BACKSTAGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d @scaffold-request.json
```
