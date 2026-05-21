# Prometheus Observability Stack

Full observability stack using Prometheus, Grafana, Loki, and Alertmanager.
Deployable locally via Docker Compose or on AWS EC2 via Terraform.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Observability Stack                       │
│                                                               │
│   Targets (scrape)                                            │
│   ┌──────────────┐                                            │
│   │  Node Exporter│──┐                                        │
│   └──────────────┘  │    ┌──────────────┐                    │
│   ┌──────────────┐  ├───►│  Prometheus  │                    │
│   │  App Metrics │──┘    │  (TSDB)      │                    │
│   └──────────────┘       └──────┬───────┘                    │
│                                 │  query / alert              │
│                    ┌────────────┼────────────┐               │
│                    │            │             │               │
│                    ▼            ▼             ▼               │
│            ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│            │  Grafana  │  │  Loki    │  │ Alertmanager │     │
│            │(dashboards│  │  (logs)  │  │              │     │
│            └──────────┘  └──────────┘  └──────┬───────┘     │
│                                                │              │
│                                         ┌──────▼───────┐     │
│                                         │  Slack / PD  │     │
│                                         └──────────────┘     │
│                                                               │
│   Deployed via: Docker Compose (local) · Terraform (AWS EC2) │
└──────────────────────────────────────────────────────────────┘
```

## Stack

- **Metrics**: Prometheus + Node Exporter
- **Logs**: Loki + Promtail
- **Dashboards**: Grafana
- **Alerting**: Alertmanager → Slack
- **Local Deploy**: Docker Compose
- **Cloud Deploy**: Terraform (AWS EC2, Security Groups)

## Quick Start

```bash
# Local (Docker Compose)
docker compose up -d
# Grafana: http://localhost:3000  (admin/admin)
# Prometheus: http://localhost:9090

# AWS deploy
cd terraform-aws
terraform init && terraform apply
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
