# Python Platform Automation Scripts

## Architecture

```
  GitHub Actions (cron: daily 08:00 IST)
           │
           ├──── Job: aws-stale-resources
           │          │
           │          │  OIDC ──► IAM Role
           │          ▼
           │     aws_stale_resources.py
           │          │  boto3 ──► EC2 / EBS / EIP APIs
           │          │  finds untagged / idle resources
           │          └──► JSON report + Slack alert
           │
           ├──── Job: k8s-health
           │          │
           │          │  OIDC ──► IAM Role ──► EKS kubeconfig
           │          ▼
           │     k8s_health_report.py
           │          │  kubernetes client ──► all namespaces
           │          │  detects: CrashLoop / OOMKill / Pending
           │          └──► Slack alert with pod details
           │
           └──── Job: ecr-cleanup  (weekly)
                      │
                      ▼
                 ecr_image_cleanup.py
                      │  boto3 ──► ECR
                      │  deletes images > 30 days (keeps last 10)
                      └──► deletion report to stdout / S3
```



Production-grade Python automation for common platform/DevOps tasks:
AWS resource cleanup, Kubernetes health checks, Terraform drift detection,
and incident alert routing — all used as GitHub Actions jobs or cron tasks.

## Scripts

| Script | What it does |
|---|---|
| `aws_stale_resources.py` | Finds and reports untagged/idle AWS resources (EC2, EBS, EIP) |
| `k8s_health_report.py` | Scans all namespaces for crashlooping/OOMKilled pods, posts to Slack |
| `tf_drift_check.py` | Runs `terraform plan` across envs, parses output, alerts on drift |
| `ecr_image_cleanup.py` | Deletes ECR images older than N days, keeps last K tags |

## Usage

```bash
pip install -r requirements.txt

# Stale resource report
python scripts/aws_stale_resources.py --region ap-south-1 --dry-run

# K8s health check (outputs JSON or posts to Slack)
python scripts/k8s_health_report.py --slack-webhook $SLACK_WEBHOOK_URL

# ECR cleanup (keep last 10 tags, dry-run first)
python scripts/ecr_image_cleanup.py --repo backend-app --keep 10 --dry-run
```

## GitHub Actions (scheduled)

```yaml
# Runs every day at 8 AM IST (2:30 AM UTC)
on:
  schedule:
    - cron: '30 2 * * *'
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
