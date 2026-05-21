# GitHub Actions OIDC → AWS Integration

Keyless AWS authentication for GitHub Actions using OIDC federation.
No long-lived IAM access keys stored in GitHub Secrets — GitHub mints a short-lived
JWT that AWS STS exchanges for temporary credentials scoped to a specific IAM role.

## Architecture

```
  GitHub Actions Runner
         │
         │  1. Request OIDC token
         ▼
  ┌──────────────┐
  │  GitHub OIDC │
  │  Provider    │
  └──────┬───────┘
         │  2. JWT token (signed by GitHub)
         ▼
  ┌──────────────────────────────────────┐
  │              AWS STS                 │
  │  AssumeRoleWithWebIdentity           │
  │                                      │
  │  Validates:                          │
  │  · aud = sts.amazonaws.com           │
  │  · sub = repo:org/repo:ref:refs/...  │
  │  · iss = token.actions.github.com    │
  └──────────────┬───────────────────────┘
                 │  3. Temporary credentials (15 min)
                 ▼
  ┌──────────────────────────────────────┐
  │           IAM Role                   │
  │  (scoped policy: ECR push, S3, etc.) │
  └──────────────┬───────────────────────┘
                 │  4. AWS API calls
                 ▼
         ECR / S3 / EKS / etc.

  Zero long-lived credentials stored in GitHub Secrets.
```

## Stack

- **CI**: GitHub Actions
- **Auth**: AWS OIDC Identity Provider + IAM Role with trust policy
- **Cloud**: AWS (IAM, STS)

## Setup

```bash
# 1. Create OIDC provider in AWS
aws iam create-open-id-connect-provider   --url https://token.actions.githubusercontent.com   --client-id-list sts.amazonaws.com   --thumbprint-list <thumbprint>

# 2. Create IAM role with trust policy (see terraform/ dir)
cd terraform && terraform apply

# 3. Add to your workflow
# permissions:
#   id-token: write
#   contents: read
#
# - uses: aws-actions/configure-aws-credentials@v4
#   with:
#     role-to-assume: arn:aws:iam::ACCOUNT:role/github-actions-role
#     aws-region: ap-south-1
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
