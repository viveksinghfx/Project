# Jenkins HA Setup on AWS

Highly available Jenkins deployment on AWS using Terraform, Ansible, and Packer.
Jenkins home is persisted on EFS so instances can be replaced without data loss.
An ALB and ASG keep Jenkins running across AZs.

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │            AWS Region                │
  Developer               │                                      │
     │   push             │   ┌──────────┐    ┌──────────────┐  │
     └──────────────────► │   │   ALB    │───►│  ASG (2-4)   │  │
                          │   └──────────┘    │  Jenkins EC2 │  │
                          │                   └──────┬───────┘  │
                          │                          │           │
                          │                   ┌──────▼───────┐  │
                          │                   │  EFS Mount   │  │
                          │                   │ (JENKINS_HOME│  │
                          │                   │  persisted)  │  │
                          │                   └──────────────┘  │
                          │                                      │
                          │   IAM Role  ·  Security Groups       │
                          └─────────────────────────────────────┘
```

**Key design decisions:**
- EFS ensures Jenkins home survives instance replacement
- ASG keeps Jenkins HA across AZs with health-check replacement
- ALB terminates HTTPS and routes to healthy instances
- Packer bakes an AMI with Jenkins pre-installed; Ansible handles configuration

## Stack

- **Infra**: Terraform (EC2, ALB, ASG, EFS, IAM, Security Groups)
- **Config Management**: Ansible (Jenkins install, plugin setup)
- **Image Baking**: Packer (pre-baked Jenkins AMI)
- **Cloud**: AWS (EC2, ALB, ASG, EFS, IAM)

## Quick Start

```bash
# 1. Bake AMI
cd packer && packer build jenkins.pkr.hcl

# 2. Provision infrastructure
cd terraform
terraform init
terraform apply -var="ami_id=<packer_output_ami>"

# 3. Configure Jenkins via Ansible
cd ansible
ansible-playbook -i inventory/aws_ec2.yaml playbooks/jenkins.yaml
```

## Project by Vivek Singh

🌐 [viveksingh.tech](https://viveksingh.tech) · [LinkedIn](https://linkedin.com/in/vsdevop) · ✉️ viveksinghfx@gmail.com
