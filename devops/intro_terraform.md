# Terraform – Complete Guide (2026 Edition)

Terraform (and its open-source fork **OpenTofu**) is the leading **Infrastructure as Code (IaC)** tool. It lets you define, provision, and manage cloud infrastructure using declarative configuration files — AWS, GCP, Azure, Kubernetes, and 3,000+ other providers.

---

## Table of Contents
1. [What is Terraform?](#what-is-terraform)
2. [Core Concepts](#core-concepts)
3. [Installation](#installation)
4. [HCL Language Basics](#hcl-language-basics)
5. [Resources & Data Sources](#resources--data-sources)
6. [Variables & Outputs](#variables--outputs)
7. [State Management](#state-management)
8. [Modules](#modules)
9. [Workspace & Environments](#workspaces--environments)
10. [Terraform Workflow](#terraform-workflow)
11. [Real-World Examples](#real-world-examples)
12. [Testing Terraform](#testing-terraform)
13. [Terraform in 2026 (OpenTofu)](#terraform-in-2026-opentofu)
14. [Cheat Sheet](#cheat-sheet)

---

## What is Terraform?

Without IaC, you click around cloud consoles or write imperative scripts. This creates "snowflake servers" — unique, undocumented, unrepeatable infrastructure.

**Terraform** lets you describe infrastructure in code:

```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
}
```

Run `terraform apply` and Terraform provisions the EC2 instance. The code IS the documentation.

### Why Terraform?

- **Declarative** — describe *what* you want, not *how* to create it
- **Idempotent** — running `apply` twice is safe
- **Multi-cloud** — AWS, GCP, Azure, Kubernetes, Datadog, GitHub, etc.
- **State tracking** — knows what exists vs. what should exist
- **Plan before apply** — see changes before making them
- **Modules** — reusable, composable infrastructure components

### Terraform vs Alternatives

| Tool | Approach | Best For |
|------|----------|----------|
| Terraform / OpenTofu | Declarative HCL | Multi-cloud, all infra types |
| Pulumi | Imperative (Python/TS/Go) | Developers who prefer real code |
| AWS CDK | Imperative (TS/Python) | AWS-native, complex logic |
| CloudFormation | Declarative JSON/YAML | AWS-only, native integration |
| Ansible | Imperative (YAML) | Configuration management, not IaC |

---

## Core Concepts

```
Configuration Files (.tf)
         ↓
    terraform init    (download providers + modules)
         ↓
    terraform plan    (show what will change)
         ↓
    terraform apply   (make the changes)
         ↓
       State File     (terraform.tfstate — records real-world resources)
```

| Concept | Description |
|---------|-------------|
| **Provider** | Plugin to interact with a cloud/service API (aws, google, azurerm) |
| **Resource** | A piece of infrastructure (EC2 instance, S3 bucket, DNS record) |
| **Data Source** | Read-only lookup of existing resources |
| **Variable** | Input parameter to make configs reusable |
| **Output** | Export values from your config (IP addresses, ARNs, etc.) |
| **State** | JSON file recording current real-world infrastructure |
| **Module** | Reusable group of resources (like a function in code) |
| **Workspace** | Multiple independent state files from the same config |

---

## Installation

### Terraform

```bash
# macOS
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Linux
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

### OpenTofu (open-source fork)

```bash
# macOS
brew install opentofu

# Linux
curl --proto '=https' --tlsv1.2 -fsSL https://get.opentofu.org/install-opentofu.sh | sh

# Verify
tofu version   # or: terraform version
```

### tfenv (version manager)

```bash
brew install tfenv
tfenv install 1.9.0
tfenv use 1.9.0
```

---

## HCL Language Basics

Terraform uses **HCL (HashiCorp Configuration Language)** — readable, declarative, JSON-compatible.

### Types

```hcl
# Primitives
name    = "my-app"           # string
count   = 3                  # number
enabled = true               # bool

# Collections
tags = {                     # map(string)
  Environment = "production"
  Team        = "platform"
}

cidr_blocks = [              # list(string)
  "10.0.0.0/8",
  "172.16.0.0/12"
]

# Expressions
name    = "app-${var.environment}"           # interpolation
doubled = var.count * 2                      # arithmetic
enabled = var.env == "production" ? true : false  # ternary
```

### Built-in Functions

```hcl
# String
lower("HELLO")         # → "hello"
upper("hello")         # → "HELLO"
join("-", ["a","b"])   # → "a-b"
split(",", "a,b,c")   # → ["a","b","c"]
format("Hello, %s!", var.name)
replace("foo-bar", "-", "_")

# Numeric
min(1, 2, 3)           # → 1
max(1, 2, 3)           # → 3
ceil(1.2)              # → 2
floor(1.8)             # → 1

# Collections
length(var.list)
toset(var.list)         # deduplicate
flatten([[1,2],[3,4]])  # → [1,2,3,4]
merge(map1, map2)
keys(map)
values(map)

# Type conversion
tostring(42)
tonumber("42")
tolist(toset(["a","b","a"]))
```

### Conditional & Loops

```hcl
# Conditional expression
instance_type = var.env == "prod" ? "t3.large" : "t3.micro"

# count (create N resources)
resource "aws_instance" "web" {
  count         = var.instance_count
  instance_type = "t3.micro"
  tags = {
    Name = "web-${count.index}"
  }
}

# for_each (create resources from a map or set)
resource "aws_s3_bucket" "buckets" {
  for_each = toset(["logs", "backups", "artifacts"])
  bucket   = "${var.project}-${each.key}-${var.env}"
}

# for expression (transform collections)
output "bucket_names" {
  value = [for b in aws_s3_bucket.buckets : b.bucket]
}

tag_map = {
  for k, v in var.tags : lower(k) => v
  if v != ""
}

# dynamic blocks (generate repeated nested blocks)
dynamic "ingress" {
  for_each = var.ingress_rules
  content {
    from_port   = ingress.value.from_port
    to_port     = ingress.value.to_port
    protocol    = ingress.value.protocol
    cidr_blocks = ingress.value.cidr_blocks
  }
}
```

---

## Resources & Data Sources

### Resources

```hcl
# Syntax: resource "<PROVIDER_TYPE>" "<LOCAL_NAME>"
resource "aws_s3_bucket" "my_bucket" {
  bucket = "my-unique-bucket-name-2026"

  tags = {
    Name        = "My Bucket"
    Environment = var.environment
  }
}

# Reference another resource with: resource_type.local_name.attribute
resource "aws_s3_bucket_versioning" "my_bucket" {
  bucket = aws_s3_bucket.my_bucket.id   # ← reference

  versioning_configuration {
    status = "Enabled"
  }
}
```

### Data Sources (read existing resources)

```hcl
# Syntax: data "<PROVIDER_TYPE>" "<LOCAL_NAME>"
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]   # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-*-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id   # ← reference data source
  instance_type = "t3.micro"
}
```

---

## Variables & Outputs

### Input Variables

```hcl
# variables.tf
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "development"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "instance_count" {
  description = "Number of instances"
  type        = number
  default     = 1
}

variable "allowed_cidr_blocks" {
  description = "CIDRs allowed to access the load balancer"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
```

### Setting Variable Values

```bash
# 1. Command line flag
terraform apply -var="environment=production"

# 2. terraform.tfvars file (auto-loaded)
# terraform.tfvars
environment   = "production"
instance_count = 3

# 3. *.auto.tfvars (auto-loaded)
# production.auto.tfvars

# 4. Environment variables (TF_VAR_ prefix)
export TF_VAR_environment=production
```

### Outputs

```hcl
# outputs.tf
output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true   # hidden in CLI output, still stored in state
}
```

```bash
# Access outputs after apply
terraform output
terraform output load_balancer_dns
terraform output -json
```

---

## State Management

State is Terraform's record of what infrastructure it manages. **Never manually edit state.**

### Remote State (required for teams)

```hcl
# backend.tf — store state in S3
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"   # prevent concurrent applies
  }
}
```

```hcl
# Terraform Cloud backend
terraform {
  cloud {
    organization = "my-org"
    workspaces {
      name = "my-app-production"
    }
  }
}
```

### State Commands

```bash
# List all tracked resources
terraform state list

# Show details of a resource
terraform state show aws_instance.web

# Move resource (rename without recreating)
terraform state mv aws_instance.web aws_instance.web_server

# Remove resource from state (stop managing, don't destroy)
terraform state rm aws_s3_bucket.logs

# Import existing resource into state
terraform import aws_s3_bucket.my_bucket existing-bucket-name

# Pull current state
terraform state pull

# Force unlock (if locked)
terraform force-unlock LOCK_ID
```

---

## Modules

Modules are **reusable packages** of Terraform configurations.

### Using a Module

```hcl
# From Terraform Registry
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "my-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = {
    Environment = var.environment
  }
}

# Access module outputs
resource "aws_instance" "web" {
  subnet_id = module.vpc.private_subnets[0]
}
```

### Writing a Module

```
modules/
└── ec2-instance/
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── README.md
```

```hcl
# modules/ec2-instance/variables.tf
variable "name" { type = string }
variable "instance_type" {
  type    = string
  default = "t3.micro"
}

# modules/ec2-instance/main.tf
resource "aws_instance" "this" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  tags = { Name = var.name }
}

# modules/ec2-instance/outputs.tf
output "instance_id" { value = aws_instance.this.id }
output "private_ip"  { value = aws_instance.this.private_ip }
```

```hcl
# Use local module
module "web_server" {
  source        = "./modules/ec2-instance"
  name          = "web-server"
  instance_type = "t3.medium"
}
```

---

## Workspaces & Environments

### Workspaces

```bash
# Create and switch workspace
terraform workspace new staging
terraform workspace new production
terraform workspace list
terraform workspace select production
terraform workspace show
```

```hcl
# Use workspace name in config
locals {
  instance_type = {
    staging    = "t3.micro"
    production = "t3.large"
  }
}

resource "aws_instance" "web" {
  instance_type = local.instance_type[terraform.workspace]
}
```

### Environment Pattern (recommended for complex setups)

```
infra/
├── modules/
│   ├── networking/
│   ├── compute/
│   └── database/
└── environments/
    ├── dev/
    │   ├── main.tf
    │   ├── variables.tf
    │   └── terraform.tfvars
    ├── staging/
    └── production/
```

---

## Terraform Workflow

```bash
# 1. Initialize (download providers, configure backend)
terraform init
terraform init -upgrade          # update providers

# 2. Validate syntax
terraform validate

# 3. Format code
terraform fmt
terraform fmt -recursive

# 4. Plan (preview changes, exit code 2 = changes exist)
terraform plan
terraform plan -out=tfplan        # save plan
terraform plan -var="env=prod"

# 5. Apply
terraform apply
terraform apply tfplan            # apply saved plan
terraform apply -auto-approve     # skip confirmation (CI/CD)
terraform apply -target=aws_instance.web  # specific resource

# 6. Destroy
terraform destroy
terraform destroy -target=aws_instance.web

# Refresh (sync state with real world)
terraform refresh

# Graph (visualize dependencies)
terraform graph | dot -Tpng > graph.png
```

---

## Real-World Examples

### AWS VPC + EKS Cluster

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project}-vpc"
  cidr = "10.0.0.0/16"

  azs             = data.aws_availability_zones.available.names
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project}-eks"
  cluster_version = "1.32"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 10
      desired_size   = 2
    }
  }
}
```

---

## Testing Terraform

### Terraform Test (native, v1.6+)

```hcl
# tests/s3_bucket.tftest.hcl
run "creates_bucket_with_versioning" {
  command = apply

  variables {
    bucket_name = "test-bucket-${run.id}"
    environment = "test"
  }

  assert {
    condition     = aws_s3_bucket.main.bucket == var.bucket_name
    error_message = "Bucket name mismatch"
  }
}
```

### Terratest (Go-based)

```go
func TestTerraformS3Bucket(t *testing.T) {
    opts := &terraform.Options{
        TerraformDir: "../modules/s3-bucket",
        Vars: map[string]interface{}{
            "bucket_name": "test-bucket-" + uuid.New().String(),
        },
    }
    defer terraform.Destroy(t, opts)
    terraform.InitAndApply(t, opts)

    bucketName := terraform.Output(t, opts, "bucket_name")
    assert.Equal(t, "test-bucket", bucketName[:11])
}
```

---

## Terraform in 2026 (OpenTofu)

### OpenTofu

In August 2023, HashiCorp changed Terraform's license from MPL to BSL (non-open-source). The community forked it as **OpenTofu**, now under the Linux Foundation.

```bash
# OpenTofu is a drop-in replacement
tofu init
tofu plan
tofu apply

# Same syntax, same providers
```

### What's New in 2026

| Feature | Description |
|---------|-------------|
| **Provider Functions** (Terraform 1.8+/OpenTofu 1.7+) | Call provider-defined functions in HCL |
| **Stacks** (Terraform Cloud) | Orchestrate multiple configurations as a unit |
| **Ephemeral Values** | Sensitive values that aren't stored in state |
| **State encryption** (OpenTofu 1.7) | Native state file encryption |
| **`terraform test`** | Built-in unit testing framework (no Terratest needed) |
| **Drift detection** | Automatic detection of out-of-band changes |

### Tools Ecosystem

| Tool | Purpose |
|------|---------|
| **Terragrunt** | DRY wrapper for Terraform (keep configs DRY across envs) |
| **Atlantis** | GitOps for Terraform — auto-plan/apply on PR |
| **Infracost** | Cost estimation before `terraform apply` |
| **tflint** | Linter for Terraform configurations |
| **Checkov** | Security scanning for IaC |
| **tfsec** | Terraform security scanner |
| **terraform-docs** | Auto-generate module documentation |
| **env0 / Spacelift** | Terraform collaboration platforms |

---

## Cheat Sheet

```bash
# === INIT & SETUP ===
terraform init                    # initialize + download providers
terraform init -upgrade           # upgrade provider versions
terraform init -reconfigure       # reconfigure backend

# === PLANNING ===
terraform validate                # syntax check
terraform fmt [-recursive]        # format code
terraform plan                    # preview changes
terraform plan -out=tfplan        # save plan to file
terraform plan -destroy           # preview destroy

# === APPLYING ===
terraform apply                   # apply changes (interactive)
terraform apply -auto-approve     # skip confirmation
terraform apply tfplan            # apply saved plan
terraform apply -target=res.name  # apply specific resource

# === DESTROYING ===
terraform destroy
terraform destroy -target=res.name
terraform destroy -auto-approve

# === STATE ===
terraform state list              # list all resources
terraform state show res.name     # show resource details
terraform state mv old new        # rename resource
terraform state rm res.name       # remove from state
terraform import res.name ID      # import existing
terraform refresh                 # sync state with reality

# === WORKSPACES ===
terraform workspace new name
terraform workspace select name
terraform workspace list
terraform workspace show
terraform workspace delete name

# === OUTPUTS ===
terraform output                  # all outputs
terraform output name             # specific output
terraform output -json            # JSON format

# === DEBUGGING ===
export TF_LOG=DEBUG               # verbose logging
export TF_LOG_PATH=./terraform.log
terraform graph | dot -Tpng > graph.png

# === OPENTOFU ===
tofu init / plan / apply / destroy    # same as terraform
```
