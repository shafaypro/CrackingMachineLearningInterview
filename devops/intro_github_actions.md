# GitHub Actions – CI/CD Guide (2026 Edition)

GitHub Actions is GitHub's built-in **CI/CD and automation platform**. It lets you automate build, test, deploy, and any workflow directly in your repository using YAML.

---

## Core Concepts

```
Event (push, PR, schedule)  →  Workflow  →  Job  →  Step  →  Action
```

| Concept | Description |
|---------|-------------|
| **Workflow** | Automated process defined in `.github/workflows/*.yml` |
| **Event** | Trigger (push, pull_request, schedule, workflow_dispatch) |
| **Job** | Group of steps running on the same runner |
| **Step** | Individual task (shell command or action) |
| **Action** | Reusable unit of work (from marketplace or local) |
| **Runner** | Machine that executes jobs (GitHub-hosted or self-hosted) |
| **Environment** | Named deployment target with protection rules |

---

## Basic Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v --cov=src

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## Triggers (on:)

```yaml
on:
  # Code events
  push:
    branches: [main]
    paths: ["src/**", "tests/**"]   # only if these paths changed

  pull_request:
    types: [opened, synchronize, reopened]
    branches: [main]

  # Manual trigger
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options: [staging, production]

  # Scheduled (cron)
  schedule:
    - cron: "0 6 * * 1-5"   # 6am Mon-Fri UTC

  # Triggered by another workflow
  workflow_run:
    workflows: ["CI"]
    types: [completed]

  # GitHub events
  release:
    types: [published]
  issues:
    types: [opened]
```

---

## Jobs & Steps

```yaml
jobs:
  build:
    name: Build and Test
    runs-on: ubuntu-latest
    timeout-minutes: 30

    # Job-level environment variables
    env:
      NODE_ENV: test

    # Service containers (side-car processes)
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: secret
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      # Use an Action from marketplace
      - uses: actions/checkout@v4

      # Shell command
      - name: Run lint
        run: |
          npm install
          npm run lint

      # Conditional step
      - name: Deploy
        if: github.ref == 'refs/heads/main' && success()
        run: ./deploy.sh

      # Step with output
      - name: Get version
        id: version
        run: echo "version=$(cat VERSION)" >> $GITHUB_OUTPUT

      - name: Use version
        run: echo "Deploying version ${{ steps.version.outputs.version }}"

      # Continue on error
      - name: Optional step
        continue-on-error: true
        run: ./optional-check.sh
```

---

## Matrix Strategy (run jobs across multiple configs)

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
        os: [ubuntu-latest, macos-latest]
      fail-fast: false    # don't cancel others on failure

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest tests/
```

---

## Job Dependencies & Fan-out/Fan-in

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: echo "linting..."

  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "testing..."

  build:
    needs: [lint, test]   # runs after both pass
    runs-on: ubuntu-latest
    steps:
      - run: echo "building..."

  deploy:
    needs: build
    if: needs.build.result == 'success'
    runs-on: ubuntu-latest
    steps:
      - run: echo "deploying..."
```

---

## Secrets & Variables

```yaml
# Access secrets
- name: Deploy
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  run: aws s3 sync ./dist s3://my-bucket

# Repository variables (non-secret config)
- run: echo "Region is ${{ vars.AWS_REGION }}"

# GitHub token (auto-provided)
- uses: actions/github-script@v7
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

---

## Reusable Workflows

```yaml
# .github/workflows/deploy.yml (reusable)
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
    secrets:
      DEPLOY_KEY:
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - run: ./deploy.sh ${{ inputs.environment }}
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
```

```yaml
# .github/workflows/ci.yml (caller)
jobs:
  deploy-staging:
    uses: ./.github/workflows/deploy.yml
    with:
      environment: staging
    secrets:
      DEPLOY_KEY: ${{ secrets.STAGING_DEPLOY_KEY }}
```

---

## Complete CI/CD Pipeline Example

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  IMAGE_NAME: ghcr.io/${{ github.repository }}

jobs:
  # ── Test ──────────────────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src --cov-report=xml

      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  # ── Build & Push Image ────────────────
  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/setup-buildx-action@v3

      - id: build
        uses: docker/build-push-action@v5
        with:
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: |
            ${{ env.IMAGE_NAME }}:latest
            ${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Deploy Staging ────────────────────
  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBECONFIG_STAGING }}

      - run: |
          helm upgrade --install my-app ./helm/my-app \
            --namespace staging \
            --set image.tag=${{ github.sha }} \
            -f helm/values-staging.yaml

  # ── Deploy Production (manual approval) ──
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production   # has required reviewers

    steps:
      - uses: actions/checkout@v4

      - uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBECONFIG_PROD }}

      - run: |
          helm upgrade --install my-app ./helm/my-app \
            --namespace production \
            --set image.tag=${{ github.sha }} \
            -f helm/values-production.yaml
```

---

## Useful Actions (2026)

```yaml
# Checkout
- uses: actions/checkout@v4
  with:
    fetch-depth: 0     # full history for git-based versioning

# Language setup
- uses: actions/setup-python@v5
- uses: actions/setup-node@v4
- uses: actions/setup-go@v5
- uses: actions/setup-java@v4

# Caching
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

# Docker
- uses: docker/login-action@v3
- uses: docker/setup-buildx-action@v3
- uses: docker/build-push-action@v5

# Cloud
- uses: aws-actions/configure-aws-credentials@v4
- uses: google-github-actions/auth@v2
- uses: azure/login@v2

# Kubernetes / Helm
- uses: azure/setup-kubectl@v3
- uses: azure/setup-helm@v3

# Security
- uses: aquasecurity/trivy-action@master    # container scanning
- uses: anchore/scan-action@v3              # SBOM + vuln scan

# Notifications
- uses: slackapi/slack-github-action@v1
```

---

## Cheat Sheet

```bash
# Workflow file location
.github/workflows/*.yml

# GitHub Context Variables
${{ github.sha }}           # commit SHA
${{ github.ref }}           # refs/heads/main
${{ github.event_name }}    # push / pull_request
${{ github.actor }}         # username
${{ github.repository }}    # owner/repo
${{ github.workspace }}     # /home/runner/work/...

# Job context
${{ job.status }}           # success / failure / cancelled
${{ runner.os }}            # Linux / macOS / Windows

# Expressions
${{ success() }}
${{ failure() }}
${{ always() }}
${{ cancelled() }}
${{ contains(github.ref, 'main') }}
${{ startsWith(github.ref, 'refs/tags/') }}

# Set output
echo "key=value" >> $GITHUB_OUTPUT

# Set env var for subsequent steps
echo "MY_VAR=hello" >> $GITHUB_ENV

# Mask value in logs
echo "::add-mask::$SECRET_VALUE"

# Group log lines
echo "::group::Section Name"
echo "::endgroup::"
```
