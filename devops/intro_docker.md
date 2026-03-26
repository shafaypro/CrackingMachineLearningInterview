# Docker – Complete Guide (2026 Edition)

Docker is the standard for packaging applications into portable, reproducible **containers**. Whether you're deploying microservices, running local dev environments, or building CI/CD pipelines, Docker is foundational to modern software delivery.

---

## Table of Contents
1. [What is Docker?](#what-is-docker)
2. [Core Concepts](#core-concepts)
3. [Installation & Setup](#installation--setup)
4. [Dockerfile](#dockerfile)
5. [Images & Containers](#images--containers)
6. [Volumes & Networking](#volumes--networking)
7. [Docker Compose](#docker-compose)
8. [Multi-Stage Builds](#multi-stage-builds)
9. [Docker in CI/CD](#docker-in-cicd)
10. [Security Best Practices](#security-best-practices)
11. [Docker in 2026](#docker-in-2026)
12. [Cheat Sheet](#cheat-sheet)

---

## What is Docker?

Docker packages an application and **all its dependencies** (libraries, runtime, config) into a **container** — a lightweight, isolated process that runs identically on any machine.

```
Without Docker:                    With Docker:
"It works on my machine!"   →     "It works everywhere."
```

### Docker vs VMs

| Feature | Virtual Machine | Docker Container |
|---------|----------------|------------------|
| Isolation | Full OS | Process-level |
| Startup time | Minutes | Milliseconds |
| Size | GBs | MBs |
| Overhead | High (hypervisor) | Minimal |
| Portability | Medium | High |

Containers share the **host OS kernel** — they're not VMs. They use Linux namespaces and cgroups for isolation.

---

## Core Concepts

### The Big Picture

```
Dockerfile  →  docker build  →  Image  →  docker run  →  Container
                                  ↕
                           Docker Registry
                         (Docker Hub, ECR, GCR)
```

| Concept | Description |
|---------|-------------|
| **Dockerfile** | Blueprint for building an image (text file with instructions) |
| **Image** | Read-only snapshot of a filesystem + metadata. Immutable. |
| **Container** | A running instance of an image. Ephemeral by default. |
| **Registry** | Remote store for images (Docker Hub, AWS ECR, GitHub GHCR) |
| **Layer** | Each Dockerfile instruction creates a cached, reusable layer |
| **Volume** | Persistent storage that survives container restarts |
| **Network** | Virtual network allowing containers to communicate |

---

## Installation & Setup

### Linux
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # run Docker without sudo
newgrp docker
```

### Verify
```bash
docker version
docker run hello-world
```

### Docker Desktop (Mac/Windows)
Download from [docker.com/products/docker-desktop](https://docker.com/products/docker-desktop). Includes Docker Engine + Compose + Kubernetes.

---

## Dockerfile

A `Dockerfile` is a script of instructions that Docker executes layer by layer to build an image.

### Basic Structure

```dockerfile
# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependency files first (cache optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (documentation only, doesn't publish)
EXPOSE 8000

# Default command to run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Key Instructions

| Instruction | Purpose | Example |
|-------------|---------|---------|
| `FROM` | Base image | `FROM python:3.12-slim` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `COPY` | Copy files from host | `COPY src/ /app/src/` |
| `ADD` | Like COPY + supports URLs/tar | `ADD archive.tar.gz /app/` |
| `RUN` | Execute command in new layer | `RUN apt-get install -y curl` |
| `ENV` | Set environment variable | `ENV PORT=8080` |
| `ARG` | Build-time variable | `ARG VERSION=1.0` |
| `EXPOSE` | Document port | `EXPOSE 5432` |
| `CMD` | Default run command | `CMD ["python", "app.py"]` |
| `ENTRYPOINT` | Fixed command (CMD appended) | `ENTRYPOINT ["gunicorn"]` |
| `USER` | Switch to non-root user | `USER appuser` |
| `HEALTHCHECK` | Container health check | See below |
| `LABEL` | Metadata | `LABEL version="1.0"` |

### CMD vs ENTRYPOINT

```dockerfile
# CMD: overridable default command
CMD ["python", "app.py"]
# docker run myimage python other.py   ← overrides CMD

# ENTRYPOINT: fixed executable, CMD becomes default args
ENTRYPOINT ["python"]
CMD ["app.py"]
# docker run myimage other.py   ← runs "python other.py"
```

### .dockerignore

Like `.gitignore` — prevents unnecessary files from being sent to the Docker build context:

```dockerignore
.git
__pycache__
*.pyc
.env
node_modules
*.log
.DS_Store
```

---

## Images & Containers

### Building Images

```bash
# Build from Dockerfile in current directory
docker build -t myapp:1.0 .

# Build with specific Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Build with build args
docker build --build-arg VERSION=2.0 -t myapp:2.0 .

# Build for specific platform (cross-compile)
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:latest .
```

### Image Management

```bash
# List local images
docker images
docker image ls

# Pull from registry
docker pull python:3.12-slim
docker pull ghcr.io/myorg/myapp:latest

# Push to registry
docker tag myapp:1.0 myrepo/myapp:1.0
docker push myrepo/myapp:1.0

# Inspect image layers
docker history myapp:1.0
docker inspect myapp:1.0

# Remove image
docker rmi myapp:1.0
docker image prune -a   # remove all unused images
```

### Running Containers

```bash
# Run interactively
docker run -it ubuntu bash

# Run in background (detached)
docker run -d --name my-postgres postgres:16

# Run with environment variables
docker run -d \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=mydb \
  --name postgres \
  postgres:16

# Run with port mapping (host:container)
docker run -d -p 8080:80 nginx

# Run with volume mount
docker run -d \
  -v /host/data:/container/data \
  myapp:1.0

# Run with resource limits
docker run -d \
  --memory="512m" \
  --cpus="1.5" \
  myapp:1.0

# Auto-remove when stopped
docker run --rm -it myapp:1.0 bash
```

### Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop / start / restart
docker stop my-app
docker start my-app
docker restart my-app

# Remove container
docker rm my-app
docker rm -f my-app  # force remove running container

# View logs
docker logs my-app
docker logs -f my-app   # follow (tail -f)
docker logs --tail 100 my-app

# Execute command in running container
docker exec -it my-app bash
docker exec my-app cat /etc/hosts

# Copy files to/from container
docker cp ./config.yaml my-app:/app/config.yaml
docker cp my-app:/app/logs/app.log ./logs/

# Inspect container
docker inspect my-app

# Stats (CPU, memory, network)
docker stats

# Clean up everything stopped
docker container prune
```

---

## Volumes & Networking

### Volumes

Volumes are the **preferred way** to persist data. They're managed by Docker and survive container restarts/removal.

```bash
# Create volume
docker volume create pgdata

# Use volume in container
docker run -d \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16

# Bind mount (maps host directory)
docker run -v $(pwd)/data:/app/data myapp

# List volumes
docker volume ls

# Inspect volume
docker volume inspect pgdata

# Remove volume
docker volume rm pgdata
docker volume prune   # remove all unused
```

**Volumes vs Bind Mounts vs tmpfs:**

| Type | Use Case | Managed By |
|------|----------|-----------|
| Volume | Production data persistence | Docker |
| Bind mount | Dev: sync host code into container | Host OS |
| tmpfs | Sensitive temp data (in memory only) | Memory |

### Networking

```bash
# List networks
docker network ls

# Create network
docker network create my-network

# Connect containers on same network
docker run -d --name db --network my-network postgres:16
docker run -d --name app --network my-network myapp:1.0
# app can reach db via hostname "db"

# Inspect network
docker network inspect my-network

# Connect/disconnect running container
docker network connect my-network my-app
docker network disconnect my-network my-app
```

**Built-in Networks:**

| Network | Description |
|---------|-------------|
| `bridge` | Default. Containers communicate via IP. |
| `host` | Container shares host network stack. |
| `none` | No networking. |
| Custom bridge | User-defined. Containers reach each other by name. |

---

## Docker Compose

Docker Compose defines and runs **multi-container applications** from a single `docker-compose.yml` file.

### Example: Full-Stack App

```yaml
# docker-compose.yml
version: "3.9"

services:
  # PostgreSQL database
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: appdb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U appuser"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis cache
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  # Backend API
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://appuser:secret@db:5432/appdb
      REDIS_URL: redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./backend:/app   # dev hot-reload
    restart: unless-stopped

  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      VITE_API_URL: http://localhost:8000
    depends_on:
      - api

volumes:
  pgdata:
```

### Compose Commands

```bash
# Start all services (detached)
docker compose up -d

# Start and rebuild images
docker compose up -d --build

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# View logs
docker compose logs -f
docker compose logs -f api

# Scale a service
docker compose up -d --scale api=3

# Run one-off command
docker compose run --rm api python manage.py migrate

# Exec into service
docker compose exec api bash

# Status
docker compose ps

# Restart single service
docker compose restart api

# Pull latest images
docker compose pull
```

---

## Multi-Stage Builds

Multi-stage builds keep production images **small** by separating build-time dependencies from runtime.

### Python Example

```dockerfile
# Stage 1: Build
FROM python:3.12 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production (lean)
FROM python:3.12-slim AS production

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app .

# Create non-root user
RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Node.js Example

```dockerfile
# Stage 1: Dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Stage 2: Build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Production
FROM node:20-alpine AS production
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY package.json .

EXPOSE 3000
CMD ["node", "dist/index.js"]
```

---

## Docker in CI/CD

### GitHub Actions Example

```yaml
name: Build and Push

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

---

## Security Best Practices

```dockerfile
# 1. Use specific image tags, not :latest
FROM python:3.12.3-slim-bookworm   # not FROM python:latest

# 2. Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# 3. Minimize layers and clean up in same RUN step
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 4. Don't copy secrets into images
# Use --secret or runtime environment variables

# 5. Use COPY over ADD (more explicit)
COPY requirements.txt .

# 6. Set read-only filesystem
# docker run --read-only myapp (with tmpfs for /tmp)

# 7. Scan images for vulnerabilities
# docker scout cves myapp:latest
# trivy image myapp:latest
```

### Secrets Management

```bash
# Build-time secret (never stored in image layer)
docker build --secret id=mysecret,src=./secret.txt .
```

```dockerfile
# In Dockerfile - use secret without storing it
RUN --mount=type=secret,id=mysecret \
    cat /run/secrets/mysecret | pip install --extra-index-url ...
```

---

## Docker in 2026

### What's Changed

| Feature | Status in 2026 |
|---------|---------------|
| **Docker Scout** | Built-in vulnerability scanning in Docker Desktop |
| **Buildx / BuildKit** | Default build backend; multi-platform builds standard |
| **Docker Init** | `docker init` generates Dockerfile + Compose automatically |
| **Testcontainers** | Standard for integration testing with real containers |
| **Wasm support** | Docker can run WebAssembly workloads alongside Linux containers |
| **OCI standard** | All major runtimes (containerd, podman) are OCI-compliant |

### Alternatives to Know

| Tool | Description |
|------|-------------|
| **Podman** | Daemonless, rootless Docker alternative (Red Hat) |
| **nerdctl** | Docker-compatible CLI for containerd |
| **Finch** | AWS's open-source container CLI for macOS |
| **Buildpacks** | Build images from source without Dockerfiles (Cloud Native Buildpacks) |

---

## Cheat Sheet

```bash
# === IMAGES ===
docker build -t name:tag .              # Build image
docker pull image:tag                   # Pull image
docker push repo/image:tag             # Push image
docker images                          # List images
docker rmi image:tag                   # Remove image
docker image prune -a                  # Remove all unused images

# === CONTAINERS ===
docker run -d -p host:container name    # Run detached with port
docker run -it image bash              # Interactive shell
docker run --rm image command          # Auto-remove on exit
docker ps                              # Running containers
docker ps -a                           # All containers
docker stop/start/restart name         # Lifecycle
docker rm name                         # Remove container
docker logs -f name                    # Tail logs
docker exec -it name bash              # Shell into running container
docker stats                           # Resource usage

# === VOLUMES ===
docker volume create vol               # Create volume
docker volume ls                       # List volumes
docker run -v vol:/path image          # Attach volume
docker run -v $(pwd):/path image       # Bind mount

# === NETWORKING ===
docker network create net              # Create network
docker network ls                      # List networks
docker run --network net image         # Use network

# === COMPOSE ===
docker compose up -d                   # Start all services
docker compose up -d --build          # Start and rebuild
docker compose down                    # Stop all
docker compose down -v                 # Stop + remove volumes
docker compose logs -f [service]       # Logs
docker compose exec service bash       # Shell into service
docker compose ps                      # Status

# === CLEANUP ===
docker system prune                    # Remove all unused
docker system prune -a --volumes       # Nuclear cleanup
docker container prune                 # Remove stopped containers
docker volume prune                    # Remove unused volumes
docker image prune -a                  # Remove unused images
```
