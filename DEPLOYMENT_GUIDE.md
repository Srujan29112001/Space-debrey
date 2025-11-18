# Space Debris Tracking System - Complete Deployment Guide

This comprehensive guide covers everything you need to build, run, test, and deploy the Space Debris Tracking System from development to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Building the Project](#building-the-project)
4. [Running Tests](#running-tests)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Cloud Provider Deployment](#cloud-provider-deployment)
8. [Production Configuration](#production-configuration)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Monitoring & Observability](#monitoring--observability)
11. [Backup & Disaster Recovery](#backup--disaster-recovery)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

#### Minimum (Development)
- **CPU**: 4 cores (x86_64)
- **RAM**: 16 GB
- **Storage**: 50 GB free space
- **GPU** (Optional): NVIDIA GPU with 4GB VRAM

#### Recommended (Production)
- **CPU**: 16+ cores (x86_64)
- **RAM**: 64 GB
- **Storage**: 500 GB SSD
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 Ti or better)

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10 or 3.11 (3.11 recommended)
- **Docker**: 24.0+ and Docker Compose 2.20+
- **Git**: 2.30+
- **CUDA**: 11.8+ (for GPU support)
- **kubectl**: 1.28+ (for Kubernetes deployments)
- **Helm**: 3.12+ (optional, for Kubernetes package management)

### Access Requirements

- **GitHub**: Access to the repository
- **Space-Track.org**: Account for TLE data (optional)
- **Cloud Provider**: AWS/GCP/Azure account (for cloud deployment)
- **Container Registry**: Docker Hub, AWS ECR, GCP GCR, or Azure ACR

---

## Local Development Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/Srujan29112001/Space-debrey.git
cd Space-debrey

# Run automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:
- Check prerequisites
- Create virtual environment
- Install all dependencies
- Create directory structure
- Generate `.env` configuration
- Optionally start Docker services
- Run validation tests

### Option 2: Manual Setup

#### 1. Clone Repository

```bash
git clone https://github.com/Srujan29112001/Space-debrey.git
cd Space-debrey
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate
```

#### 3. Upgrade Core Tools

```bash
pip install --upgrade pip setuptools wheel
```

#### 4. Install Dependencies

**For GPU support:**
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

**For CPU only:**
```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

#### 5. Create Directory Structure

```bash
mkdir -p data/{telescope_images,radar_data,tle_data}/{raw,processed}
mkdir -p data/{training,validation,test}/{debris_images,orbits,conjunctions}
mkdir -p models/{yolo,dino,pinn,transformer,mamba,ensemble}
mkdir -p logs cache checkpoints
```

#### 6. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Generate JWT secret
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env

# Edit .env and configure:
# - Database passwords
# - API keys
# - External service credentials
nano .env  # or vim .env
```

**Critical `.env` variables to configure:**
```bash
# Database
NEO4J_PASSWORD=your_secure_password_here
POSTGRES_PASSWORD=your_secure_password_here

# JWT Security
JWT_SECRET_KEY=<generated_secret_from_above>

# Space-Track.org (for real TLE data)
SPACETRACK_USERNAME=your_username
SPACETRACK_PASSWORD=your_password

# Email alerts (if using)
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Set environment
ENVIRONMENT=development
```

#### 7. Start Supporting Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
sleep 30

# Check service health
docker-compose ps
```

Expected services:
- Neo4j (ports 7474, 7687)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Kafka (port 9092)
- Prometheus (port 9090)
- Grafana (port 3000)

#### 8. Initialize Databases

```bash
# Initialize Neo4j schema
python -c "from space_debris_tracker.knowledge_graph import SpaceKnowledgeGraph; kg = SpaceKnowledgeGraph(); print('Neo4j initialized')"

# Initialize PostgreSQL schema (if applicable)
# python scripts/init_postgres.py
```

#### 9. Generate Sample Data (Optional)

```bash
# Generate all sample data
python scripts/generate_sample_data.py --all

# Or generate specific types
python scripts/generate_sample_data.py --type images --count 100
python scripts/generate_sample_data.py --type tle --count 500
python scripts/generate_sample_data.py --type conjunctions --count 50
```

---

## Building the Project

### Python Package Build

```bash
# Activate virtual environment
source venv/bin/activate

# Build distribution packages
python -m build

# This creates:
# - dist/space_debris_tracker-0.1.0.tar.gz
# - dist/space_debris_tracker-0.1.0-py3-none-any.whl
```

### Docker Image Build

#### Single Platform Build (Local Development)

```bash
# Build the Docker image
docker build -t space-debris-tracker:latest .

# Tag for specific version
docker build -t space-debris-tracker:v1.0.0 .

# View built image
docker images | grep space-debris-tracker
```

#### Multi-Platform Build (Production)

```bash
# Create buildx builder (first time only)
docker buildx create --name multiarch --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t yourdockerhub/space-debris-tracker:latest \
  -t yourdockerhub/space-debris-tracker:v1.0.0 \
  --push \
  .
```

#### Build with Custom Options

```bash
# Build with GPU support
docker build \
  --build-arg CUDA_VERSION=11.8.0 \
  -t space-debris-tracker:gpu \
  .

# Build with specific Python version
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t space-debris-tracker:py311 \
  .

# Build without cache (clean build)
docker build --no-cache -t space-debris-tracker:latest .
```

### Verify Docker Build

```bash
# Run a quick test
docker run --rm space-debris-tracker:latest python -c "import space_debris_tracker; print('Success!')"

# Check image size
docker images space-debris-tracker:latest --format "{{.Size}}"
```

---

## Running Tests

### Unit Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ -v --cov=space_debris_tracker --cov-report=html

# Run specific test file
pytest tests/unit/test_detector.py -v

# Run tests matching pattern
pytest tests/unit/ -k "test_detection" -v
```

### Integration Tests

```bash
# Start required services first
docker-compose up -d

# Wait for services
sleep 30

# Run integration tests
pytest tests/integration/ -v

# Run with services cleanup
pytest tests/integration/ -v --setup-show
```

### Performance Tests

```bash
# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only

# Save benchmark results
pytest tests/performance/ -v --benchmark-save=baseline

# Compare with baseline
pytest tests/performance/ -v --benchmark-compare=baseline
```

### End-to-End Tests

```bash
# Run full system tests (requires all services)
pytest tests/e2e/ -v --tb=short

# Run with specific markers
pytest tests/e2e/ -v -m "smoke"
pytest tests/e2e/ -v -m "critical"
```

### Test Report Generation

```bash
# Generate HTML coverage report
pytest tests/ --cov=space_debris_tracker --cov-report=html
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux

# Generate JUnit XML report (for CI/CD)
pytest tests/ --junitxml=reports/junit.xml

# Generate combined reports
pytest tests/ \
  --cov=space_debris_tracker \
  --cov-report=html \
  --cov-report=xml \
  --junitxml=reports/junit.xml
```

### Code Quality Checks

```bash
# Run flake8
flake8 space_debris_tracker/ --max-line-length=100 --count --statistics

# Run black (auto-format)
black space_debris_tracker/ tests/

# Run mypy (type checking)
mypy space_debris_tracker/ --ignore-missing-imports

# Run isort (import sorting)
isort space_debris_tracker/ tests/

# Run pylint
pylint space_debris_tracker/ --max-line-length=100

# Run security checks
bandit -r space_debris_tracker/
safety check
```

---

## Docker Deployment

### Development Environment

#### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose down -v
```

#### Service URLs

After starting with `docker-compose up -d`:

- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Production Environment

#### 1. Build Production Image

```bash
# Build production-optimized image
docker build \
  --target production \
  -t space-debris-tracker:prod \
  -f Dockerfile.production \
  .
```

#### 2. Tag and Push to Registry

**Docker Hub:**
```bash
docker tag space-debris-tracker:prod yourusername/space-debris-tracker:v1.0.0
docker tag space-debris-tracker:prod yourusername/space-debris-tracker:latest
docker push yourusername/space-debris-tracker:v1.0.0
docker push yourusername/space-debris-tracker:latest
```

**AWS ECR:**
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag space-debris-tracker:prod 123456789.dkr.ecr.us-east-1.amazonaws.com/space-debris-tracker:v1.0.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/space-debris-tracker:v1.0.0
```

**Google GCR:**
```bash
# Authenticate
gcloud auth configure-docker

# Tag and push
docker tag space-debris-tracker:prod gcr.io/your-project/space-debris-tracker:v1.0.0
docker push gcr.io/your-project/space-debris-tracker:v1.0.0
```

#### 3. Run Production Container

```bash
# Run API server
docker run -d \
  --name space-tracker-api \
  -p 8000:8000 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_PASSWORD=your_password \
  -e ENVIRONMENT=production \
  -v /data/models:/app/models:ro \
  -v /data/logs:/app/logs \
  --restart unless-stopped \
  space-debris-tracker:prod

# Run dashboard
docker run -d \
  --name space-tracker-dashboard \
  -p 8501:8501 \
  -e API_URL=http://api:8000 \
  --restart unless-stopped \
  space-debris-tracker:prod \
  streamlit run space_debris_tracker/dashboard/app.py --server.port 8501
```

#### 4. Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check container status
docker ps
docker stats space-tracker-api

# View logs
docker logs -f space-tracker-api
```

### Docker Compose Production Setup

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    image: yourusername/space-debris-tracker:v1.0.0
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    environment:
      - ENVIRONMENT=production
      - NEO4J_URI=bolt://neo4j:7687
    volumes:
      - /opt/space-tracker/models:/app/models:ro
      - /opt/space-tracker/logs:/app/logs
    restart: always
```

Deploy:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Verify kubectl is configured
kubectl version --client
kubectl cluster-info

# Verify you have access
kubectl get nodes
```

### Local Kubernetes (Development)

#### Using Minikube

```bash
# Start Minikube with GPU support
minikube start \
  --driver=docker \
  --cpus=4 \
  --memory=8192 \
  --disk-size=50g \
  --gpus=all

# Enable addons
minikube addons enable metrics-server
minikube addons enable ingress

# Deploy to Minikube
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml

# Access services
minikube service space-tracker-api -n space-tracking
minikube service space-tracker-dashboard -n space-tracking
```

#### Using Kind (Kubernetes in Docker)

```bash
# Create cluster
kind create cluster --name space-tracker --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

# Deploy
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml
```

### Production Kubernetes Deployment

#### 1. Prepare Secrets

```bash
# Create namespace
kubectl create namespace space-tracking

# Create secrets
kubectl create secret generic space-tracker-secrets \
  --from-literal=neo4j-password='your_secure_password' \
  --from-literal=postgres-password='your_secure_password' \
  --from-literal=jwt-secret='your_jwt_secret' \
  --from-literal=api-key='your_api_key' \
  -n space-tracking

# Create registry secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=your-registry.io \
  --docker-username=your-username \
  --docker-password=your-password \
  --docker-email=your-email@example.com \
  -n space-tracking
```

#### 2. Deploy Application

```bash
# Deploy all resources
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml

# Verify deployment
kubectl get all -n space-tracking

# Check pod status
kubectl get pods -n space-tracking -w

# Check logs
kubectl logs -f deployment/space-tracker-api -n space-tracking
```

#### 3. Configure Ingress (Optional)

Create `ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: space-tracker-ingress
  namespace: space-tracking
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.spacedebris.example.com
    - dashboard.spacedebris.example.com
    secretName: space-tracker-tls
  rules:
  - host: api.spacedebris.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: space-tracker-api
            port:
              number: 80
  - host: dashboard.spacedebris.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: space-tracker-dashboard
            port:
              number: 80
```

Apply:
```bash
kubectl apply -f ingress.yaml
```

#### 4. Configure Persistent Volumes

```bash
# Create PersistentVolumeClaim for models
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: space-tracking
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
EOF
```

#### 5. Configure Horizontal Pod Autoscaling

```bash
# View HPA status
kubectl get hpa -n space-tracking

# Manually scale
kubectl scale deployment space-tracker-api --replicas=5 -n space-tracking

# Update HPA limits
kubectl patch hpa space-tracker-api-hpa -n space-tracking -p '{"spec":{"maxReplicas":50}}'
```

#### 6. Monitor Deployment

```bash
# Watch rollout status
kubectl rollout status deployment/space-tracker-api -n space-tracking

# View events
kubectl get events -n space-tracking --sort-by='.lastTimestamp'

# View resource usage
kubectl top pods -n space-tracking
kubectl top nodes

# Describe deployment
kubectl describe deployment space-tracker-api -n space-tracking
```

### Update Deployment

#### Rolling Update

```bash
# Update image
kubectl set image deployment/space-tracker-api \
  api=yourusername/space-debris-tracker:v1.1.0 \
  -n space-tracking

# Watch rollout
kubectl rollout status deployment/space-tracker-api -n space-tracking

# Rollback if needed
kubectl rollout undo deployment/space-tracker-api -n space-tracking
```

#### Using kubectl apply

```bash
# Edit deployment
nano space_debris_tracker/deployment/kubernetes/deployment.yaml

# Apply changes
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml

# Verify
kubectl get pods -n space-tracking -w
```

---

## Cloud Provider Deployment

### AWS Deployment

#### Option 1: ECS (Elastic Container Service)

**1. Create ECR Repository:**
```bash
aws ecr create-repository --repository-name space-debris-tracker --region us-east-1
```

**2. Push Image:**
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag space-debris-tracker:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/space-debris-tracker:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/space-debris-tracker:latest
```

**3. Create ECS Cluster:**
```bash
aws ecs create-cluster --cluster-name space-tracker-cluster --region us-east-1
```

**4. Create Task Definition:**
```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

**5. Create Service:**
```bash
aws ecs create-service \
  --cluster space-tracker-cluster \
  --service-name space-tracker-api \
  --task-definition space-tracker-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### Option 2: EKS (Elastic Kubernetes Service)

**1. Create EKS Cluster:**
```bash
eksctl create cluster \
  --name space-tracker-cluster \
  --version 1.28 \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type g4dn.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed
```

**2. Configure kubectl:**
```bash
aws eks update-kubeconfig --name space-tracker-cluster --region us-east-1
```

**3. Deploy:**
```bash
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml
```

**4. Install GPU Operator (for GPU support):**
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### Google Cloud Platform (GCP)

#### Using GKE (Google Kubernetes Engine)

**1. Create GKE Cluster:**
```bash
gcloud container clusters create space-tracker-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --addons HorizontalPodAutoscaling,HttpLoadBalancing
```

**2. Configure kubectl:**
```bash
gcloud container clusters get-credentials space-tracker-cluster --zone us-central1-a
```

**3. Install GPU drivers:**
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

**4. Deploy:**
```bash
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml
```

### Azure Deployment

#### Using AKS (Azure Kubernetes Service)

**1. Create Resource Group:**
```bash
az group create --name SpaceTrackerRG --location eastus
```

**2. Create AKS Cluster:**
```bash
az aks create \
  --resource-group SpaceTrackerRG \
  --name space-tracker-cluster \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

**3. Configure kubectl:**
```bash
az aks get-credentials --resource-group SpaceTrackerRG --name space-tracker-cluster
```

**4. Deploy:**
```bash
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml
```

---

## Production Configuration

### Security Hardening

#### 1. Use Secrets Management

**AWS Secrets Manager:**
```python
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# In your application
neo4j_password = get_secret('space-tracker/neo4j-password')
```

**Google Secret Manager:**
```python
from google.cloud import secretmanager

def get_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')
```

#### 2. Enable HTTPS/TLS

**Generate SSL certificates with Let's Encrypt:**
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### 3. Network Security

```bash
# Create NetworkPolicy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: space-tracker-network-policy
  namespace: space-tracking
spec:
  podSelector:
    matchLabels:
      app: space-tracker-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: space-tracker-dashboard
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
EOF
```

### Performance Optimization

#### 1. Enable Model Compilation

```python
# In .env
COMPILE_MODELS=true
MIXED_PRECISION=true
```

#### 2. Configure Connection Pooling

```python
# PostgreSQL connection pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
```

#### 3. Enable Caching

```python
# Redis caching
ENABLE_REDIS_CACHE=true
CACHE_TTL_SECONDS=3600
ENABLE_RESPONSE_CACHE=true
```

### Resource Limits

#### Kubernetes Resource Quotas

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: space-tracker-quota
  namespace: space-tracking
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.cpu: "200"
    limits.memory: 400Gi
    persistentvolumeclaims: "10"
EOF
```

### High Availability

#### 1. Multi-Region Deployment

```bash
# Deploy to multiple regions
for region in us-east-1 us-west-2 eu-west-1; do
  kubectl apply -f deployment.yaml --context=$region
done
```

#### 2. Database Replication

**Neo4j Cluster:**
```yaml
# neo4j-cluster.yaml
apiVersion: v1
kind: Service
metadata:
  name: neo4j-cluster
spec:
  clusterIP: None
  selector:
    app: neo4j
  ports:
  - port: 7687
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
spec:
  serviceName: neo4j-cluster
  replicas: 3
  ...
```

---

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Run tests
        run: |
          pytest tests/unit/ -v --cov=space_debris_tracker
          flake8 space_debris_tracker/

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            yourusername/space-debris-tracker:latest
            yourusername/space-debris-tracker:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBECONFIG }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/space-tracker-api \
            api=yourusername/space-debris-tracker:${{ github.sha }} \
            -n space-tracking
          kubectl rollout status deployment/space-tracker-api -n space-tracking
```

### GitLab CI/CD

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install -e .
    - pytest tests/unit/ -v --cov=space_debris_tracker
    - flake8 space_debris_tracker/
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT
    - kubectl set image deployment/space-tracker-api api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n space-tracking
    - kubectl rollout status deployment/space-tracker-api -n space-tracking
  only:
    - main
```

---

## Monitoring & Observability

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

**Key metrics:**
- `detection_latency_seconds` - Object detection latency
- `prediction_latency_seconds` - Trajectory prediction latency
- `conjunction_probability` - Collision probability gauge
- `api_request_duration_seconds` - API response times
- `gpu_utilization` - GPU usage percentage

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Add Prometheus data source:
   - URL: http://prometheus:9090
4. Import dashboard: `monitoring/grafana-dashboards/space-tracker.json`

### Logging

#### Centralized Logging with ELK Stack

```bash
# Deploy Elasticsearch, Logstash, Kibana
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: elasticsearch
  namespace: space-tracking
spec:
  ports:
  - port: 9200
  selector:
    app: elasticsearch
---
# ... (Elasticsearch, Logstash, Kibana deployments)
EOF
```

#### Application Logging

```python
# Configure structured logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/space_debris_tracker.log
```

### Alerting

#### Prometheus AlertManager

Create `alerting-rules.yaml`:

```yaml
groups:
- name: space-tracker-alerts
  rules:
  - alert: HighCollisionProbability
    expr: conjunction_probability > 0.001
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High collision probability detected"
      description: "Collision probability {{ $value }} exceeds threshold"

  - alert: APIHighLatency
    expr: api_request_duration_seconds{quantile="0.99"} > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "API latency is high"
```

---

## Backup & Disaster Recovery

### Database Backups

#### Neo4j Backup

```bash
# Create backup
docker exec neo4j neo4j-admin dump \
  --database=space_catalog \
  --to=/backups/neo4j-backup-$(date +%Y%m%d).dump

# Restore backup
docker exec neo4j neo4j-admin load \
  --from=/backups/neo4j-backup-20240101.dump \
  --database=space_catalog \
  --force
```

#### PostgreSQL Backup

```bash
# Create backup
docker exec postgres pg_dump -U spaceuser space_tracking > backup-$(date +%Y%m%d).sql

# Restore backup
docker exec -i postgres psql -U spaceuser space_tracking < backup-20240101.sql
```

### Automated Backups

Create cron job:
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /opt/space-tracker/scripts/backup.sh
```

### Disaster Recovery Plan

1. **Regular backups**: Daily automated backups to S3/GCS
2. **Multi-region replication**: Data replicated across regions
3. **Infrastructure as Code**: All configs in version control
4. **Recovery Time Objective (RTO)**: < 4 hours
5. **Recovery Point Objective (RPO)**: < 1 hour

---

## Troubleshooting

### Common Issues

#### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### Issue: GPU Not Detected

**Problem:** `CUDA not available`

**Solution:**
```bash
# Check GPU
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Issue: Database Connection Failed

**Problem:** `Neo4jError: Could not connect to bolt://localhost:7687`

**Solution:**
```bash
# Check Neo4j is running
docker-compose ps neo4j

# Start Neo4j
docker-compose up -d neo4j

# Wait and test connection
sleep 30
docker exec neo4j cypher-shell -u neo4j -p password "RETURN 1"
```

#### Issue: Port Already in Use

**Problem:** `Address already in use: 8000`

**Solution:**
```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
API_PORT=8001 python -m space_debris_tracker.api.server
```

#### Issue: Out of GPU Memory

**Problem:** `CUDA out of memory`

**Solution:**
```bash
# Edit config.yaml
batch_size: 4  # Reduce from 16
mixed_precision: true
gradient_accumulation_steps: 4
```

#### Issue: Docker Build Fails

**Problem:** Build fails with dependency errors

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Build without cache
docker build --no-cache -t space-debris-tracker:latest .

# Check Docker resources
docker system df
```

### Debug Commands

```bash
# Check system resources
df -h
free -h
nvidia-smi

# Check Docker
docker ps -a
docker-compose ps
docker logs <container_name>

# Check Kubernetes
kubectl get pods -n space-tracking
kubectl describe pod <pod_name> -n space-tracking
kubectl logs <pod_name> -n space-tracking

# Check services
curl http://localhost:8000/health
curl http://localhost:9090/api/v1/query?query=up

# Check database
docker exec -it neo4j cypher-shell -u neo4j -p password
docker exec -it postgres psql -U spaceuser space_tracking

# Network debugging
nc -zv localhost 8000
telnet localhost 7687
```

### Performance Profiling

```bash
# Profile Python code
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats

# Profile API
pip install py-spy
py-spy record -o profile.svg --pid <API_PID>

# Profile memory
pip install memory_profiler
python -m memory_profiler script.py
```

---

## Additional Resources

### Documentation
- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

### External Links
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Neo4j Documentation](https://neo4j.com/docs/)

### Support
- **Issues**: https://github.com/Srujan29112001/Space-debrey/issues
- **Discussions**: https://github.com/Srujan29112001/Space-debrey/discussions

---

## Summary

This guide covered:
- ✅ Local development setup (automated and manual)
- ✅ Building Python packages and Docker images
- ✅ Running comprehensive tests
- ✅ Docker deployment (development and production)
- ✅ Kubernetes deployment (local and cloud)
- ✅ Cloud provider deployment (AWS, GCP, Azure)
- ✅ Production configuration and security
- ✅ CI/CD pipeline setup
- ✅ Monitoring and observability
- ✅ Backup and disaster recovery
- ✅ Troubleshooting common issues

For quick reference:

**Development:**
```bash
./scripts/setup.sh && source venv/bin/activate
```

**Testing:**
```bash
pytest tests/ -v --cov=space_debris_tracker
```

**Local Deployment:**
```bash
docker-compose up -d
```

**Production Deployment:**
```bash
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml
```

Happy deploying! 🚀
