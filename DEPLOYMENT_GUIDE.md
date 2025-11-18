# Space Debris Tracking System - Complete Deployment & Build Guide

This comprehensive guide covers everything you need to know about building, running, and deploying the Space Debris Tracking & Autonomous Collision Prediction System.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Building the Project](#building-the-project)
4. [Running the Project](#running-the-project)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **RAM**: 16 GB (32 GB recommended)
- **Storage**: 50 GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS (12+), or Windows with WSL2
- **Python**: 3.10 or 3.11

#### Recommended for ML/AI Features
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or higher
- **cuDNN**: Compatible with your CUDA version

### Software Requirements

#### Required Tools
```bash
# Python 3.10+
python3 --version

# Git
git --version

# Docker & Docker Compose (for containerized deployment)
docker --version
docker-compose --version

# For Kubernetes deployment
kubectl version
```

#### Optional Tools
```bash
# For GPU support
nvidia-smi

# For container orchestration
helm version

# For monitoring
prometheus --version
```

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Srujan29112001/Space-debrey.git
cd Space-debrey
```

### 2. Automated Setup (Recommended)

The quickest way to get started:

```bash
# Run the automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This script will:
- ✅ Check system requirements
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Generate `.env` configuration file
- ✅ Run basic validation tests

### 3. Manual Setup (Alternative)

If you prefer manual control or the automated script fails:

#### Step 3.1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

#### Step 3.2: Upgrade Core Tools

```bash
pip install --upgrade pip setuptools wheel
```

#### Step 3.3: Install PyTorch

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**
```bash
pip install torch torchvision torchaudio
```

#### Step 3.4: Install Project Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

#### Step 3.5: Install Development Dependencies (Optional)

```bash
pip install pytest pytest-asyncio pytest-cov black flake8 mypy pre-commit
```

### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Generate secure JWT secret
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env

# Edit configuration with your settings
nano .env  # or vim, code, etc.
```

**Key configurations to update in `.env`:**
- Database passwords (Neo4j, PostgreSQL, Redis)
- API keys and JWT secrets
- GPU settings (if applicable)
- External service credentials (Space-Track.org, email SMTP, etc.)

### 5. Create Directory Structure

```bash
# Create required directories
mkdir -p data/{telescope_images,radar_data,tle_data,training,validation,test}
mkdir -p models/{yolo,dino,pinn,transformer,mamba,ensemble}
mkdir -p logs cache checkpoints
```

### 6. Verify Installation

```bash
# Verify Python package installation
python -c "import space_debris_tracker; print('✓ Package installed successfully')"

# Check GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify CLI tools
space-tracker --help
```

---

## Building the Project

### Building Python Package

#### Development Build

```bash
# Install in editable mode (changes reflect immediately)
pip install -e .
```

#### Production Build

```bash
# Build distribution packages
python setup.py sdist bdist_wheel

# Output will be in dist/
ls -lh dist/
```

### Building Docker Images

#### Build Single Image

```bash
# Build the Docker image
docker build -t space-debris-tracker:latest .

# Verify the image
docker images | grep space-debris-tracker
```

#### Build with Custom Tags

```bash
# Build with version tag
docker build -t space-debris-tracker:1.0.0 .

# Build with multiple tags
docker build -t space-debris-tracker:latest -t space-debris-tracker:1.0.0 .
```

#### Build for Different Architectures

```bash
# For multi-architecture support (AMD64 and ARM64)
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t space-debris-tracker:latest .
```

### Build Optimization Tips

#### Faster Builds with Layer Caching

```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t space-debris-tracker:latest .
```

#### Build with No Cache

```bash
# Force rebuild without cache
docker build --no-cache -t space-debris-tracker:latest .
```

---

## Running the Project

### Option 1: Local Python Execution

#### Start Services Individually

**Terminal 1 - API Server:**
```bash
source venv/bin/activate
python -m space_debris_tracker.api.server

# Or using the CLI entry point
space-api

# Or using uvicorn directly
uvicorn space_debris_tracker.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Dashboard:**
```bash
source venv/bin/activate
streamlit run space_debris_tracker/dashboard/app.py

# Or using the CLI entry point
space-dashboard
```

**Terminal 3 - Background Services (Neo4j):**
```bash
# Start Neo4j in Docker
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j-data:/data \
  neo4j:5.14
```

#### Access Services

- **API Documentation**: http://localhost:8000/docs
- **API ReDoc**: http://localhost:8000/redoc
- **Dashboard**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474

### Option 2: Using Docker Compose (Recommended)

#### Start All Services

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
```

#### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

#### Scale Services

```bash
# Scale API to 5 instances
docker-compose up -d --scale api=5

# Scale dashboard to 3 instances
docker-compose up -d --scale dashboard=3
```

#### Access Services (Docker Compose)

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **Redis**: localhost:6379
- **PostgreSQL**: localhost:5432
- **Kafka**: localhost:9092
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Option 3: Using Individual Docker Containers

```bash
# Start Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5.14

# Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start PostgreSQL
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_DB=space_tracking \
  -e POSTGRES_USER=spaceuser \
  -e POSTGRES_PASSWORD=spacepass \
  postgres:16-alpine

# Start API
docker run -d --name space-api -p 8000:8000 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  --link neo4j --link redis --link postgres \
  space-debris-tracker:latest

# Start Dashboard
docker run -d --name space-dashboard -p 8501:8501 \
  -e API_URL=http://space-api:8000 \
  --link space-api \
  space-debris-tracker:latest \
  streamlit run space_debris_tracker/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

### Generate Sample Data

```bash
# Generate all types of sample data
python scripts/generate_sample_data.py --all

# Generate specific types
python scripts/generate_sample_data.py --type images --count 100
python scripts/generate_sample_data.py --type tle --count 500
python scripts/generate_sample_data.py --type conjunctions --count 50
```

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v             # Performance tests

# Run with coverage
pytest tests/ -v --cov=space_debris_tracker --cov-report=html
open htmlcov/index.html  # View coverage report

# Run specific test file
pytest tests/unit/test_detector.py -v

# Run with markers
pytest -m "not slow" -v                  # Skip slow tests
pytest -m integration -v                 # Only integration tests
```

---

## Docker Deployment

### Production Docker Setup

#### 1. Build Production Image

```bash
# Build optimized production image
docker build -t space-debris-tracker:prod \
  --build-arg BUILD_ENV=production \
  --target runtime \
  .
```

#### 2. Configure Environment

```bash
# Copy and configure production environment
cp .env.example .env.production

# Update production values
nano .env.production
```

#### 3. Start Production Stack

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### 4. Health Checks

```bash
# Check all containers are healthy
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Check container logs
docker-compose logs --tail=100 api
```

### Docker Best Practices

#### Resource Limits

Create `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  neo4j:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

#### Persistent Data

```bash
# Backup Neo4j data
docker exec neo4j neo4j-admin dump --to=/tmp/backup.dump
docker cp neo4j:/tmp/backup.dump ./backup/neo4j-$(date +%Y%m%d).dump

# Backup PostgreSQL
docker exec postgres pg_dump -U spaceuser space_tracking > ./backup/postgres-$(date +%Y%m%d).sql
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Container registry access
- GPU nodes (for ML workloads)

### 1. Push Image to Registry

```bash
# Tag image for registry
docker tag space-debris-tracker:latest your-registry.com/space-debris-tracker:latest

# Push to registry
docker push your-registry.com/space-debris-tracker:latest
```

### 2. Create Namespace

```bash
# Create dedicated namespace
kubectl create namespace space-tracking

# Set as default namespace
kubectl config set-context --current --namespace=space-tracking
```

### 3. Deploy Application

```bash
# Apply all Kubernetes manifests
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml

# Verify deployments
kubectl get deployments -n space-tracking
kubectl get pods -n space-tracking
kubectl get services -n space-tracking
```

### 4. Configure Secrets

```bash
# Create secrets from .env file
kubectl create secret generic space-tracker-secrets \
  --from-env-file=.env.production \
  -n space-tracking

# Verify secrets
kubectl get secrets -n space-tracking
```

### 5. Expose Services

```bash
# Get service external IPs
kubectl get services -n space-tracking

# For LoadBalancer type (cloud environments)
kubectl get svc space-tracker-api -n space-tracking

# For NodePort type (local/on-prem)
kubectl get svc space-tracker-api -n space-tracking -o jsonpath='{.spec.ports[0].nodePort}'
```

### 6. Configure Ingress (Optional)

Create `ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: space-tracker-ingress
  namespace: space-tracking
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.spacedebris.ai
    - dashboard.spacedebris.ai
    secretName: space-tracker-tls
  rules:
  - host: api.spacedebris.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: space-tracker-api
            port:
              number: 80
  - host: dashboard.spacedebris.ai
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

Apply ingress:
```bash
kubectl apply -f ingress.yaml
```

### 7. Monitoring

```bash
# View logs
kubectl logs -f deployment/space-tracker-api -n space-tracking

# View resource usage
kubectl top pods -n space-tracking
kubectl top nodes

# Describe pod for troubleshooting
kubectl describe pod <pod-name> -n space-tracking
```

### 8. Scaling

```bash
# Manual scaling
kubectl scale deployment space-tracker-api --replicas=10 -n space-tracking

# Horizontal Pod Autoscaler (HPA) is already configured in deployment.yaml
kubectl get hpa -n space-tracking

# View autoscaling events
kubectl describe hpa space-tracker-api-hpa -n space-tracking
```

### 9. Updates & Rolling Deployments

```bash
# Update image
kubectl set image deployment/space-tracker-api \
  api=your-registry.com/space-debris-tracker:v1.1.0 \
  -n space-tracking

# Check rollout status
kubectl rollout status deployment/space-tracker-api -n space-tracking

# Rollback if needed
kubectl rollout undo deployment/space-tracker-api -n space-tracking

# View rollout history
kubectl rollout history deployment/space-tracker-api -n space-tracking
```

### 10. Cleanup

```bash
# Delete all resources in namespace
kubectl delete namespace space-tracking

# Or delete specific resources
kubectl delete -f space_debris_tracker/deployment/kubernetes/deployment.yaml
```

---

## Production Deployment

### Pre-Production Checklist

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Secrets properly secured (use Vault, AWS Secrets Manager, etc.)
- [ ] Database backups configured
- [ ] Monitoring and alerting setup
- [ ] SSL/TLS certificates configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Logging configured
- [ ] Performance tested under load
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] Disaster recovery plan in place

### Security Hardening

#### 1. Secure Environment Variables

```bash
# Never commit .env files
echo ".env*" >> .gitignore

# Use secrets management
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name space-tracker/prod/config \
  --secret-string file://.env.production

# HashiCorp Vault
vault kv put secret/space-tracker @.env.production
```

#### 2. Enable HTTPS

```bash
# Generate SSL certificates (Let's Encrypt)
certbot certonly --standalone -d api.spacedebris.ai

# Configure in Nginx/Traefik/Ingress
```

#### 3. Configure Firewall

```bash
# Allow only necessary ports
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 22/tcp    # SSH (restrict to specific IPs)
ufw enable
```

#### 4. Database Security

```bash
# PostgreSQL - Create read-only user
psql -U postgres
CREATE USER readonly WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE space_tracking TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;

# Neo4j - Enable authentication
# Edit neo4j.conf
dbms.security.auth_enabled=true
```

### Performance Tuning

#### 1. Database Optimization

**Neo4j:**
```bash
# Edit neo4j.conf
dbms.memory.heap.initial_size=4G
dbms.memory.heap.max_size=8G
dbms.memory.pagecache.size=4G
```

**PostgreSQL:**
```bash
# Edit postgresql.conf
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 10485kB
min_wal_size = 1GB
max_wal_size = 4GB
```

#### 2. Application Optimization

```bash
# Enable model compilation (PyTorch 2.0+)
export TORCH_COMPILE=1

# Enable mixed precision
export MIXED_PRECISION=true

# Configure workers
export API_WORKERS=8
```

#### 3. Load Balancing

Use Nginx as reverse proxy:

```nginx
upstream space_tracker_api {
    least_conn;
    server api1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api3:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.spacedebris.ai;

    location / {
        proxy_pass http://space_tracker_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Backup Strategy

```bash
# Automated daily backups
crontab -e

# Add these lines:
# Daily Neo4j backup at 2 AM
0 2 * * * docker exec neo4j neo4j-admin dump --to=/backups/neo4j-$(date +\%Y\%m\%d).dump

# Daily PostgreSQL backup at 2:30 AM
30 2 * * * docker exec postgres pg_dump -U spaceuser space_tracking | gzip > /backups/postgres-$(date +\%Y\%m\%d).sql.gz

# Weekly cleanup (keep last 7 days)
0 3 * * 0 find /backups -name "*.dump" -mtime +7 -delete
```

---

## Monitoring & Observability

### Prometheus Metrics

#### Access Metrics

```bash
# View application metrics
curl http://localhost:8000/metrics

# View Prometheus UI
open http://localhost:9090
```

#### Key Metrics to Monitor

- `detection_latency_seconds` - Detection inference time
- `prediction_latency_seconds` - Trajectory prediction time
- `conjunction_probability` - Current collision probabilities
- `api_request_duration_seconds` - API response times
- `gpu_utilization_percent` - GPU usage
- `kafka_lag` - Message processing lag

#### Custom Alerts

Create `monitoring/prometheus-rules/alerts.yml`:

```yaml
groups:
- name: space_tracker_alerts
  rules:
  - alert: HighAPILatency
    expr: api_request_duration_seconds{quantile="0.95"} > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      description: "API P95 latency is {{ $value }}s"

  - alert: HighCollisionProbability
    expr: conjunction_probability > 0.001
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High collision probability detected"
      description: "Collision probability: {{ $value }}"

  - alert: GPUMemoryHigh
    expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage high"
```

### Grafana Dashboards

#### Access Grafana

```bash
# Open Grafana
open http://localhost:3000

# Login: admin/admin (change on first login)
```

#### Import Dashboards

1. Go to **Dashboards** → **Import**
2. Use ID: `1860` for Node Exporter dashboard
3. Use ID: `3662` for PostgreSQL dashboard
4. Custom dashboard JSON in `monitoring/grafana-dashboards/`

### Application Logging

#### Configure Logging

Edit `.env`:
```bash
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/space_debris_tracker.log
```

#### View Logs

```bash
# Local logs
tail -f logs/space_debris_tracker.log

# Docker logs
docker-compose logs -f --tail=100 api

# Kubernetes logs
kubectl logs -f deployment/space-tracker-api -n space-tracking

# Logs with JSON parsing
tail -f logs/space_debris_tracker.log | jq .
```

#### Centralized Logging (ELK Stack)

```bash
# Add to docker-compose.yml
elasticsearch:
  image: elasticsearch:8.11.0
  ports:
    - "9200:9200"
  environment:
    - discovery.type=single-node

logstash:
  image: logstash:8.11.0
  volumes:
    - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
  depends_on:
    - elasticsearch

kibana:
  image: kibana:8.11.0
  ports:
    - "5601:5601"
  depends_on:
    - elasticsearch
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'space_debris_tracker'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install -e .
```

#### 2. CUDA/GPU Issues

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# 1. Reduce batch size in config.yaml
batch_size: 4  # Instead of 16

# 2. Enable gradient accumulation
gradient_accumulation_steps: 4

# 3. Use mixed precision
mixed_precision: true

# 4. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Problem:**
```
CUDA not available
```

**Solutions:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Database Connection Issues

**Problem:**
```
Neo4jError: Could not connect to bolt://localhost:7687
```

**Solutions:**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Start Neo4j
docker-compose up -d neo4j

# Check Neo4j logs
docker-compose logs neo4j

# Test connection
docker exec neo4j cypher-shell -u neo4j -p password "RETURN 1"
```

#### 4. Port Already in Use

**Problem:**
```
OSError: [Errno 48] Address already in use: ('0.0.0.0', 8000)
```

**Solutions:**
```bash
# Find process using port
lsof -i :8000
# OR
netstat -tulpn | grep 8000

# Kill process
kill -9 <PID>

# Use different port
API_PORT=8001 python -m space_debris_tracker.api.server
```

#### 5. Docker Build Failures

**Problem:**
```
failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solutions:**
```bash
# Build with more verbose output
docker build --progress=plain -t space-debris-tracker:latest .

# Check requirements.txt for errors
cat requirements.txt

# Build without cache
docker build --no-cache -t space-debris-tracker:latest .

# Check disk space
df -h
```

#### 6. Kubernetes Pod CrashLoopBackOff

**Problem:**
```
space-tracker-api-7d6b8c9f4d-x7z8q   0/1     CrashLoopBackOff   5          3m
```

**Solutions:**
```bash
# Check pod logs
kubectl logs <pod-name> -n space-tracking

# Describe pod for events
kubectl describe pod <pod-name> -n space-tracking

# Check resource limits
kubectl get pod <pod-name> -n space-tracking -o yaml | grep -A 5 resources

# Check secrets/configmaps
kubectl get secrets -n space-tracking
kubectl get configmaps -n space-tracking
```

#### 7. Performance Issues

**Problem:** Slow API responses

**Solutions:**
```bash
# 1. Check system resources
docker stats

# 2. Enable Redis caching
ENABLE_REDIS_CACHE=true

# 3. Increase workers
API_WORKERS=8

# 4. Profile the application
python -m cProfile -o profile.stats -m space_debris_tracker.api.server
python -m pstats profile.stats

# 5. Check database query performance
# Neo4j
PROFILE MATCH (n) RETURN n LIMIT 10;

# PostgreSQL
EXPLAIN ANALYZE SELECT * FROM satellites;
```

### Getting Help

#### Check Documentation
- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines

#### Community Support
- **GitHub Issues**: https://github.com/Srujan29112001/Space-debrey/issues
- **Discussions**: https://github.com/Srujan29112001/Space-debrey/discussions

#### Enable Debug Mode

```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG

# Run with verbose logging
python -m space_debris_tracker.api.server --log-level debug
```

---

## Performance Optimization

### Benchmarking

```bash
# API load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:8000/api/v1/satellites/25544

# API load testing with wrk
wrk -t 4 -c 100 -d 30s http://localhost:8000/api/v1/satellites/25544

# Run performance tests
pytest tests/performance/ -v --benchmark
```

### Optimization Checklist

- [ ] Enable Redis caching
- [ ] Use connection pooling
- [ ] Enable model compilation (PyTorch 2.0+)
- [ ] Use mixed precision training/inference
- [ ] Optimize database queries (indexes, EXPLAIN)
- [ ] Enable CDN for static assets
- [ ] Use asynchronous endpoints where possible
- [ ] Implement request batching
- [ ] Enable gzip compression
- [ ] Configure proper worker/thread counts
- [ ] Use GPU acceleration for ML models
- [ ] Implement API response caching
- [ ] Optimize Docker image size
- [ ] Use horizontal pod autoscaling (K8s)

### Resource Monitoring

```bash
# Monitor system resources
htop

# Monitor GPU
nvidia-smi -l 1

# Monitor Docker resources
docker stats

# Monitor Kubernetes resources
kubectl top nodes
kubectl top pods -n space-tracking
```

---

## Appendix

### Quick Reference Commands

#### Local Development
```bash
source venv/bin/activate
python -m space_debris_tracker.api.server
streamlit run space_debris_tracker/dashboard/app.py
pytest tests/ -v
```

#### Docker
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

#### Kubernetes
```bash
kubectl apply -f space_debris_tracker/deployment/kubernetes/deployment.yaml
kubectl get pods -n space-tracking
kubectl logs -f deployment/space-tracker-api -n space-tracking
kubectl scale deployment space-tracker-api --replicas=5 -n space-tracking
```

### Environment Variables Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8000 | API server port |
| `NEO4J_URI` | bolt://localhost:7687 | Neo4j connection URI |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device IDs |
| `LOG_LEVEL` | INFO | Logging level |
| `BATCH_SIZE` | 16 | Model batch size |
| `REDIS_HOST` | localhost | Redis host |

### Port Reference

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| API | 8000 | HTTP | REST/GraphQL API |
| Dashboard | 8501 | HTTP | Streamlit dashboard |
| Neo4j Browser | 7474 | HTTP | Neo4j web interface |
| Neo4j Bolt | 7687 | Bolt | Neo4j protocol |
| Redis | 6379 | TCP | Cache/message broker |
| PostgreSQL | 5432 | TCP | Relational database |
| Kafka | 9092 | TCP | Message streaming |
| Prometheus | 9090 | HTTP | Metrics |
| Grafana | 3000 | HTTP | Dashboards |

### Useful Links

- **Project Repository**: https://github.com/Srujan29112001/Space-debrey
- **PyTorch**: https://pytorch.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Neo4j**: https://neo4j.com/docs/
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/

---

## Summary

This guide covered:

✅ Complete prerequisites and system requirements
✅ Local development setup (automated and manual)
✅ Building the project (Python, Docker)
✅ Running the project (local, Docker, Kubernetes)
✅ Production deployment best practices
✅ Monitoring and observability setup
✅ Comprehensive troubleshooting guide
✅ Performance optimization strategies

**Next Steps:**

1. Choose your deployment method (local, Docker, or Kubernetes)
2. Follow the setup instructions for your chosen method
3. Generate sample data and run tests
4. Access the API documentation and dashboard
5. Set up monitoring and alerts
6. Optimize for your specific use case

**Need Help?** Open an issue on GitHub or check the troubleshooting section.

---

**Happy Deploying! 🚀🛰️**
