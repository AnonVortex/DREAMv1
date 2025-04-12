# DREAMv1 Deployment Guide

## Overview
This guide covers the deployment process for DREAMv1 in both development and production environments.

## Prerequisites

### System Requirements
- CPU: 8+ cores
- RAM: 16GB+ (32GB recommended)
- Storage: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (for ML workloads)

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.22+
- Helm 3.0+
- Python 3.8+
- NVIDIA Container Toolkit (for GPU support)

## Local Development Deployment

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/AnonVortex/DREAMv1.git
cd hmas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Services
```bash
# Build and start all services
docker-compose up -d

# Start specific service
docker-compose up -d [service_name]

# View logs
docker-compose logs -f [service_name]
```

### 3. Development Tools
```bash
# Start development shell
docker-compose exec [service_name] bash

# Run tests
docker-compose exec [service_name] pytest

# Check logs
docker-compose logs -f
```

## Production Deployment

### 1. Infrastructure Setup

#### Kubernetes Cluster
```bash
# Create namespace
kubectl create namespace hmas

# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

#### Storage
```bash
# Create persistent volumes
kubectl apply -f k8s/storage/

# Setup MongoDB
helm install mongodb bitnami/mongodb -n hmas

# Setup Redis
helm install redis bitnami/redis -n hmas
```

#### Monitoring
```bash
# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack -n hmas

# Install Grafana
kubectl apply -f k8s/monitoring/grafana.yaml
```

### 2. Service Deployment

#### Build Images
```bash
# Build all services
docker-compose build

# Build specific service
docker build -t hmas/[service] -f [service]/Dockerfile .

# Push to registry
docker push hmas/[service]:version
```

#### Deploy Services
```bash
# Apply configurations
kubectl apply -f k8s/config/

# Deploy core services
kubectl apply -f k8s/core/

# Deploy supporting services
kubectl apply -f k8s/support/
```

### 3. Configuration

#### Environment Variables
```bash
# Create secrets
kubectl create secret generic hmas-secrets \
  --from-file=.env \
  -n hmas

# Create configmaps
kubectl apply -f k8s/config/configmaps.yaml
```

#### Service Configuration
```yaml
# k8s/core/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: hmas-service
spec:
  ports:
    - port: 80
      targetPort: 8000
  selector:
    app: hmas
```

### 4. Security

#### TLS Configuration
```bash
# Install cert-manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.7.1

# Apply TLS configuration
kubectl apply -f k8s/security/tls/
```

#### Network Policies
```yaml
# k8s/security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hmas-network-policy
spec:
  podSelector:
    matchLabels:
      app: hmas
  policyTypes:
    - Ingress
    - Egress
```

## Scaling

### Horizontal Scaling
```bash
# Scale deployment
kubectl scale deployment [deployment] --replicas=3 -n hmas

# Configure autoscaling
kubectl autoscale deployment [deployment] \
  --min=2 \
  --max=5 \
  --cpu-percent=80
```

### Resource Management
```yaml
# k8s/core/deployment.yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

## Monitoring

### Prometheus Configuration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'hmas'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboards
- System metrics dashboard
- Service performance dashboard
- Learning metrics dashboard
- Agent activity dashboard

## Maintenance

### Backup Procedures
```bash
# Backup MongoDB
kubectl exec -it [mongodb-pod] -- mongodump \
  --out=/backup/$(date +%Y%m%d)

# Backup Redis
kubectl exec -it [redis-pod] -- redis-cli save
```

### Updates
```bash
# Update deployments
kubectl set image deployment/[deployment] \
  container=[new-image] -n hmas

# Rolling updates
kubectl rollout status deployment/[deployment]
```

### Health Checks
```bash
# Check service health
kubectl get pods -n hmas
kubectl describe pod [pod-name]
kubectl logs [pod-name]
```

## Troubleshooting

### Common Issues
1. Pod startup failures:
   - Check logs: `kubectl logs [pod]`
   - Check events: `kubectl get events`
   - Verify resources: `kubectl describe pod [pod]`

2. Service connectivity:
   - Check service: `kubectl get svc`
   - Test DNS: `kubectl exec -it [pod] -- nslookup [service]`
   - Verify endpoints: `kubectl get endpoints`

3. Resource issues:
   - Check usage: `kubectl top pods`
   - View metrics: `kubectl get hpa`
   - Monitor nodes: `kubectl describe nodes`

### Debug Tools
```bash
# Interactive debugging
kubectl debug [pod] -it --image=busybox

# Port forwarding
kubectl port-forward [pod] 8000:8000

# Exec into container
kubectl exec -it [pod] -- bash
```

## Rollback Procedures

### Deployment Rollback
```bash
# View rollout history
kubectl rollout history deployment/[deployment]

# Rollback to previous version
kubectl rollout undo deployment/[deployment]

# Rollback to specific version
kubectl rollout undo deployment/[deployment] \
  --to-revision=[revision]
```

### Database Rollback
```bash
# Restore MongoDB backup
kubectl exec -it [mongodb-pod] -- mongorestore \
  /backup/[backup-date]

# Restore Redis backup
kubectl cp [backup-file] [redis-pod]:/data/dump.rdb
kubectl exec -it [redis-pod] -- redis-cli BGREWRITEAOF
```

## Performance Optimization

### Caching Strategy
- Redis cache configuration
- Cache invalidation policies
- Cache monitoring

### Database Optimization
- Index optimization
- Query performance
- Connection pooling

### Network Optimization
- Service mesh configuration
- Load balancer settings
- Network policies
