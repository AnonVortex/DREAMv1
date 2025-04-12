# Deployment Guide

## Overview

This guide provides detailed instructions for deploying the Hierarchical Multi-Agent System (HMAS) in various environments, from development to production.

## Prerequisites

### Required Tools
- Docker 20.10+
- Kubernetes 1.21+
- Helm 3.0+ (optional, for monitoring)
- Python 3.9+
- kubectl CLI
- Container registry access

### System Requirements
- Minimum 4 CPU cores
- 8GB RAM
- 50GB storage
- Network access for container registry

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hmas.git
cd hmas
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Docker Configuration

### Building the Image

1. Navigate to deployment directory:
```bash
cd deployment
```

2. Build the image:
```bash
docker build -t hmas/core:latest .
```

### Image Configuration
- Base image: python:3.9-slim
- Working directory: /app
- Exposed ports: 8000
- Volume mounts: /app/data, /app/logs, /app/checkpoints
- Non-root user: hmas

## Kubernetes Deployment

### Namespace Setup
```yaml
# deployment/k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hmas-system
  labels:
    name: hmas-system
    environment: production
```

### Configuration
```yaml
# deployment/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hmas-config
  namespace: hmas-system
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  META_LEARNING_ENABLED: "true"
  MONITORING_ENABLED: "true"
  MAX_WORKERS: "4"
  CHECKPOINT_INTERVAL: "300"
```

### Core Deployment
```yaml
# deployment/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hmas-core
  namespace: hmas-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hmas-core
  template:
    metadata:
      labels:
        app: hmas-core
    spec:
      containers:
      - name: hmas-core
        image: hmas/core:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Deployment Process

### 1. Preparation
- Update configuration files with your environment settings
- Ensure access to container registry
- Verify Kubernetes cluster access

### 2. Deployment Script
The `deploy.sh` script in `deployment/scripts/` handles the deployment process:
- Checks prerequisites
- Builds and pushes Docker image
- Creates namespace
- Applies Kubernetes configurations
- Sets up monitoring
- Verifies deployment

### 3. Execution
```bash
./deployment/scripts/deploy.sh
```

## Monitoring Setup

### Prometheus & Grafana
1. Add Helm repository:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

2. Install monitoring stack:
```bash
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --set grafana.enabled=true
```

### Metrics Collection
- System metrics via Prometheus
- Custom metrics via API endpoints
- Resource utilization tracking
- Performance monitoring

## Security Configuration

### 1. Authentication
- JWT token configuration
- Role-based access control
- API key management
- Session handling

### 2. Network Security
- TLS configuration
- Network policies
- Ingress rules
- Rate limiting

### 3. Resource Protection
- Pod security policies
- Resource quotas
- Security contexts
- Access controls

## Scaling Configuration

### 1. Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hmas-hpa
  namespace: hmas-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hmas-core
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. Resource Management
- CPU/Memory requests and limits
- Storage configuration
- Network resources
- Cache settings

## Maintenance

### 1. Backup Procedures
- Database backups
- Configuration backups
- State persistence
- Recovery procedures

### 2. Updates
- Rolling updates strategy
- Version control
- Rollback procedures
- Database migrations

### 3. Monitoring
- Log aggregation
- Metrics collection
- Alert configuration
- Performance tracking

## Troubleshooting

### Common Issues
1. Pod startup failures
   - Check logs: `kubectl logs -n hmas-system <pod-name>`
   - Verify resources: `kubectl describe pod -n hmas-system <pod-name>`

2. Configuration issues
   - Verify ConfigMap: `kubectl get configmap -n hmas-system hmas-config -o yaml`
   - Check environment variables

3. Network issues
   - Test connectivity: `kubectl exec -n hmas-system <pod-name> -- curl -v localhost:8000/health`
   - Verify service: `kubectl get svc -n hmas-system`

### Debug Tools
- kubectl debug
- Port forwarding
- Log streaming
- Exec into containers

## Production Considerations

### 1. High Availability
- Multi-zone deployment
- Load balancing
- Failover configuration
- Backup strategies

### 2. Performance
- Resource optimization
- Cache configuration
- Network optimization
- Storage performance

### 3. Monitoring
- Alert configuration
- Dashboard setup
- Log aggregation
- Performance tracking

## Cleanup

### Resource Removal
```bash
# Remove deployment
kubectl delete -f deployment/k8s/

# Remove namespace
kubectl delete namespace hmas-system

# Remove monitoring
helm uninstall prometheus -n monitoring
```

### Data Cleanup
- Database cleanup
- Storage cleanup
- Cache clearing
- Log rotation 
