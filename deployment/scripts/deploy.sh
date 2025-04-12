#!/bin/bash

# Configuration
DOCKER_REGISTRY="your-registry.azurecr.io"
IMAGE_NAME="hmas/core"
IMAGE_TAG=$(git rev-parse --short HEAD)
NAMESPACE="hmas-system"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        log_warn "Helm is not installed - some features may not be available"
    fi
}

# Build and push Docker image
build_and_push() {
    log_info "Building Docker image..."
    docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
    docker tag ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
    
    log_info "Pushing Docker image..."
    docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace if it doesn't exist..."
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
}

# Apply Kubernetes configurations
apply_configs() {
    log_info "Applying Kubernetes configurations..."
    
    # Apply configurations in order
    kubectl apply -f deployment/k8s/configmap.yaml
    kubectl apply -f deployment/k8s/secrets.yaml
    kubectl apply -f deployment/k8s/storage.yaml
    kubectl apply -f deployment/k8s/deployment.yaml
    kubectl apply -f deployment/k8s/service.yaml
    kubectl apply -f deployment/k8s/hpa.yaml
    kubectl apply -f deployment/k8s/monitoring.yaml
    kubectl apply -f deployment/k8s/ingress.yaml
    
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/hmas-core -n ${NAMESPACE}
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    if command -v helm &> /dev/null; then
        # Add Prometheus repository
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        # Install Prometheus and Grafana
        helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --set grafana.enabled=true \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
    else
        log_warn "Helm not found - skipping monitoring setup"
    fi
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n ${NAMESPACE}
    
    # Check service status
    kubectl get svc -n ${NAMESPACE}
    
    # Check ingress status
    kubectl get ingress -n ${NAMESPACE}
}

# Main deployment process
main() {
    log_info "Starting deployment process..."
    
    # Check prerequisites
    check_prerequisites
    
    # Build and push Docker image
    build_and_push
    
    # Create namespace
    create_namespace
    
    # Apply Kubernetes configurations
    apply_configs
    
    # Setup monitoring
    setup_monitoring
    
    # Verify deployment
    verify_deployment
    
    log_info "Deployment completed successfully!"
}

# Run main function
main 