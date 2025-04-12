"""
Branding and naming constants for the HMAS project.
"""

# Project Names
PROJECT_NAME = "HMAS"
PROJECT_FULL_NAME = "Hierarchical Multi-Agent System"
AGI_NAME = "DREAMv1"
AGI_FULL_NAME = "Dynamic Reasoning and Evolving Autonomous Mind v1"

# Organization
ORGANIZATION_NAME = "HMAS"
ORGANIZATION_FULL_NAME = "Hierarchical Multi-Agent Systems"
ORGANIZATION_DOMAIN = "hmas.ai"

# Version Information
VERSION = "1.0.0"
BUILD = "alpha"

# Contact Information
CONTACT_EMAIL = "contact@hmas.ai"
SUPPORT_EMAIL = "support@hmas.ai"
SECURITY_EMAIL = "security@hmas.ai"

# URLs
WEBSITE_URL = "https://hmas.ai"
DOCS_URL = "https://docs.hmas.ai"
API_URL = "https://api.hmas.ai"

# Repository
REPO_URL = "https://github.com/hmas/hmas"
REPO_BRANCH = "main"

# Docker
DOCKER_REGISTRY = "ghcr.io/hmas"
DOCKER_NAMESPACE = "hmas"

# Kubernetes
K8S_NAMESPACE = "hmas"
K8S_LABEL_PREFIX = "hmas.ai"

# Documentation
COPYRIGHT = f"© {ORGANIZATION_NAME}. All rights reserved."
LICENSE = "Dual License (Commercial/Open Source)"

# System Identifiers
SYSTEM_PREFIX = "HMAS"
AGI_SYSTEM_NAME = f"{SYSTEM_PREFIX}_{AGI_NAME}"

def get_service_name(service: str) -> str:
    """Get the standardized name for a service."""
    return f"{SYSTEM_PREFIX.lower()}-{service}"

def get_container_name(service: str) -> str:
    """Get the standardized container name for a service."""
    return f"{DOCKER_NAMESPACE}/{service}:latest"

def get_k8s_name(resource: str) -> str:
    """Get the standardized Kubernetes resource name."""
    return f"{SYSTEM_PREFIX.lower()}-{resource}" 