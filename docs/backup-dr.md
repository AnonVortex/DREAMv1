# Backup and Disaster Recovery Procedures

## Overview

This document outlines the backup and disaster recovery procedures for the HMAS system. The system uses automated workflows for both backup and restoration processes, ensuring data safety and business continuity.

## Backup Procedure

### Automated Daily Backups

The system performs automated daily backups at 2 AM UTC using GitHub Actions. The backup workflow (`backup.yml`) performs the following:

1. **Kubernetes State Backup**
   - Uses Velero to backup the entire `hmas` namespace
   - Includes all deployments, services, and configurations
   - Stored in S3 with timestamp

2. **MongoDB Backup**
   - Full database dump using `mongodump`
   - Compressed and stored in S3
   - Includes all collections and indexes

3. **Redis Backup**
   - RDB snapshot using `SAVE` command
   - Stored in S3 with timestamp
   - Includes all keys and data structures

4. **Configuration Backup**
   - Exports all ConfigMaps and Secrets
   - Stores all custom resources
   - Compressed and stored in S3

### Retention Policy

- Backups are retained for 30 days
- Automated cleanup of older backups
- Each backup is tagged with timestamp (YYYYMMDD-HHMMSS)

### Backup Verification

- Automated verification after each backup
- Slack notifications for backup status
- Logs stored for audit purposes

## Disaster Recovery Procedure

### Prerequisites

1. AWS credentials with access to backup bucket
2. Kubernetes cluster access
3. GitHub Actions access

### Recovery Types

1. **Full System Recovery**
   - Restores entire system state
   - Use `restore_type: all`

2. **Selective Recovery**
   - Kubernetes resources only: `restore_type: k8s`
   - MongoDB only: `restore_type: mongodb`
   - Redis only: `restore_type: redis`
   - Configurations only: `restore_type: config`

### Recovery Process

1. **Initiation**
   - Go to GitHub Actions
   - Select "Disaster Recovery" workflow
   - Input backup timestamp and restore type
   - Trigger workflow

2. **Automated Recovery Steps**
   - Scale down existing deployments
   - Restore selected components
   - Verify restoration
   - Scale up deployments
   - Run health checks

3. **Verification**
   - System health checks
   - Integration tests
   - Service availability checks

### Post-Recovery Tasks

1. **Verification**
   - Check all service endpoints
   - Verify data consistency
   - Monitor system metrics

2. **Documentation**
   - Record incident details
   - Document recovery time
   - Note any issues encountered

## Emergency Contacts

- Primary DevOps: [Contact Information]
- Secondary DevOps: [Contact Information]
- System Administrator: [Contact Information]

## Required Secrets

The following secrets must be configured in GitHub:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region
- `BACKUP_BUCKET`: S3 bucket name
- `KUBECONFIG`: Kubernetes configuration
- `MONGODB_USERNAME`: MongoDB admin username
- `MONGODB_PASSWORD`: MongoDB admin password
- `REDIS_PASSWORD`: Redis password
- `SLACK_WEBHOOK_URL`: Slack notification webhook

## Testing and Maintenance

### Regular Testing

1. **Monthly Recovery Tests**
   - Schedule test recoveries
   - Verify all components
   - Document results

2. **Backup Validation**
   - Weekly backup integrity checks
   - Random restoration tests
   - Data consistency verification

### Maintenance

1. **Workflow Updates**
   - Regular review of workflows
   - Update dependencies
   - Security patches

2. **Documentation Updates**
   - Keep contact information current
   - Update procedures as needed
   - Record lessons learned

## Troubleshooting

### Common Issues

1. **Backup Failures**
   - Check S3 permissions
   - Verify pod access
   - Check storage space

2. **Recovery Failures**
   - Verify backup existence
   - Check resource availability
   - Review error logs

### Recovery Time Objectives (RTO)

- Full system recovery: < 1 hour
- Individual component recovery: < 30 minutes
- Configuration recovery: < 15 minutes

### Recovery Point Objectives (RPO)

- Maximum data loss: 24 hours
- Recommended recovery point: Latest successful backup

## Cloud-Specific Deployment Instructions

- Add cloud-specific deployment instructions
- Update troubleshooting guides
- Document rollback procedures
- Add monitoring dashboard setup 

1. Consolidate Configuration:
   - Create unified config system
   - Remove redundant configs
   - Implement environment-specific settings

2. Standardize Service Layer:
   - Implement base service class
   - Update all services to use base class
   - Add consistent error handling

3. Environment Setup:
   - Complete environment variable documentation
   - Create environment-specific configurations
   - Set up cloud provider secrets 

import os
os.environ["HMAS_DEBUG"] = "true"
os.environ["HMAS_DB_HOST"] = "custom-host"

env = EnvironmentManager("test-service")
config = env.get_config()
assert config.debug == True
assert config.database.host == "custom-host"

# Test development URLs
dev_env = EnvironmentManager("test-service", "development")
assert dev_env.get_service_url("memory") == "http://localhost:8000"

# Test production URLs
prod_env = EnvironmentManager("test-service", "production")
assert prod_env.get_service_url("memory") == "http://memory"

from shared.environment import EnvironmentConfig
from pydantic import ValidationError

# Test required fields
try:
    EnvironmentConfig(
        env_type="development",
        service_name="test",
        port=8000,
        database={},  # Missing required fields
        redis={},     # Missing required fields
        security={}   # Missing required fields
    )
    assert False, "Should raise ValidationError"
except ValidationError:
    pass 