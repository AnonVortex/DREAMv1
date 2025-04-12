# Contributing to DREAMv1

## License

For commercial licensing inquiries, please contact: [info@hmas.ai]

## Overview

Thank you for your interest in contributing to the Hierarchical Multi-Agent System (HMAS) with DREAMv1! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](./CODE_OF_CONDUCT.md) to maintain a positive and inclusive community.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Set up the development environment
4. Create a new branch for your feature/fix

## Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hmas.git
cd hmas

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: description of your changes"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Maintenance tasks

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=hmas tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions/methods
- Keep functions/methods focused and concise

## Documentation

- Update relevant documentation for your changes
- Include docstrings for new functions/classes
- Update the changelog if applicable
- Add examples for new features

## Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review from maintainers
5. Address review feedback

## Release Process

1. Version bump following semver
2. Update changelog
3. Create release PR
4. Tag release
5. Deploy to production

## Getting Help

- Open an issue for bugs/features
- Join our community chat
- Check existing documentation
- Contact maintainers

## Project Structure

```
hmas/
├── docs/              # Documentation
├── examples/          # Example code
├── hmas/             # Main package
│   ├── perception/   # Perception service
│   ├── memory/       # Memory service
│   ├── learning/     # Learning service
│   ├── reasoning/    # Reasoning service
│   ├── communication/# Communication service
│   ├── feedback/     # Feedback service
│   ├── integration/  # Integration service
│   └── specialized/  # Specialized service
├── tests/            # Test suite
├── deployment/       # Deployment configs
└── scripts/         # Utility scripts
```

## Additional Resources

- [Architecture Documentation](./Architecture.md)
- [API Documentation](./api.md)
- [SDK Documentation](./sdk.md)
- [Changelog](./changelog.md) 
