# Contributing to DREAMv1

Thank you for your interest in contributing to H-MAS! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Workflow

1. **Fork the Repository**
   - Fork the repository to your GitHub account
   - Clone your fork locally
   ```bash
   git clone https://github.com/AnonVortex/DREAMv1.git
   cd h-mas
   ```

2. **Set Up Development Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**
   - Follow the coding style guidelines
   - Write tests for new features
   - Update documentation as needed

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

6. **Push Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Open a pull request from your fork to the main repository
   - Follow the pull request template
   - Wait for review and address any feedback

## Coding Standards

### Python Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function parameters and return values
- Document all public functions and classes with docstrings
- Keep functions small and focused on a single responsibility

### Testing
- Write tests for all new features
- Maintain test coverage above 80%
- Use appropriate test markers (unit, integration, e2e)
- Follow the testing guidelines in `tests/README.md`

### Documentation
- Update relevant documentation when making changes
- Include docstrings for all public APIs
- Add examples in docstrings where appropriate
- Keep the README and other documentation up to date

## Pull Request Process

1. **Title and Description**
   - Use clear, descriptive titles
   - Follow the pull request template
   - Include relevant issue numbers

2. **Code Review**
   - Address all review comments
   - Keep the PR focused and manageable
   - Ensure all tests pass

3. **Merge Requirements**
   - All tests must pass
   - Code coverage must not decrease
   - Documentation must be updated
   - Code must be reviewed and approved

## Issue Reporting

When reporting issues:
- Use the issue template
- Provide detailed reproduction steps
- Include relevant logs and error messages
- Specify the version of the software

## Feature Requests

For feature requests:
- Explain the use case
- Describe the expected behavior
- Provide examples if possible
- Consider contributing the feature yourself

## Questions and Support

For questions and support:
- Check the documentation first
- Search existing issues
- Open a new issue if needed
- Join our community chat

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create release tag
6. Build and publish packages
7. Update documentation

## License

By contributing, you agree that your contributions will be licensed under the project's [Dual-License](LICENSE). 
