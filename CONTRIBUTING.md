# Contributing to Space Debris Tracking System

First off, thank you for considering contributing to our project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python style guide (PEP 8)
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Development Setup

1. Fork the repo
2. Clone your fork
3. Create a virtual environment
4. Install dependencies: `pip install -r requirements-dev.txt`
5. Install pre-commit hooks: `pre-commit install`
6. Create a new branch: `git checkout -b my-feature-branch`
7. Make your changes
8. Run tests: `pytest`
9. Run linters: `black . && flake8 . && mypy .`
10. Commit your changes
11. Push to your fork
12. Open a Pull Request

## Code Style

* Follow PEP 8
* Use type hints
* Write docstrings for all public methods
* Keep functions small and focused
* Use meaningful variable names

## Testing

* Write unit tests for new features
* Ensure all tests pass before submitting PR
* Aim for >80% code coverage
* Use pytest fixtures for common test setup

## Documentation

* Update README.md if needed
* Add docstrings to new functions/classes
* Update API documentation
* Add examples for new features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
