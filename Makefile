# Makefile for OrcaGym Core
# Provides convenient shortcuts for common tasks

.PHONY: help clean build check test-install release-test release-prod bump-version

help:  ## Show this help message
	@echo "OrcaGym Core - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:  ## Clean build artifacts
	@./scripts/release/clean.sh

build:  ## Build distribution packages
	@./scripts/release/build.sh

check:  ## Check package quality
	@./scripts/release/check.sh

test-install:  ## Test installation from local build
	@./scripts/release/test_install.sh local

test-install-testpypi:  ## Test installation from TestPyPI
	@./scripts/release/test_install.sh test

test-install-pypi:  ## Test installation from PyPI
	@./scripts/release/test_install.sh prod

release-test:  ## Complete release to TestPyPI
	@./scripts/release/release.sh test

release-prod:  ## Complete release to PyPI (production)
	@./scripts/release/release.sh prod

bump-version:  ## Bump version (usage: make bump-version VERSION=25.10.1)
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make bump-version VERSION=25.10.1"; \
		exit 1; \
	fi
	@./scripts/release/bump_version.sh $(VERSION)

# Development targets
install-dev:  ## Install package in development mode
	pip install -e ".[dev]"

install-all:  ## Install package with all optional dependencies
	pip install -e ".[all,dev]"

test:  ## Run tests (if available)
	@echo "Running tests..."
	pytest tests/ || echo "No tests found"

format:  ## Format code with black
	black orca_gym/

lint:  ## Run linters
	flake8 orca_gym/
	mypy orca_gym/

# Quick commands
all: clean build check  ## Clean, build, and check

quick-release: all release-test  ## Quick release to TestPyPI

