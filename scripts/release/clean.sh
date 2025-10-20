#!/bin/bash
# Clean build artifacts
# Usage: ./scripts/release/clean.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "🧹 Cleaning build artifacts..."

# Remove build directories
rm -rf build/
rm -rf dist/
rm -rf dist-test/
rm -rf *.egg-info
rm -rf .eggs/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.egg" -delete

echo "✅ Clean completed!"

