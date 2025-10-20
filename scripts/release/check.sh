#!/bin/bash
# Check distribution packages
# Usage: ./scripts/release/check.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "🔍 Checking distribution packages..."

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "⚠️  'twine' not found. Installing..."
    pip install twine
fi

# Check if dist/ exists
if [ ! -d "dist" ]; then
    echo "❌ Error: dist/ directory not found. Run build.sh first."
    exit 1
fi

# Check packages
twine check dist/*

echo ""
echo "✅ All checks passed!"

