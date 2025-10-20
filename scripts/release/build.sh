#!/bin/bash
# Build distribution packages
# Usage: ./scripts/release/build.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "📦 Building distribution packages..."

# Check if build tool is installed
if ! command -v python -m build &> /dev/null; then
    echo "⚠️  'build' not found. Installing..."
    pip install build
fi

# Build the package (official name)
python -m build

echo ""
echo "📦 Building TestPyPI package (orca-gym-test)..."

# Prepare temp workspace for test package build
TEST_OUTDIR="$PROJECT_ROOT/dist-test"
TMP_DIR=$(mktemp -d -t orcagym-testbuild-XXXX)

# Copy source excluding build artifacts and VCS files
rsync -a \
    --exclude '.git' \
    --exclude '.gitignore' \
    --exclude 'dist' \
    --exclude 'dist-test' \
    --exclude 'build' \
    --exclude '*.egg-info' \
    "$PROJECT_ROOT/" "$TMP_DIR/"

# Modify package name in pyproject.toml to orca-gym-test for TestPyPI build
if grep -q 'name = "orca-gym"' "$TMP_DIR/pyproject.toml"; then
    sed -i 's/name = "orca-gym"/name = "orca-gym-test"/' "$TMP_DIR/pyproject.toml"
else
    echo "⚠️  Unable to find package name in pyproject.toml for test build; leaving as-is."
fi

# Build the test package with modified name
mkdir -p "$TEST_OUTDIR"
(cd "$TMP_DIR" && python -m build --outdir "$TEST_OUTDIR")

# Cleanup temp directory
rm -rf "$TMP_DIR"

echo ""
echo "✅ Build completed!"
echo ""
echo "Generated files (dist/):"
ls -lh dist/ || true
echo ""
echo "Generated files for TestPyPI (dist-test/):"
ls -lh "$TEST_OUTDIR" || true

