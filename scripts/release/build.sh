#!/bin/bash
# Build distribution packages
# Usage: ./scripts/release/build.sh

set -e

# Published universal wheels must include the complete optional Linux Runtime,
# even if the shell inherited a developer-only lightweight build override.
export ORCA_SENSOR_BUNDLE_NATIVE=1

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
echo "📦 Building TestPyPI package (orca-gym)..."

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

# Keep the same package name (orca-gym) for TestPyPI build
# This allows for complete installation testing with the same package name

# Build the test package with same name
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
