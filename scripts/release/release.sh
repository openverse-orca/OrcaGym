#!/bin/bash
# Complete release workflow
# Usage: ./scripts/release/release.sh [test|prod]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

# Parse arguments
TARGET="${1:-test}"

if [[ "$TARGET" != "test" && "$TARGET" != "prod" ]]; then
    echo "❌ Error: Invalid target. Use 'test' or 'prod'"
    echo "Usage: $0 [test|prod]"
    exit 1
fi

echo "================================"
echo "  OrcaGym Core Release Process"
echo "================================"
echo ""
echo "Target: $TARGET"
echo ""

# Step 1: Clean
echo "Step 1/4: Cleaning..."
bash "$SCRIPT_DIR/clean.sh"
echo ""

# Step 2: Build
echo "Step 2/4: Building..."
bash "$SCRIPT_DIR/build.sh"
echo ""

# Step 3: Check
echo "Step 3/4: Checking..."
bash "$SCRIPT_DIR/check.sh"
echo ""

# Step 4: Upload
echo "Step 4/4: Uploading..."
if [ "$TARGET" = "test" ]; then
    bash "$SCRIPT_DIR/upload_test.sh"
else
    bash "$SCRIPT_DIR/upload_prod.sh"
fi

echo ""
echo "================================"
echo "  ✅ Release Completed!"
echo "================================"

