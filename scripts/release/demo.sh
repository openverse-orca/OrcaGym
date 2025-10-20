#!/bin/bash
# Demo script to showcase the release workflow
# This is for demonstration purposes only - NOT for actual releases
# Usage: ./scripts/release/demo.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          OrcaGym Core - Release Workflow Demo               ║"
echo "║                                                              ║"
echo "║  This script demonstrates the complete release process      ║"
echo "║  WITHOUT actually uploading to PyPI                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to pause and show next step
pause_demo() {
    echo ""
    echo "Press Enter to continue to next step..."
    read -r
    echo ""
}

# Step 1: Show current version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 1: Check current version"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CURRENT_VERSION=$(grep -Po '(?<=version = ")[^"]*' pyproject.toml)
echo "Current version: $CURRENT_VERSION"
pause_demo

# Step 2: Show available scripts
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Step 2: Available release scripts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -1 scripts/release/*.sh | xargs -n1 basename
pause_demo

# Step 3: Clean
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧹 Step 3: Cleaning build artifacts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$ ./scripts/release/clean.sh"
./scripts/release/clean.sh
pause_demo

# Step 4: Build
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Step 4: Building distribution packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$ ./scripts/release/build.sh"
./scripts/release/build.sh
pause_demo

# Step 5: Check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Step 5: Checking package quality"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$ ./scripts/release/check.sh"
./scripts/release/check.sh
pause_demo

# Step 6: Show package contents
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 Step 6: Package contents preview"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Built packages:"
ls -lh dist/
echo ""
echo "Preview wheel contents (first 20 files):"
unzip -l dist/*.whl | head -25
pause_demo

# Step 7: Show what would happen next
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Step 7: Upload process (simulation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To upload to TestPyPI, you would run:"
echo "  $ ./scripts/release/upload_test.sh"
echo "  or"
echo "  $ make release-test"
echo ""
echo "To upload to PyPI (production), you would run:"
echo "  $ ./scripts/release/upload_prod.sh"
echo "  or"
echo "  $ make release-prod"
echo ""
echo "⚠️  NOTE: This demo does NOT upload anything."
pause_demo

# Step 8: Show Makefile commands
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ Step 8: Convenient Makefile commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
make help
pause_demo

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Demo completed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Summary of release scripts:"
echo ""
echo "  Individual scripts:"
echo "    • ./scripts/release/clean.sh       - Clean build artifacts"
echo "    • ./scripts/release/build.sh       - Build packages"
echo "    • ./scripts/release/check.sh       - Check package quality"
echo "    • ./scripts/release/upload_test.sh - Upload to TestPyPI"
echo "    • ./scripts/release/upload_prod.sh - Upload to PyPI"
echo ""
echo "  Workflow scripts:"
echo "    • ./scripts/release/release.sh test  - Complete release to TestPyPI"
echo "    • ./scripts/release/release.sh prod  - Complete release to PyPI"
echo ""
echo "  Utility scripts:"
echo "    • ./scripts/release/bump_version.sh <version>  - Update version"
echo "    • ./scripts/release/test_install.sh <source>   - Test installation"
echo ""
echo "  Makefile shortcuts:"
echo "    • make release-test    - Quick release to TestPyPI"
echo "    • make release-prod    - Quick release to PyPI"
echo "    • make help            - Show all available commands"
echo ""
echo "📚 Documentation:"
echo "    • scripts/release/README.md           - Full documentation"
echo "    • scripts/release/QUICK_REFERENCE.md  - Quick reference"
echo "    • PYPI_RELEASE.md                     - PyPI release guide"
echo ""
echo "🔗 Next steps:"
echo "    1. Configure your PyPI credentials (~/.pypirc)"
echo "    2. Update version: make bump-version VERSION=x.y.z"
echo "    3. Test release: make release-test"
echo "    4. Production release: make release-prod"
echo ""

