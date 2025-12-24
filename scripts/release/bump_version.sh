#!/bin/bash
# Bump version number in pyproject.toml
# Usage: ./scripts/release/bump_version.sh <new_version>

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

NEW_VERSION="$1"

if [ -z "$NEW_VERSION" ]; then
    echo "❌ Error: Version number required"
    echo "Usage: $0 <new_version>"
    echo "Example: $0 25.10.1"
    exit 1
fi

# Validate version format (basic check)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "⚠️  Warning: Version format doesn't match X.Y.Z pattern"
    read -p "Continue anyway? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "❌ Version update cancelled."
        exit 1
    fi
fi

# Get current version
CURRENT_VERSION=$(grep -Po '(?<=version = ")[^"]*' pyproject.toml)

echo "Current version: $CURRENT_VERSION"
echo "New version: $NEW_VERSION"
echo ""

# Confirm
read -p "Update version? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Version update cancelled."
    exit 1
fi

# Update pyproject.toml
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
else
    # Linux
    sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
fi

echo ""
echo "✅ Version updated to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "1. Review the change: git diff pyproject.toml"
echo "2. Commit: git commit -am 'Bump version to $NEW_VERSION'"
echo "3. Tag: git tag -a v$NEW_VERSION -m 'Release v$NEW_VERSION'"
echo "4. Push: git push && git push --tags"
echo "5. Release: ./scripts/release/release.sh test"

