#!/bin/bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()

    # Check for required tools
    for tool in git python uv; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    print_success "All prerequisites met"
}

# Get version from pyproject.toml
get_current_version() {
    grep '^version = ' pyproject.toml | sed 's/version = "//' | sed 's/".*//'
}

# Prompt for new version
prompt_for_version() {
    local current_version=$(get_current_version)
    echo ""
    print_info "Current version: $current_version"
    echo -n "Enter new version (or press Enter to keep current): "
    read -r new_version

    if [ -z "$new_version" ]; then
        new_version="$current_version"
    fi

    echo "$new_version"
}

# Update version in pyproject.toml
update_version() {
    local version=$1
    sed -i.bak "s/^version = .*/version = \"$version\"/" pyproject.toml
    rm -f pyproject.toml.bak
    print_success "Updated version to $version in pyproject.toml"
}

# Check git status
check_git_status() {
    if ! git diff-index --quiet HEAD --; then
        print_error "Working tree has uncommitted changes. Please commit or stash them."
        exit 1
    fi
}

# Create and push git tag
create_and_push_tag() {
    local version=$1
    local tag="v$version"

    print_info "Creating git tag: $tag"

    if git rev-parse "$tag" &> /dev/null; then
        print_error "Tag $tag already exists"
        exit 1
    fi

    git tag -a "$tag" -m "Release version $version"
    print_success "Created tag $tag"

    print_info "Pushing tag to remote..."
    git push origin "$tag"
    print_success "Pushed tag $tag to remote"
}

# Build distribution
build_distribution() {
    print_info "Building distribution..."

    # Clean previous builds
    rm -rf dist/ build/ *.egg-info 2>/dev/null || true

    # Build using uvx
    uvx --from build pyproject-build
    print_success "Built distribution packages"
}

# Prompt for PyPI credentials
prompt_for_credentials() {
    echo ""
    echo "PyPI Upload Credentials"
    echo "----------------------"
    echo "Choose authentication method:"
    echo "1. PyPI token (recommended)"
    echo "2. Username and password"
    echo ""
    read -p "Enter choice (1 or 2): " auth_choice

    if [ "$auth_choice" = "1" ]; then
        read -sp "Enter PyPI token: " pypi_token
        echo ""
        export TWINE_PASSWORD="$pypi_token"
        export TWINE_USERNAME="__token__"
    elif [ "$auth_choice" = "2" ]; then
        read -p "Enter PyPI username: " pypi_username
        read -sp "Enter PyPI password: " pypi_password
        echo ""
        export TWINE_USERNAME="$pypi_username"
        export TWINE_PASSWORD="$pypi_password"
    else
        print_error "Invalid choice"
        exit 1
    fi
}

# Upload to PyPI
upload_to_pypi() {
    print_info "Uploading to PyPI..."

    if [ -z "${TWINE_USERNAME:-}" ] || [ -z "${TWINE_PASSWORD:-}" ]; then
        print_error "PyPI credentials not set"
        exit 1
    fi

    uvx twine upload dist/*
    print_success "Successfully uploaded to PyPI"
}

# Main release workflow
main() {
    echo ""
    echo "===================================="
    echo "   Interlace Release Script"
    echo "===================================="
    echo ""

    # Step 1: Check prerequisites
    print_info "Checking prerequisites..."
    check_prerequisites
    echo ""

    # Step 2: Check git status
    print_info "Checking git status..."
    check_git_status
    print_success "Working tree is clean"
    echo ""

    # Step 3: Get version
    version=$(prompt_for_version)
    echo ""

    # Step 4: Confirm before proceeding
    echo "Release Summary:"
    echo "  Version: $version"
    echo "  Actions:"
    echo "    1. Update version in pyproject.toml"
    echo "    2. Commit changes"
    echo "    3. Create and push git tag"
    echo "    4. Build distribution"
    echo "    5. Upload to PyPI"
    echo ""
    read -p "Proceed with release? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_warning "Release cancelled"
        exit 0
    fi
    echo ""

    # Step 5: Update version
    update_version "$version"

    # Step 6: Commit version change
    print_info "Committing version update..."
    git add pyproject.toml
    git commit -m "Release version $version"
    print_success "Committed version update"
    echo ""

    # Step 7: Create and push tag
    create_and_push_tag "$version"
    echo ""

    # Step 8: Build distribution
    build_distribution
    echo ""

    # Step 9: Prompt for credentials and upload
    prompt_for_credentials
    echo ""
    upload_to_pypi

    echo ""
    print_success "Release $version completed successfully!"
    echo ""
}

# Run main
main "$@"
