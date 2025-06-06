#!/bin/bash
set -e

# Documentation build script for Endia
# Handles both local development and CI/CD builds

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOCS_DIR"

# Check if we're in CI/CD or local development
if [[ "${CI:-false}" == "true" || "${GITHUB_ACTIONS:-false}" == "true" ]]; then
    echo "🔧 Building documentation for CI/CD (minimal mode)"
    BUILD_MODE="ci"
else
    echo "🔧 Building documentation for local development"
    BUILD_MODE="local"
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf _build/html
rm -rf api/generated/*.rst 2>/dev/null || true

# Install requirements if needed
if [[ "$BUILD_MODE" == "local" ]]; then
    echo "📦 Checking documentation dependencies..."
    pip install -r requirements.txt -q
fi

# Build documentation
echo "📚 Building documentation..."

if [[ "$BUILD_MODE" == "ci" ]]; then
    # CI build: minimal, no autosummary, mock imports
    sphinx-build -b html -W --keep-going \
        -D autosummary_generate=False \
        -D autodoc_mock_imports="max,max.dtype,max.graph,max.tensor,mojo,numpy,jax,torch" \
        . _build/html
else
    # Local build: try to generate API docs if endia is available
    if python -c "import endia" 2>/dev/null; then
        echo "✨ Generating API documentation..."
        python ../scripts/generate_api_docs.py
        sphinx-build -b html -W --keep-going . _build/html
    else
        echo "⚠️  Endia not installed, building without API generation"
        sphinx-build -b html -W --keep-going \
            -D autosummary_generate=False \
            . _build/html
    fi
fi

# Ensure .nojekyll for GitHub Pages
touch _build/html/.nojekyll

echo "✅ Documentation build complete!"
echo "📂 Output: $(pwd)/_build/html"

if [[ "$BUILD_MODE" == "local" ]]; then
    echo "🌐 Open: file://$(pwd)/_build/html/index.html"
fi
