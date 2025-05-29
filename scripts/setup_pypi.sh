#!/bin/bash
# Setup script for PyPI credentials
# Run this once to configure your PyPI API token

echo "🔐 PyPI Token Setup for nabla-ml"
echo "================================="
echo ""
echo "To publish to PyPI, you need to set up your API token."
echo "Get your token from: https://pypi.org/manage/account/token/"
echo ""

# Check if already configured
if [[ -n "$TWINE_PASSWORD" && "$TWINE_USERNAME" == "__token__" ]]; then
    echo "✅ PyPI credentials are already configured in your environment"
    echo "Username: $TWINE_USERNAME"
    echo "Token: ${TWINE_PASSWORD:0:10}..."
    exit 0
fi

echo "Choose setup method:"
echo "1) Environment variables (temporary - this session only)"
echo "2) Add to shell profile (permanent)"
echo "3) Show manual setup instructions"
echo ""

read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        read -p "Enter your PyPI token (pypi-...): " token
        export TWINE_USERNAME="__token__"
        export TWINE_PASSWORD="$token"
        echo ""
        echo "✅ Credentials set for this session!"
        echo "Run the release script now to use them."
        ;;
    2)
        echo ""
        read -p "Enter your PyPI token (pypi-...): " token
        echo ""
        echo "Adding to your shell profile..."
        
        # Detect shell and add to appropriate file
        if [[ "$SHELL" == *"zsh"* ]]; then
            profile_file="$HOME/.zshrc"
        elif [[ "$SHELL" == *"bash"* ]]; then
            profile_file="$HOME/.bash_profile"
        else
            profile_file="$HOME/.profile"
        fi
        
        echo "" >> "$profile_file"
        echo "# PyPI credentials for nabla-ml" >> "$profile_file"
        echo "export TWINE_USERNAME=\"__token__\"" >> "$profile_file"
        echo "export TWINE_PASSWORD=\"$token\"" >> "$profile_file"
        
        echo "✅ Added to $profile_file"
        echo "Restart your terminal or run: source $profile_file"
        ;;
    3)
        echo ""
        echo "Manual setup instructions:"
        echo "========================="
        echo ""
        echo "Option A: Environment variables"
        echo "export TWINE_USERNAME=\"__token__\""
        echo "export TWINE_PASSWORD=\"your-pypi-token-here\""
        echo ""
        echo "Option B: ~/.pypirc file"
        echo "[pypi]"
        echo "username = __token__"
        echo "password = your-pypi-token-here"
        echo ""
        echo "Option C: Keyring (most secure)"
        echo "pip install keyring"
        echo "keyring set https://upload.pypi.org/legacy/ __token__"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
