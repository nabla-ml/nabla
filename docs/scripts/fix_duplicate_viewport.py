#!/usr/bin/env python3
"""
Script to fix duplicate viewport tags in generated HTML files.
This is a common issue when extending Sphinx themes.
"""

import glob
import os
import re


def fix_duplicate_viewport_tags(html_dir: str):
    """Fix duplicate viewport meta tags in HTML files."""

    html_files = glob.glob(os.path.join(html_dir, "**/*.html"), recursive=True)
    fixed_count = 0

    for html_file in html_files:
        with open(html_file, encoding="utf-8") as f:
            content = f.read()

        # Find and fix duplicate viewport tags
        original_content = content

        # Pattern to match multiple viewport meta tags on the same line or separate lines
        viewport_pattern = r'<meta name="viewport"[^>]*>\s*<meta name="viewport"[^>]*>'

        # Replace with a single, proper viewport tag
        content = re.sub(
            viewport_pattern,
            '<meta name="viewport" content="width=device-width, initial-scale=1.0" />',
            content,
        )

        # Also handle case where they're on separate lines
        content = re.sub(
            r'(<meta name="viewport"[^>]*>)\s*\n\s*(<meta name="viewport"[^>]*>)',
            r'<meta name="viewport" content="width=device-width, initial-scale=1.0" />',
            content,
        )

        if content != original_content:
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(content)
            fixed_count += 1
            print(f"✅ Fixed {html_file}")

    print(f"🎯 Fixed duplicate viewport tags in {fixed_count} HTML files")
    return fixed_count


if __name__ == "__main__":
    import sys

    html_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html"

    if not os.path.exists(html_dir):
        print(f"❌ HTML directory not found: {html_dir}")
        sys.exit(1)

    print(f"🔧 Fixing duplicate viewport tags in {html_dir}")
    fix_duplicate_viewport_tags(html_dir)
    print("✨ All done!")
