#!/usr/bin/env python3
"""
SEO Verification Script for Nabla Documentation
Checks all implemented SEO features to ensure they're working correctly.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and print result."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False


def check_html_content(filepath, patterns, description):
    """Check if HTML file contains specific patterns."""
    if not Path(filepath).exists():
        print(f"❌ {description}: File not found - {filepath}")
        return False

    with Path(filepath).open(encoding="utf-8") as f:
        content = f.read()

    found_patterns = []
    missing_patterns = []

    for pattern_name, pattern in patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            found_patterns.append(pattern_name)
        else:
            missing_patterns.append(pattern_name)

    if missing_patterns:
        print(f"⚠️  {description}:")
        for pattern in found_patterns:
            print(f"    ✅ {pattern}")
        for pattern in missing_patterns:
            print(f"    ❌ {pattern}")
        return False
    else:
        print(f"✅ {description}: All patterns found")
        return True


def check_sitemap(filepath):
    """Check sitemap.xml structure and content."""
    if not Path(filepath).exists():
        print(f"❌ Sitemap: {filepath} (NOT FOUND)")
        return False

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Check namespace
        if "sitemaps.org" not in root.tag:
            print(f"❌ Sitemap: Invalid namespace in {filepath}")
            return False

        # Count URLs
        urls = root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url")
        url_count = len(urls)

        if url_count > 0:
            print(f"✅ Sitemap: {url_count} URLs found in {filepath}")
            return True
        else:
            print(f"❌ Sitemap: No URLs found in {filepath}")
            return False

    except ET.ParseError as e:
        print(f"❌ Sitemap: XML parsing error in {filepath}: {e}")
        return False


def main():
    """Run comprehensive SEO verification."""
    print("🔍 Nabla Documentation SEO Verification")
    print("=" * 50)

    # Paths
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "html"
    static_dir = docs_dir / "_static"

    all_checks_passed = True

    # 1. Check essential files
    print("\n📁 Essential Files:")
    files_to_check = [
        (static_dir / "favicon.svg", "Favicon"),
        (static_dir / "robots.txt", "Robots.txt"),
        (static_dir / "seo.js", "SEO JavaScript"),
        (docs_dir / "_templates" / "layout.html", "Custom template"),
        (build_dir / "sitemap.xml", "Generated sitemap"),
        (build_dir / "index.html", "Generated homepage"),
    ]

    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # 2. Check robots.txt content
    print("\n🤖 Robots.txt Content:")
    robots_patterns = {
        "User-agent directive": r"User-agent:\s*\*",
        "Allow root": r"Allow:\s*/",
        "Sitemap location": r"Sitemap:\s*https://nablaml\.com/sitemap\.xml",
        "Block static files": r"Disallow:\s*/_static/",
    }

    if not check_html_content(
        static_dir / "robots.txt", robots_patterns, "Robots.txt patterns"
    ):
        all_checks_passed = False

    # 3. Check HTML meta tags
    print("\n🏷️  HTML Meta Tags:")
    meta_patterns = {
        "Description meta": r'<meta name="description" content="[^"]*GPU-accelerated array computation[^"]*"',
        "Keywords meta": r'<meta name="keywords" content="[^"]*python[^"]*arrays[^"]*"',
        "Robots meta": r'<meta name="robots" content="index, follow"',
        "Open Graph title": r'<meta property="og:title" content="[^"]*Nabla[^"]*"',
        "Open Graph description": r'<meta property="og:description" content="[^"]*"',
        "Twitter card": r'<meta property="twitter:card" content="summary"',
    }

    if not check_html_content(
        build_dir / "index.html", meta_patterns, "Meta tags in index.html"
    ):
        all_checks_passed = False

    # 4. Check sitemap
    print("\n🗺️  Sitemap Verification:")
    if not check_sitemap(build_dir / "sitemap.xml"):
        all_checks_passed = False

    # 5. Check structured data in SEO.js
    print("\n📊 Structured Data:")
    seo_js_patterns = {
        "JSON-LD script": r'script\.type\s*=\s*["\']application/ld\+json["\']',
        "Schema.org context": r'"@context":\s*"https://schema\.org"',
        "SoftwareApplication type": r'"@type":\s*"SoftwareApplication"',
        "Application name": r'"name":\s*"Nabla"',
    }

    if not check_html_content(
        static_dir / "seo.js", seo_js_patterns, "Structured data in seo.js"
    ):
        all_checks_passed = False

    # 6. Check favicon
    print("\n🖼️  Favicon:")
    if (static_dir / "favicon.svg").exists():
        with (static_dir / "favicon.svg").open() as f:
            favicon_content = f.read()

        if "<svg" in favicon_content and "viewBox" in favicon_content:
            print("✅ Favicon: Valid SVG format")
        else:
            print("❌ Favicon: Invalid SVG format")
            all_checks_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 SEO Verification PASSED: All checks successful!")
        print("📈 Your Nabla documentation is optimized for search engines.")
        return 0
    else:
        print("⚠️  SEO Verification FAILED: Some checks failed.")
        print("🔧 Please review the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
