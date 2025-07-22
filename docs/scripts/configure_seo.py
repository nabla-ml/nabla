#!/usr/bin/env python3
"""
Environment-aware SEO configuration for CI/CD builds
Sets proper domain and validates SEO implementation
"""
import os
import sys
from pathlib import Path

def configure_seo_for_environment():
    """Configure SEO settings based on environment variables"""
    
    # Get base URL from environment or default
    base_url = os.environ.get('DOCS_BASE_URL', 'https://nablaml.com')
    is_ci = os.environ.get('CI', 'false').lower() == 'true'
    
    print(f"🌐 Configuring SEO for: {base_url}")
    print(f"📦 CI Environment: {'Yes' if is_ci else 'No'}")
    
    # Update sitemap generator with environment-specific URL
    sitemap_script = Path(__file__).parent / "generate_sitemap.py"
    if sitemap_script.exists():
        with open(sitemap_script, 'r') as f:
            content = f.read()
        
        # Replace base_url in the script
        updated_content = content.replace(
            'base_url = "https://nablaml.com"',
            f'base_url = "{base_url}"'
        )
        
        with open(sitemap_script, 'w') as f:
            f.write(updated_content)
            
        print("✅ Updated sitemap generator with environment URL")
    
    return base_url

def validate_seo_consistency(html_dir_path: str, expected_domain: str):
    """Validate that all SEO elements use consistent domain"""
    html_dir = Path(html_dir_path)
    issues = []
    
    if not html_dir.exists():
        issues.append(f"HTML directory does not exist: {html_dir}")
        return issues
    
    # Check sitemap
    sitemap_file = html_dir / "sitemap.xml"
    if sitemap_file.exists():
        sitemap_content = sitemap_file.read_text()
        expected_base = expected_domain.rstrip('/')
        if expected_base not in sitemap_content:
            issues.append(f"Sitemap doesn't contain expected domain: {expected_base}")
    else:
        # Sitemap might be in _static during build process
        static_sitemap = html_dir / "_static" / "sitemap.xml"
        if static_sitemap.exists():
            print(f"📝 Note: Found sitemap in _static directory, this is expected during build")
        else:
            issues.append("Sitemap not found")
    
    # Check robots.txt
    robots_file = html_dir / "robots.txt"
    if not robots_file.exists():
        static_robots = html_dir / "_static" / "robots.txt"
        if static_robots.exists():
            print(f"📝 Note: Found robots.txt in _static directory, this is expected during build")
        else:
            issues.append("Robots.txt not found")
    
    # Check main pages for canonical URLs (only if HTML files exist)
    index_file = html_dir / "index.html"
    if index_file.exists():
        content = index_file.read_text()
        # Check if canonical URL contains the expected domain (flexible matching)
        import re
        canonical_match = re.search(r'rel="canonical" href="([^"]*)"', content)
        if canonical_match:
            actual_canonical = canonical_match.group(1)
            expected_base = expected_domain.rstrip('/')
            if not actual_canonical.startswith(expected_base):
                issues.append(f"Homepage canonical URL domain mismatch. Expected: {expected_base}, Found: {actual_canonical}")
        else:
            issues.append(f"Homepage missing canonical URL")
    else:
        print(f"📝 Note: index.html not found in {html_dir}, skipping canonical URL check")
    
    return issues

if __name__ == "__main__":
    html_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html"
    
    # Configure environment
    base_url = configure_seo_for_environment()
    
    # Validate consistency if HTML dir exists
    if Path(html_dir).exists():
        print(f"🔍 Validating SEO consistency in {html_dir}")
        issues = validate_seo_consistency(html_dir, base_url)
        
        if issues:
            print("❌ SEO consistency issues found:")
            for issue in issues:
                print(f"  • {issue}")
            sys.exit(1)
        else:
            print("✅ All SEO elements are consistent!")
    
    print("🎉 SEO configuration complete!")
