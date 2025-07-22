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
    
    # Check sitemap
    sitemap_file = html_dir / "sitemap.xml"
    if sitemap_file.exists():
        sitemap_content = sitemap_file.read_text()
        if expected_domain not in sitemap_content:
            issues.append(f"Sitemap doesn't contain expected domain: {expected_domain}")
    else:
        issues.append("Sitemap not found")
    
    # Check robots.txt
    robots_file = html_dir / "robots.txt"
    if robots_file.exists():
        robots_content = robots_file.read_text()
        if expected_domain not in robots_content:
            issues.append(f"Robots.txt doesn't contain expected domain: {expected_domain}")
    else:
        issues.append("Robots.txt not found")
    
    # Check main pages for canonical URLs
    for html_file in html_dir.glob("*.html"):
        if html_file.name in ['index.html']:
            content = html_file.read_text()
            if f'rel="canonical" href="{expected_domain}"' not in content:
                issues.append(f"Homepage missing proper canonical URL for {expected_domain}")
    
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
