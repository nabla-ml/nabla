import os
import datetime
import sys
from pathlib import Path

# Add docs directory to sys.path to import conf
docs_dir = Path(__file__).parent.parent
sys.path.insert(0, str(docs_dir))

from conf import html_baseurl

def get_priority(url, base_url):
    """Determine the priority of a URL based on content type and depth."""
    # Homepage
    if url == f"{base_url}/index.html" or url == f"{base_url}/":
        return "1.0"
    # Section index pages (high value for navigation)
    if url.endswith("/index.html"):
        if "/api/index.html" in url or "/examples/index.html" in url:
            return "0.9"
        # Module-level index pages (nn, ops, transforms, core)
        if any(s in url for s in ["/nn/index", "/ops/index", "/transforms/index", "/core/index"]):
            return "0.85"
    # High-value feature pages: finetune (LoRA/QLoRA), optimizers, sharding
    if any(s in url for s in ["finetune", "nn_optim", "sharding"]):
        return "0.8"
    # Examples (evergreen content, high search value)
    if "examples/" in url:
        return "0.8"
    # API reference pages
    if "api/" in url:
        return "0.7"
    return "0.5"

def generate_sitemap():
    """
    Generates a sitemap.xml for the Nabla documentation website.
    """
    print("🗺️  Generating sitemap...")

    # Configuration
    build_dir = docs_dir / "_build" / "html"
    base_url = html_baseurl.rstrip('/')  # Use base URL from conf.py
    sitemap_path = build_dir / "sitemap.xml"


    if not build_dir.exists():
        print(f"❌ Build directory not found at: {build_dir}")
        return

    # Find all HTML files
    html_files = list(build_dir.rglob("*.html"))

    # Start XML sitemap
    sitemap_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    sitemap_content += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'

    # Exclude list
    exclude_files = ["search.html", "genindex.html", "objects.inv"]

    # Add each HTML file to the sitemap
    for html_file in html_files:
        if html_file.name in exclude_files:
            continue
        
        # Exclude files in _static directory
        if "_static" in html_file.relative_to(build_dir).parts:
            continue

        # Get relative path and construct URL
        relative_path = html_file.relative_to(build_dir).as_posix()
        url = f"{base_url}/{relative_path}"

        # Get last modification time
        last_mod_ts = os.path.getmtime(html_file)
        last_mod = datetime.datetime.fromtimestamp(last_mod_ts).strftime('%Y-%m-%d')

        sitemap_content += "  <url>\n"
        sitemap_content += f"    <loc>{url}</loc>\n"
        sitemap_content += f"    <lastmod>{last_mod}</lastmod>\n"
        priority = get_priority(url, base_url)
        sitemap_content += "    <changefreq>monthly</changefreq>\n"
        sitemap_content += f"    <priority>{priority}</priority>\n"
        sitemap_content += "  </url>\n"

    sitemap_content += "</urlset>\n"

    # Write the sitemap file
    with open(sitemap_path, "w", encoding="utf-8") as f:
        f.write(sitemap_content)

    print(f"✅ Sitemap generated with {len(html_files)} URLs at: {sitemap_path}")

if __name__ == "__main__":
    generate_sitemap()
