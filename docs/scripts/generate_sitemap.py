import os
import datetime
from pathlib import Path

def generate_sitemap():
    """
    Generates a sitemap.xml for the Nabla documentation website.
    """
    print("🗺️  Generating sitemap...")

    # Configuration
    docs_dir = Path(__file__).parent.parent
    build_dir = docs_dir / "_build" / "html"
    base_url = "https://www.nablaml.com"
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

        # Get relative path and construct URL
        relative_path = html_file.relative_to(build_dir).as_posix()
        url = f"{base_url}/{relative_path}"

        # Get last modification time
        last_mod_ts = os.path.getmtime(html_file)
        last_mod = datetime.datetime.fromtimestamp(last_mod_ts).strftime('%Y-%m-%d')

        sitemap_content += "  <url>\n"
        sitemap_content += f"    <loc>{url}</loc>\n"
        sitemap_content += f"    <lastmod>{last_mod}</lastmod>\n"
        sitemap_content += "    <changefreq>monthly</changefreq>\n"
        sitemap_content += "    <priority>0.8</priority>\n"
        sitemap_content += "  </url>\n"

    sitemap_content += "</urlset>\n"

    # Write the sitemap file
    with open(sitemap_path, "w", encoding="utf-8") as f:
        f.write(sitemap_content)

    print(f"✅ Sitemap generated with {len(html_files)} URLs at: {sitemap_path}")

if __name__ == "__main__":
    generate_sitemap()
